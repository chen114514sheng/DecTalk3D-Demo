import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import numpy as np


class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(TextEncoder, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            cache_dir=os.environ.get("OPENCLIP_CACHE_DIR"),
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')  # 将文本转换为token
        for param in self.model.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(512, embed_dim)
        )

    def forward(self, text):
        text_tokens = self.tokenizer(text).to(next(self.model.parameters()).device)
        text_features = self.model.encode_text(text_tokens).unsqueeze(1)  # (B, 1, 512)
        text_features = self.mlp(text_features)  # (B, 1, 64)
        return text_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 正弦编码和余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码添加到输入中，x 的形状是 (N, B, embed_dim)
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerLayer, self).__init__()
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer编码层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, mask=None):
        # src 的形状应为 (B, N, embed_dim)
        src = src.permute(1, 0, 2)  # 转换形状为 (N, B, embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = output.permute(1, 0, 2)  # 转换回 (B, N, embed_dim)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer解码层
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: 解码器的输入 (B, T, embed_dim)。
            memory: 编码器的输出 (B, S, embed_dim)。
        Returns:
            output: 解码器的输出 (B, T, embed_dim)。
        """
        # tgt 的形状应为 (B, T, embed_dim)
        tgt = tgt.permute(1, 0, 2)  # 转换形状为 (T, B, embed_dim)
        tgt = self.pos_encoder(tgt)
        # memory 的形状应为 (B, S, embed_dim)
        memory = memory.permute(1, 0, 2)  # 转换形状为 (S, B, embed_dim)
        # 解码器
        output = self.transformer_decoder(tgt,memory, tgt_mask, memory_mask)
        output = output.permute(1, 0, 2)  # 转换回 (B, T, embed_dim)
        return output


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 定义嵌入层（embedding layer)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 将嵌入向量（embedding vectors）的权重初始化为 -1 / self.num_embeddings 到 1 / self.num_embeddings 之间的随机数
        nn.init.uniform_(self.embeddings.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs, sample=False, temperature=0.2, k=1):  # sample=False/True, temperature=0.2, k=1
        # 展开后形状(B * N, embed_dim)
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # 计算输入向量和嵌入向量之间的距离
        distances = torch.cdist(flat_input, self.embeddings.weight)

        if not sample:
            # 确定性采样：找到最小距离的嵌入索引
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        else:
            # 随机采样
            logits = -distances  # 转换为相似度
            logits = logits / temperature  # 使用temperature缩放
            probabilities = F.softmax(logits, dim=-1)
            encoding_indices = torch.multinomial(probabilities, num_samples=k)  # 随机采样索引
            # 如果 k > 1，使用最后一个采样结果
            if k > 1:
                encoding_indices = encoding_indices[:, -1:]

        # 生成 one-hot 编码，形状为 (B * N, num_embeddings)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # 使用 one-hot 编码与嵌入矩阵相乘，得到量化后的表示
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='none')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='none')
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 保持梯度
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps

    def set_init_lr(self, init_lr):
        self.init_lr = init_lr

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Config:
    def __init__(self, shape, expression, flame_model, static_landmark_embedding, dynamic_landmark_embedding):
        # 模型文件路径 male/female/generic
        self.flame_model_path = flame_model
        # 批处理大小
        self.batch_size = 256
        # 是否使用面部轮廓
        self.use_face_contour = True
        # 形状参数数量
        self.shape_params = shape
        # 表情参数数量
        self.expression_params = expression
        # 是否使用 3D 位移
        self.use_3D_translation = True
        # 静态标志嵌入路径
        self.static_landmark_embedding_path = static_landmark_embedding
        # 动态标志嵌入路径
        self.dynamic_landmark_embedding_path = dynamic_landmark_embedding


def mve_compute(vertices_gt_all, vertices_output_all):
    # MVE (Mean Vertex Error): 所有顶点的平均差
    vertices_gt = np.array(vertices_gt_all)
    vertices_output = np.array(vertices_output_all)
    mve = np.linalg.norm(vertices_gt - vertices_output, axis=2)
    return mve


def lve_compute(vertices_gt_all, vertices_output_all, mouth_map):
    # LVE (Lips Vertex Error): 嘴唇区域的最大L2误差
    vertices_gt = np.array(vertices_gt_all)
    vertices_output = np.array(vertices_output_all)
    L2_dis_mouth = np.square(vertices_gt[:, mouth_map, :] - vertices_output[:, mouth_map, :])
    lve = np.max(np.sum(L2_dis_mouth, axis=2), axis=1)
    return lve


def fdd_compute(vertices_gt_all, vertices_output_all, upper_map, seq_template):
    # FDD (Frame Difference Diversity): 上半部分区域帧差异
    vertices_gt = np.array(vertices_gt_all)
    vertices_output = np.array(vertices_output_all)
    motion_gt_upper = vertices_gt[:, upper_map, :] - seq_template[:, upper_map, :]
    motion_pred_upper = vertices_output[:, upper_map, :] - seq_template[:, upper_map, :]
    fdd = np.abs(np.std(motion_gt_upper, axis=0) - np.std(motion_pred_upper, axis=0))
    return fdd
