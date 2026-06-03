import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from VQVAE2.VQVAE import VQVAE
from Utils import TextEncoder, AudioEncoder, TimestepEncoding, TransformerLayer, DiffusionTransformerLayer


def sample_top_k(logits, k):
    values, indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probs.view(-1, k), 1)
    return indices.view(-1, k).gather(-1, sampled).view(logits.shape[:-1])


class ConditionList(list):
    def __init__(self, name, *args):
        super().__init__(*args)
        self.name = name


class DiffusionTransformerDenoiser(nn.Module):
    def __init__(self, name, embed_dim, num_heads, num_embeddings,
                 num_layers_temporal, num_layers_semantic, num_layers):
        super(DiffusionTransformerDenoiser, self).__init__()
        # 嵌入层，用于将离散token索引转换为连续向量
        self.token_embedding = nn.Embedding(num_embeddings + 1, embed_dim)  # for mask token
        self.timestep_encoder = TimestepEncoding(embed_dim)
        # 分类头，用于预测离散token的分布
        self.classifier = nn.Linear(embed_dim, num_embeddings)

        # DiT去噪器
        self.dit_encoder_1 = DiffusionTransformerLayer(embed_dim, num_heads, num_layers_temporal)
        if name == 'top':
            self.dit_encoder_2 = DiffusionTransformerLayer(embed_dim, num_heads, num_layers_semantic)
        if name == 'bottom':
            self.dit_encoder_2 = DiffusionTransformerLayer(embed_dim, num_heads, num_layers_temporal)
        self.dit_encoder_3 = DiffusionTransformerLayer(embed_dim, num_heads, num_layers_semantic)
        self.dit_encoder = DiffusionTransformerLayer(embed_dim, num_heads, num_layers)

    def forward(self, noisy_vectors, conditions, timesteps):
        a = self.token_embedding(noisy_vectors)  # (B, seq_len, embed_dim)
        t_emb = self.timestep_encoder(timesteps)  # (B, d_model)

        b = self.dit_encoder_1(a + conditions[2], t_emb)
        if conditions.name == 'top':
            c = self.dit_encoder_2(b + conditions[2], t_emb + conditions[1])
            d = self.dit_encoder_3(c + conditions[2], t_emb + sum(conditions[:2]))
        if conditions.name == 'bottom':
            x = sum(conditions[1:])
            c = self.dit_encoder_2(b + x, t_emb)
            d = self.dit_encoder_3(c + x, t_emb + conditions[0])
        features = self.dit_encoder(d + a, t_emb)

        logits = self.classifier(features)  # (B, seq_len, num_embeddings)
        return logits


class FaceGenerationModel(nn.Module):
    def __init__(self, vqvae_dir, embed_dim,
                 num_heads1, num_layers_style, num_layers_top1, num_layers_bottom1, num_embeddings,
                 num_heads2, num_layers_temporal, num_layers_semantic, num_layers,
                 gpu, num_diffusion_timesteps=1000, temperature=1.0):
        super(FaceGenerationModel, self).__init__()
        # 加载VQVAE
        self.vqvae = VQVAE(embed_dim, num_heads1, num_layers_style, num_layers_top1, num_layers_bottom1,
                           num_embeddings)
        self.vqvae.load_state_dict(torch.load(vqvae_dir, map_location="cpu")['model_state_dict'])
        for param in self.vqvae.parameters():
            param.requires_grad = False
        # 编码器
        self.person_proj1 = nn.Linear(46, embed_dim)
        self.person_proj2 = nn.Linear(46, embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim)
        self.top_proj = nn.Linear(embed_dim, embed_dim)
        # 离散扩散去噪器
        self.top_denoiser = DiffusionTransformerDenoiser('top', embed_dim, num_heads2, num_embeddings,
                                                         num_layers_temporal, num_layers_semantic, num_layers)
        self.bottom_denoiser = DiffusionTransformerDenoiser('bottom', embed_dim, num_heads2, num_embeddings,
                                                            num_layers_temporal, num_layers_semantic, num_layers)
        # 参数
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.num_timesteps = num_diffusion_timesteps
        self.temperature = temperature
        # 使用交叉熵损失
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion_mse = nn.MSELoss(reduction='none')
        # cosine掩码调度
        t = torch.linspace(0, 1, num_diffusion_timesteps)
        mask_prob = torch.sin(t * math.pi / 2) ** 2
        self.register_buffer('mask_prob', mask_prob)

    def discrete_forward_diffusion(self, token_indices, timesteps, mask_prob_schedule, num_embeddings):
        """离散扩散的前向过程：逐步掩码token"""
        batch_size, seq_len = token_indices.shape
        # 获取当前时间步的掩码概率
        mask_prob = mask_prob_schedule[timesteps].view(-1, 1)  # (B, 1)
        # 生成掩码：1表示掩码，0表示保留
        mask = torch.bernoulli(mask_prob.expand(batch_size, seq_len)).bool()
        # 创建掩码后的token：掩码位置设为num_embeddings（特殊掩码token）
        noisy_tokens = torch.where(mask, torch.full_like(token_indices, num_embeddings), token_indices)
        return noisy_tokens, mask

    def prepare_conditions(self, person_one_hot, text, audio, q_top=None):
        # 编码条件
        person_condition1 = self.person_proj1(person_one_hot)
        person_condition2 = self.person_proj2(person_one_hot)
        #person_condition1 = torch.zeros_like(person_condition1)
        #person_condition2 = torch.zeros_like(person_condition2)
        text_features = self.text_encoder(text).squeeze(1)
        #text_features = torch.zeros_like(text_features)
        audio_condition = self.audio_encoder(audio)

        # 顶层底层条件
        top_conditions = ConditionList('top', [person_condition1, text_features, audio_condition])
        if q_top is not None:
            top_condition = self.top_proj(q_top)
            #top_condition = torch.zeros_like(top_condition)
            bottom_conditions = ConditionList('bottom', [person_condition2, top_condition, audio_condition])
            return top_conditions, bottom_conditions
        else:
            return top_conditions

    def compute_loss(self, person_one_hot, text, audio, exp, jaw, mask):
        # 获取真实的离散token索引
        style = self.vqvae.style_encoder(person_one_hot, exp, jaw)
        z_top = self.vqvae.top_encoder(text, style)
        _, q_top = self.vqvae.vq_layer_top(z_top)
        z_bottom = self.vqvae.bottom_encoder(q_top, style)
        _, q_bottom = self.vqvae.vq_layer_bottom(z_bottom)
        # 获取离散token的索引，修改VQVAE的VectorQuantizer来返回索引
        q_top_indices = self.vqvae.vq_layer_top.get_indices(z_top)
        q_bottom_indices = self.vqvae.vq_layer_bottom.get_indices(z_bottom)

        # 计算掩码
        mask_top = mask[:, ::8].bool()  # 添加.bool()确保是布尔类型
        mask_bottom = mask[:, ::8].bool()  # 添加.bool()确保是布尔类型
        # 准备条件
        batch_size = person_one_hot.shape[0]
        device = person_one_hot.device
        top_conds, bottom_conds = self.prepare_conditions(person_one_hot, text, audio, q_top)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)  # 使用统一timesteps
        # timestep-aware loss 权重
        t_weight = self.mask_prob[timesteps]  # (B,)
        t_weight = t_weight / (t_weight.mean() + 1e-6)  # 归一化，防止整体 loss scale 漂移
        # 初始化全掩码token
        noisy_top, mask_top_diff = self.discrete_forward_diffusion(
            q_top_indices, timesteps, self.mask_prob, self.num_embeddings)
        noisy_bottom, mask_bottom_diff = self.discrete_forward_diffusion(
            q_bottom_indices, timesteps, self.mask_prob, self.num_embeddings)

        # 顶层损失 - 离散扩散
        logits_top = self.top_denoiser(noisy_top, top_conds, timesteps)
        loss_mask_top = mask_top_diff & mask_top.unsqueeze(1)
        ce_loss_top = self.criterion(logits_top.permute(0, 2, 1), q_top_indices)
        loss_top = (ce_loss_top * loss_mask_top * t_weight.view(-1, 1)).sum() / (loss_mask_top.sum() + 1e-6)
        # 底层损失 - 离散扩散
        logits_bottom = self.bottom_denoiser(noisy_bottom, bottom_conds, timesteps)
        loss_mask_bottom = mask_bottom_diff & mask_bottom.unsqueeze(1)
        ce_loss_bottom = self.criterion(logits_bottom.permute(0, 2, 1), q_bottom_indices)
        loss_bottom = (ce_loss_bottom * loss_mask_bottom * t_weight.view(-1, 1)).sum() / (loss_mask_bottom.sum() + 1e-6)

        # 使用softmax近似argmax，保持梯度
        pred_top_probs = F.softmax(logits_top / self.temperature, dim=-1)
        pred_bottom_probs = F.softmax(logits_bottom / self.temperature, dim=-1)
        # 使用概率分布与嵌入矩阵相乘，而不是离散索引查找，这样操作是可微的，能够保持梯度
        pred_top_emb = torch.matmul(pred_top_probs, self.vqvae.vq_layer_top.embeddings.weight)
        pred_bottom_emb = torch.matmul(pred_bottom_probs, self.vqvae.vq_layer_bottom.embeddings.weight)
        # 计算重建损失
        pred_bottom = self.vqvae.bottom_decoder(pred_top_emb, pred_bottom_emb)
        pred_top = self.vqvae.top_decoder(text, pred_top_emb)
        pred_exp, pred_jaw = self.vqvae.style_decoder(person_one_hot, pred_top, pred_bottom)
        loss_exp = (self.criterion_mse(pred_exp, exp) * mask.unsqueeze(2)).sum() / (100 * mask.sum())
        loss_jaw = (self.criterion_mse(pred_jaw, jaw) * mask.unsqueeze(2)).sum() / (3 * mask.sum())
        return loss_top, loss_bottom, loss_exp, loss_jaw

    @torch.no_grad()
    def sample(self, person_one_hot, text, audio,
               num_sampling_steps_top=50, num_sampling_steps_bottom=50, temperature=1.0, k=5):
        batch_size = person_one_hot.shape[0]
        device = person_one_hot.device
        time_steps = 32
        # 离散扩散采样时间步
        timesteps_top = torch.linspace(self.num_timesteps - 1, 0, num_sampling_steps_top, device=device).long()
        timesteps_bottom = torch.linspace(self.num_timesteps - 1, 0, num_sampling_steps_bottom, device=device).long()
        # 初始化全掩码token
        x_top = torch.full((batch_size, time_steps), self.num_embeddings, device=device, dtype=torch.long)
        x_bottom = torch.full((batch_size, time_steps), self.num_embeddings, device=device, dtype=torch.long)

        # === 顶层离散采样 ===
        # 准备条件（使用初始掩码token）
        top_conds = self.prepare_conditions(person_one_hot, text, audio)
        for t in timesteps_top:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # 预测token分布
            logits_top = self.top_denoiser(x_top, top_conds, t_batch)
            x_top_pred = sample_top_k(logits_top / temperature, k)
            # 确定哪些位置需要更新（基于掩码概率）
            current_mask_prob = self.mask_prob[t]
            update_mask = torch.bernoulli(torch.full((batch_size, time_steps), 1.0 - current_mask_prob,
                                                     device=device)).bool()
            # 更新token：在非掩码位置使用预测值
            x_top = torch.where(update_mask, x_top_pred, x_top)
        # 最终顶层token
        q_top = self.vqvae.vq_layer_top.embeddings(x_top)

        # === 底层离散采样 ===
        # 准备条件（使用顶层token）
        _, bottom_conds = self.prepare_conditions(person_one_hot, text, audio, q_top)
        for t in timesteps_bottom:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            # 预测token分布
            logits_bottom = self.bottom_denoiser(x_bottom, bottom_conds, t_batch)
            x_bottom_pred = sample_top_k(logits_bottom / temperature, k)
            # 确定哪些位置需要更新
            current_mask_prob = self.mask_prob[t]
            update_mask = torch.bernoulli(torch.full((batch_size, time_steps), 1.0 - current_mask_prob,
                                                     device=device)).bool()
            # 更新token
            x_bottom = torch.where(update_mask, x_bottom_pred, x_bottom)
        # 最终底层token
        q_bottom = self.vqvae.vq_layer_bottom.embeddings(x_bottom)

        # 解码
        bottom = self.vqvae.bottom_decoder(q_top, q_bottom)
        top = self.vqvae.top_decoder(text, q_top)
        exp_output, jaw_output = self.vqvae.style_decoder(person_one_hot, top, bottom)
        return exp_output, jaw_output
