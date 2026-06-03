import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from VQVAE2.VQVAE2 import VQVAE2
from Utils import TextEncoder, TransformerLayer, TransformerDecoderLayer


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers_top, num_layers_bottom):
        super(AudioEncoder, self).__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        model_dir = os.environ.get("TORCHAUDIO_MODEL_DIR")
        dl_kwargs = {"model_dir": model_dir} if model_dir else None
        self.model = bundle.get_model(dl_kwargs=dl_kwargs)
        for param in self.model.parameters():
            param.requires_grad = False

        self.squash_bottom = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        self.squash_top = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        self.expand_top = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.transformer_bottom = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers_bottom)
        self.transformer_top = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers_top)
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.style_embedding_bottom = nn.Linear(46, embed_dim)
        self.style_embedding_top = nn.Linear(46, embed_dim)
        self.text_encoder_bottom = TextEncoder(embed_dim)
        self.text_encoder_top = TextEncoder(embed_dim)
        self.mlp = nn.Linear(768, embed_dim)

    def wav2vec2(self, audio):
        Resample = torchaudio.functional.resample(audio.squeeze(-1), orig_freq=48000, new_freq=16000)
        audio_features, _ = self.model(Resample)
        audio_features = F.interpolate(audio_features.permute(0, 2, 1), size=(int(audio.shape[1] / 1920),),
                                       mode='linear', align_corners=False).permute(0, 2, 1)
        return audio_features

    def bottom(self, audio_features):
        audio_features = self.mlp(audio_features)
        audio_features = self.squash_bottom(audio_features.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N/8, 768)
        return audio_features

    def top(self, audio_features):
        audio_features = self.squash_top(audio_features.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N/16, 768)
        return audio_features

    def bottom_encoder(self, person_one_hot, text, q_top_pred, z_bottom_pred):
        q_top_pred = self.expand_top(q_top_pred.permute(0, 2, 1)).permute(0, 2, 1)
        q_top_pred = self.top_embedding(q_top_pred)
        style_features_bottom = self.style_embedding_bottom(person_one_hot.unsqueeze(1))
        text_features_bottom = self.text_encoder_bottom(text)

        z_bottom_pred = self.transformer_bottom(style_features_bottom * text_features_bottom * (q_top_pred + z_bottom_pred))
        #z_bottom_pred = self.transformer_bottom(q_top_pred + z_bottom_pred)
        #z_bottom_pred = self.transformer_bottom(style_features_bottom * text_features_bottom * z_bottom_pred)
        #z_bottom_pred = self.transformer_bottom(z_bottom_pred)
        return z_bottom_pred

    def top_encoder(self, person_one_hot, text, z_top_pred):
        style_features_top = self.style_embedding_top(person_one_hot.unsqueeze(1))
        text_features_top = self.text_encoder_top(text)

        z_top_pred = self.transformer_top(style_features_top * text_features_top * z_top_pred)
        #z_top_pred = self.transformer_top(z_top_pred)
        return z_top_pred


class FaceGenerationModel(nn.Module):
    def __init__(self, embed_dim,
                 num_heads1, num_layers_top1, num_layers_bottom1, num_layers_decoder,
                 num_embeddings_top, num_embeddings_bottom,
                 num_heads2, num_layers_top2, num_layers_bottom2):
        super(FaceGenerationModel, self).__init__()
        # 加载模型 _STtb _tb _STb _b
        self.vqvae = VQVAE2(embed_dim, num_heads1,
                            num_layers_top1, num_layers_bottom1, num_layers_decoder,
                            num_embeddings_top, num_embeddings_bottom)
        for param in self.vqvae.parameters():
            param.requires_grad = False

        self.audio_encoder = AudioEncoder(embed_dim, num_heads2,
                                          num_layers_top2, num_layers_bottom2)

    def predict(self, person_one_hot, text, audio):
        # 生成音频特征
        audio_features = self.audio_encoder.wav2vec2(audio)  # (B, N, 768)
        z_bottom_pred = self.audio_encoder.bottom(audio_features)  # (B, N/8, 768)
        z_top_pred = self.audio_encoder.top(z_bottom_pred)  # (B, N/16, 768)
        # top预测
        z_top_pred = self.audio_encoder.top_encoder(person_one_hot, text, z_top_pred)
        _, q_top_pred = self.vqvae.vq_layer_top(z_top_pred, sample=True, temperature=0.2, k=1)  # 0.2, 1
        # bottom预测
        z_bottom_pred = self.audio_encoder.bottom_encoder(person_one_hot, text, q_top_pred, z_bottom_pred)
        _, q_bottom_pred = self.vqvae.vq_layer_bottom(z_bottom_pred, sample=True, temperature=0.2, k=1)  # 0.2, 1

        # 解码人脸特征
        exp_out, jaw_out = self.vqvae.decoder(person_one_hot, text, q_top_pred, q_bottom_pred)
        # 拼接 exp_output, jaw_output
        exp_output = exp_out
        jaw_output = jaw_out
        return exp_output, jaw_output

    def forward(self, person_one_hot, text, audio, exp, jaw):
        # 生成人脸特征
        z_bottom = self.vqvae.bottom_encoder.squash(exp, jaw)
        z_top = self.vqvae.top_encoder.squash(z_bottom)
        # top
        z_top = self.vqvae.top_encoder(person_one_hot, text, z_top)
        _, q_top = self.vqvae.vq_layer_top(z_top)
        # bottom
        z_bottom = self.vqvae.bottom_encoder(person_one_hot, text, q_top, z_bottom)
        _, q_bottom = self.vqvae.vq_layer_bottom(z_bottom)

        # 生成音频特征
        audio_features = self.audio_encoder.wav2vec2(audio)  # (B, N, 768)
        z_bottom_pred = self.audio_encoder.bottom(audio_features)  # (B, N/8, 768)
        z_top_pred = self.audio_encoder.top(z_bottom_pred)  # (B, N/16, 768)
        # top预测
        z_top_pred = self.audio_encoder.top_encoder(person_one_hot, text, z_top_pred)
        _, q_top_pred = self.vqvae.vq_layer_top(z_top_pred)
        # bottom预测
        z_bottom_pred = self.audio_encoder.bottom_encoder(person_one_hot, text, q_top_pred, z_bottom_pred)
        _, q_bottom_pred = self.vqvae.vq_layer_bottom(z_bottom_pred)

        # 解码人脸特征
        exp_out, jaw_out = self.vqvae.decoder(person_one_hot, text, q_top_pred, q_bottom_pred)
        # 拼接 exp_output, jaw_output
        z_top_output = z_top
        z_bottom_output = z_bottom
        z_top_pred_output = z_top_pred
        z_bottom_pred_output = z_bottom_pred
        exp_output = exp_out
        jaw_output = jaw_out
        return z_top_output, z_bottom_output, z_top_pred_output, z_bottom_pred_output, exp_output, jaw_output
