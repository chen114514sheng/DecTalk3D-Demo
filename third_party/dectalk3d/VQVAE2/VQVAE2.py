import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from Utils import TextEncoder, TransformerLayer, TransformerDecoderLayer, VectorQuantizer


class BottomEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(BottomEncoder, self).__init__()
        self.squash_bottom = nn.Sequential(
            nn.Conv1d(103, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
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
        self.expand_top = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.transformer_bottom = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.style_embedding = nn.Linear(46, embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

    def squash(self, exp, jaw):
        x = torch.cat((exp, jaw), dim=2)
        z_bottom = self.squash_bottom(x.permute(0, 2, 1)).permute(0, 2, 1)
        return z_bottom

    def forward(self, person_one_hot, text, q_top, z_bottom):
        q_top = self.expand_top(q_top.permute(0, 2, 1)).permute(0, 2, 1)
        q_top = self.top_embedding(q_top)
        style_features = self.style_embedding(person_one_hot.unsqueeze(1))
        text_features = self.text_encoder(text)

        z_bottom = self.transformer_bottom(style_features * text_features * (q_top + z_bottom))
        #z_bottom = self.transformer_bottom(q_top + z_bottom)
        #z_bottom = self.transformer_bottom(style_features * text_features * z_bottom)
        #z_bottom = self.transformer_bottom(z_bottom)
        return z_bottom


class TopEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TopEncoder, self).__init__()
        self.squash_top = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        self.transformer_top = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)
        self.style_embedding = nn.Linear(46, embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

    def squash(self, z_bottom):
        z_top = self.squash_top(z_bottom.permute(0, 2, 1)).permute(0, 2, 1)
        return z_top

    def forward(self, person_one_hot, text, z_top):
        style_features = self.style_embedding(person_one_hot.unsqueeze(1))
        text_features = self.text_encoder(text)

        z_top = self.transformer_top(style_features * text_features * z_top)
        #z_top = self.transformer_top(z_top)
        return z_top


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.expand_top = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.expand_bottom = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(embed_dim, affine=True),
            nn.LeakyReLU(0.2)
        )
        self.transformer = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)
        self.conv1d = nn.Conv1d(embed_dim, 103, kernel_size=3, padding=1, padding_mode='replicate')
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.style_embedding = nn.Linear(46, embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

    def forward(self, person_one_hot, text, q_top, q_bottom):
        q_bottom = self.expand_bottom(q_bottom.permute(0, 2, 1)).permute(0, 2, 1)
        q_top = self.expand_top(q_top.permute(0, 2, 1)).permute(0, 2, 1)
        q_top = self.top_embedding(q_top)
        style_features = self.style_embedding(person_one_hot.unsqueeze(1))
        text_features = self.text_encoder(text)

        x = self.transformer(style_features * text_features * (q_top + q_bottom))
        #x = self.transformer(q_top + q_bottom)
        #x = self.transformer(style_features * text_features * q_bottom)
        #x = self.transformer(q_bottom)

        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x[:, :, :100], x[:, :, 100:]


class VQVAE2(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 num_layers_top, num_layers_bottom, num_layers_decoder,
                 num_embeddings_top, num_embeddings_bottom):
        super(VQVAE2, self).__init__()
        self.bottom_encoder = BottomEncoder(embed_dim, num_heads, num_layers_top)
        self.top_encoder = TopEncoder(embed_dim, num_heads, num_layers_bottom)
        self.decoder = Decoder(embed_dim, num_heads, num_layers_decoder)

        self.vq_layer_bottom = VectorQuantizer(num_embeddings_bottom, embed_dim, 0.25)
        self.vq_layer_top = VectorQuantizer(num_embeddings_top, embed_dim, 0.25)

    def forward(self, person_one_hot, text, exp, jaw):
        z_bottom = self.bottom_encoder.squash(exp, jaw)
        z_top = self.top_encoder.squash(z_bottom)

        z_top = self.top_encoder(person_one_hot, text, z_top)
        loss_top, q_top = self.vq_layer_top(z_top)
        z_bottom = self.bottom_encoder(person_one_hot, text, q_top, z_bottom)
        loss_bottom, q_bottom = self.vq_layer_bottom(z_bottom)

        exp_output, jaw_output = self.decoder(person_one_hot, text, q_top, q_bottom)
        return exp_output, jaw_output, loss_top, loss_bottom

    @torch.no_grad()
    def encode_codes(self, person_one_hot, text, exp, jaw):
        """
        编码源样本，得到量化后的 top / bottom 离散特征。
        只做编码，不做解码。
        """
        z_bottom = self.bottom_encoder.squash(exp, jaw)
        z_top = self.top_encoder.squash(z_bottom)

        z_top = self.top_encoder(person_one_hot, text, z_top)
        _, q_top = self.vq_layer_top(z_top)

        z_bottom = self.bottom_encoder(person_one_hot, text, q_top, z_bottom)
        _, q_bottom = self.vq_layer_bottom(z_bottom)

        return q_top, q_bottom

    @torch.no_grad()
    def decode_codes(self, person_one_hot, text, q_top, q_bottom):
        """
        给定条件和量化特征，直接解码得到 exp / jaw。
        """
        exp_output, jaw_output = self.decoder(person_one_hot, text, q_top, q_bottom)
        return exp_output, jaw_output

    @torch.no_grad()
    def reconstruct_with_swapped_condition(
        self,
        source_person_one_hot,
        source_text,
        source_exp,
        source_jaw,
        target_person_one_hot=None,
        target_text=None,
    ):
        """
        先用 source 样本编码，再在解码端替换条件。
        - 若 target_text 不为 None：交换文本条件
        - 若 target_person_one_hot 不为 None：交换身份条件
        """
        q_top, q_bottom = self.encode_codes(
            source_person_one_hot,
            source_text,
            source_exp,
            source_jaw
        )

        decode_person = source_person_one_hot if target_person_one_hot is None else target_person_one_hot
        decode_text = source_text if target_text is None else target_text

        exp_output, jaw_output = self.decode_codes(
            decode_person,
            decode_text,
            q_top,
            q_bottom
        )
        return exp_output, jaw_output