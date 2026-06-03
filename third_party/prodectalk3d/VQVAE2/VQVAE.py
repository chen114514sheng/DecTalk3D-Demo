import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from Utils import TextEncoder, TransformerLayer, DiffusionTransformerLayer, VectorQuantizer


class StyleEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(StyleEncoder, self).__init__()
        self.style_embedding = nn.Linear(46, embed_dim)
        self.squash_style = nn.Sequential(
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
        self.transformer_style = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)

    def forward(self, person_one_hot, exp, jaw):
        style_features = self.style_embedding(person_one_hot.unsqueeze(1))
        x = torch.cat((exp, jaw), dim=2)
        style = self.squash_style(x.permute(0, 2, 1)).permute(0, 2, 1)
        style = self.transformer_style(style_features * style)
        #style = self.transformer_style(style)
        return style


class TopEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TopEncoder, self).__init__()
        self.text_encoder = TextEncoder(embed_dim)
        self.transformer_top = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)

    def forward(self, text, style):
        text_features = self.text_encoder(text)
        z_top = self.transformer_top(text_features * style)
        #z_top = self.transformer_top(style)
        return z_top


class BottomEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(BottomEncoder, self).__init__()
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.transformer_bottom = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)

    def forward(self, q_top, style):
        q_top = self.top_embedding(q_top)
        z_bottom = self.transformer_bottom(q_top + style)
        #z_bottom = self.transformer_bottom(style)
        return z_bottom


class BottomDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(BottomDecoder, self).__init__()
        self.expand_top = nn.Sequential(
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
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.bottom_embedding = nn.Linear(embed_dim, embed_dim)
        self.transformer_bottom = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)

    def forward(self, q_top, q_bottom):
        q_top = self.expand_top(q_top.permute(0, 2, 1)).permute(0, 2, 1)
        q_bottom = self.expand_bottom(q_bottom.permute(0, 2, 1)).permute(0, 2, 1)
        q_top = self.top_embedding(q_top)
        q_bottom = self.bottom_embedding(q_bottom)
        bottom = self.transformer_bottom(q_top + q_bottom)
        #bottom = self.transformer_bottom(q_bottom)
        return bottom


class TopDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TopDecoder, self).__init__()
        self.text_encoder = TextEncoder(embed_dim)
        self.expand_top = nn.Sequential(
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
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.transformer_top = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)

    def forward(self, text, q_top):
        text_features = self.text_encoder(text)
        q_top = self.expand_top(q_top.permute(0, 2, 1)).permute(0, 2, 1)
        q_top = self.top_embedding(q_top)
        top = self.transformer_top(text_features * q_top)
        #top = self.transformer_top(q_top)
        return top


class StyleDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(StyleDecoder, self).__init__()
        self.style_embedding = nn.Linear(46, embed_dim)
        self.top_embedding = nn.Linear(embed_dim, embed_dim)
        self.bottom_embedding = nn.Linear(embed_dim, embed_dim)
        self.transformer_style = TransformerLayer(d_model=embed_dim, nhead=num_heads, num_layers=num_layers)
        self.exp_head = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
        self.jaw_head = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(embed_dim // 2, 3, kernel_size=1)
        )

    def forward(self, person_one_hot, top, bottom):
        style_features = self.style_embedding(person_one_hot.unsqueeze(1))
        top = self.top_embedding(top)
        bottom = self.bottom_embedding(bottom)
        x = self.transformer_style(style_features * (top + bottom))
        #x = self.transformer_style(top + bottom)
        exp = self.exp_head(x.permute(0, 2, 1)).permute(0, 2, 1)
        jaw = self.jaw_head(x.permute(0, 2, 1)).permute(0, 2, 1)
        return exp, jaw


class VQVAE(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 num_layers_style, num_layers_top, num_layers_bottom, num_embeddings):
        super(VQVAE, self).__init__()
        self.style_encoder = StyleEncoder(embed_dim, num_heads, num_layers_style)
        self.top_encoder = TopEncoder(embed_dim, num_heads, num_layers_top)
        self.bottom_encoder = BottomEncoder(embed_dim, num_heads, num_layers_bottom)

        self.bottom_decoder = BottomDecoder(embed_dim, num_heads, num_layers_bottom)
        self.top_decoder = TopDecoder(embed_dim, num_heads, num_layers_top)
        self.style_decoder = StyleDecoder(embed_dim, num_heads, num_layers_style)

        self.vq_layer_top = VectorQuantizer(num_embeddings, embed_dim, 0.25)
        self.vq_layer_bottom = VectorQuantizer(num_embeddings, embed_dim, 0.25)

    def forward(self, person_one_hot, text, exp, jaw):
        style = self.style_encoder(person_one_hot, exp, jaw)
        z_top = self.top_encoder(text, style)
        loss_top, q_top = self.vq_layer_top(z_top)
        z_bottom = self.bottom_encoder(q_top, style)
        loss_bottom, q_bottom = self.vq_layer_bottom(z_bottom)

        bottom = self.bottom_decoder(q_top, q_bottom)
        top = self.top_decoder(text, q_top)
        exp_output, jaw_output = self.style_decoder(person_one_hot, top, bottom)
        return loss_top, loss_bottom, exp_output, jaw_output