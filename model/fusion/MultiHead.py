import torch.nn as nn
import torch 
import math

# adopted from https://github.com/hkproj/pytorch-transformer/blob/main/model.py
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
class ReduceChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        interchannels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, interchannels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(interchannels, interchannels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MultiHeadFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, lastdimension, mode='decoupled', h=1):
        super().__init__()
        assert mode in ['decoupled'], "MultiHeadFusion currently only support for decoupled mode"
        self.mode = mode

        if mode == 'decoupled':
            box = []
            cls = []
            for idx, channels2D in enumerate(channels_2D):
                box.append(
                    nn.ModuleList([
                            ReduceChannels(channels2D[0], interchannels),
                            ReduceChannels(channels_3D, interchannels),
                            MultiHeadAttentionBlock(lastdimension[0][idx][0] * lastdimension[0][idx][1], h)
                        ])
                )
                
                cls.append(
                    nn.ModuleList([
                            ReduceChannels(channels2D[1], interchannels),
                            ReduceChannels(channels_3D, interchannels),
                            MultiHeadAttentionBlock(lastdimension[1][idx][0] * lastdimension[1][idx][1], h)
                    ])
                )

            self.box = nn.ModuleList(box)
            self.cls = nn.ModuleList(cls)

    def forward(self, ft_2D, ft_3D):
        B_3D, C_3D, H_3D, W_3D = ft_3D.shape

        fts = []

        if self.mode == 'decoupled':
            for idx, ft2D in enumerate(ft_2D):
                B_2D, C_2D, H_2D, W_2D = ft2D[0].shape
                assert H_2D/H_3D == W_2D/W_3D, "can't upscale"

                upsampling = nn.Upsample(scale_factor=H_2D/H_3D)
                ft_3D_t = upsampling(ft_3D)

                ft_box         = self.box[idx][0](ft2D[0])
                ft_3D_box      = self.box[idx][1](ft_3D_t)
                B1, C1, H1, W1 = ft_3D_box.shape
                ft_box         = ft_box.view(B1, C1, -1)
                ft_3D_box      = ft_3D_box.view(B1, C1, -1)

                ft_cls         = self.cls[idx][0](ft2D[1])
                ft_3D_cls      = self.cls[idx][1](ft_3D_t)
                B2, C2, H2, W2 = ft_3D_cls.shape 
                ft_cls         = ft_cls.view(B2, C2, -1)
                ft_3D_cls      = ft_3D_cls.view(B2, C2, -1)

                ft_box = self.box[idx][-1](ft_3D_box, ft_box, ft_box)
                ft_box = ft_box.view(B1, C1, H1, W1)

                ft_cls = self.cls[idx][-1](ft_cls, ft_3D_cls, ft_3D_cls)
                ft_cls = ft_cls.view(B2, C2, H2, W2)

                fts.append([ft_box, ft_cls])
        
        return fts