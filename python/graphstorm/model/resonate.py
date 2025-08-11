import torch
from torch.nn import Linear, LayerNorm, MultiheadAttention, ModuleList, Sequential, Module, ReLU
from torch.nn.functional import relu, dropout, gelu
import structlog as sl

logger = sl.get_logger(__name__)

class AttentionBlock(Module):
    def __init__(self, dim_model, dim_key, dim_value, dim_ffn, attention_heads, dropout_rate_attention=.05, **kwargs):
        super(AttentionBlock, self).__init__()
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_ffn = dim_ffn
        self.attention_heads = attention_heads
        self.dropout_rate_attention = dropout_rate_attention

        self.q_proj = Linear(self.dim_model, self.attention_heads * self.dim_key, bias=False)  # 4 → 56
        self.k_proj = Linear(self.dim_model, self.attention_heads * self.dim_key, bias=False)  # 4 → 56
        self.v_proj = Linear(self.dim_model, self.attention_heads * self.dim_value, bias=False)  # 4 → 21

        self.mha = MultiheadAttention(
            # embed_dim=self.dim_model * self.params['attention_heads'], # 4 * 8 = 32
            embed_dim=self.attention_heads * self.dim_key,  # heads * query_dim
            num_heads=self.attention_heads,
            dropout=self.dropout_rate_attention,
            bias=False,
            batch_first=True,
            kdim=self.attention_heads * self.dim_key,
            vdim=self.attention_heads * self.dim_value
        )

        self.out_proj = Linear(self.attention_heads * self.dim_key, self.dim_model, bias=False)  # 21 → 4

        # these values need to be changed
        self.norm = LayerNorm(self.dim_model, eps=1e-6)

        self.ffn = Sequential(
            Linear(self.dim_model, out_features=self.dim_ffn, bias=False),
            ReLU(),
            Linear(in_features=self.dim_ffn, out_features=self.dim_model, bias=False)
        )

        self.out_norm = LayerNorm(self.dim_model, eps=1e-6)

    def forward(self, x, edge_index=None):
        # multi-head attention
        q = self.q_proj(x)  
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q = k = v = x

        attn_output, _ = self.mha(q, k, v)

        attn_output = self.out_proj(attn_output)  # (batch, 128, 4)

        # residual connection
        x = x + attn_output
        
        x = self.norm(x)

        # ffn
        ffn_output = self.ffn(x)

        # residual
        x = x + ffn_output

        # layernorm
        x = self.out_norm(x)

        return x


class HMA(Module): 
    def __init__(
        self,
        *,
        in_features: int = 512, 
        out_features: int,
        version: int = 19,
        **kwargs
    ):
        super().__init__()

        if str(version) == '19':
            hyperparameters = {'activation_input': 'linear', 'attention_heads': 5, 'dim_ffn': 648, 'dim_input': 128, 'dim_key': 5, 'dim_model': 14, 'dim_value': 8, 'dropout_rate_attention': 0.912, 'dropout_rate_bottleneck': 0.0, 'dropout_rate_input': 0.608, 'gaussian_noise_bottleneck': 0.0, 'gaussian_noise_input': 0.0, 'n_attention_steps': 5}
        elif str(version) == '18':
            hyperparameters = {'activation_input': 'linear', 'attention_heads': 5, 'dim_ffn': 414, 'dim_input': 256, 'dim_key': 6, 'dim_model': 10, 'dim_value': 4, 'dropout_rate_attention': 0.02, 'dropout_rate_bottleneck': 0.0, 'dropout_rate_input': 0.42, 'gaussian_noise_bottleneck': 0.0, 'gaussian_noise_input': 0.0, 'n_attention_steps': 5}
        self.hyperparameters = hyperparameters

        input_dimension = hyperparameters['dim_model'] * hyperparameters['dim_input'] # 14 * 128


        # encoder
        self.input_proj = Linear(in_features, input_dimension)
        # # transformations
        self.attention_layers = ModuleList([
            AttentionBlock(**hyperparameters) for _ in range(hyperparameters['n_attention_steps'])
        ])

        # decoder
        self.decoder = Linear(input_dimension, out_features=out_features)

    def forward(self, x, *args, **kwargs):
        x = self.input_proj(x)

        x = dropout(x, p=self.hyperparameters['dropout_rate_input'], training=self.training) # (batch_size, 512)

        # # TRANSFORMATIONS
        # # reshape to have dim_input "embeddings" with "dim_model" dimensions    
        x = x.view(-1, self.hyperparameters['dim_input'], self.hyperparameters['dim_model']) # (batch_size, 128, 4)
        
        # # normalize by sqrt(dim_model) as in the paper
        x = x * (self.hyperparameters['dim_model'] ** .5)

        # # pass through attn blocks
        for attn in self.attention_layers:
            x = attn(x)

        # # OUTPUT
        x = x.flatten(start_dim=1)  # flatten with respect of batch
        
        # x = dropout(x, p=self.hyperparameters['dropout_rate_bottleneck'], training=self.training)  # this is 0
        # x = x + torch.randn_like(x) * 0  # this hyperparam is also 0
        
        x = self.decoder(x)
        return x