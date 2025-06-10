# Authors: Keshik, Michael
# Minimal implementation of MLP Hyena-X operator (Channel mixer) used in https://arxiv.org/abs/2506.05340

# Import base modules
import math
import yaml
from functools import partial


# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Install causal-conv1d from here: https://github.com/Dao-AILab/causal-conv1d (Suggest to build from source)
from causal_conv1d import causal_conv1d_fn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- Helper functions -----
def load_yaml_file(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data
# -----------------------------

class CausalConv1D_filter(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(CausalConv1D_filter, self).__init__(*args, **kwargs)

    def forward(self, input):
        w = rearrange(self.weight, "d 1 w -> d w")
        custom_output = causal_conv1d_fn(input, w, bias=self.bias, activation=None, seq_idx=None)
        return custom_output


class MLP_Hyena_X(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            expansion_ratio=2,
            order=2, 
            inner_factor=1,
            num_heads=1,
            num_blocks=1,
            short_filter_order=4, 
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            expansion_ratio: MLP expansion ratio. Defaults to 2.
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
        """
        super().__init__()
        assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
        assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.expansion_ratio = expansion_ratio
        self.inner_factor = inner_factor
        self.short_filter_order = short_filter_order
        self.setup_projections()
        self.setup_filters()


    def setup_projections(self):
        "Initializes input and output projections (over the width dimension)"
        linear_cls = nn.Linear 
        self.in_proj = linear_cls(int(self.d_model), self.d_model*self.expansion_ratio*3) # Iso-FLOPs setup
        self.out_proj = linear_cls(int(self.d_model*self.expansion_ratio), self.d_model) # Iso-FLOPs setup
    

    def setup_filters(self):   
        "Initializes the explicit and implicit filters"
        assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
        total_width = self.l_max
        
        self.short_filter = CausalConv1D_filter(
            in_channels=total_width, 
            out_channels=total_width, 
            kernel_size=self.short_filter_order, 
            groups=total_width, 
            padding=self.short_filter_order - 1
        )
        

    def forward(self, u):
        uc = self.in_proj(u) # Dense
        uc = self.short_filter(uc)  # Conv
        q, k, v = uc.split(int(self.d_model*self.expansion_ratio), dim=-1)    
        v = k * v # Gate 1
        y = q * v # Gate 2
        y = self.out_proj(y) # Dense
        return y


    @property
    def d_output(self):
        return self.d_model



def test_mlp_hyena_x():
    device='cuda'
    dtype = torch.bfloat16 

    mlp_hyena_x_instance = MLP_Hyena_X(1152, 256).to(dtype).to(device)
    print(mlp_hyena_x_instance)
    num_params = count_parameters(mlp_hyena_x_instance)
    print(f"Total parameters: {num_params:,}")

    # Test the operator by simply passing a sequence of (1, seq_len, embedding_dim)
    seq_len = 256
    embedding_dim = 1152
    random_tensor = torch.rand(1, seq_len, embedding_dim, dtype=dtype).to(device) # BATCH_SIZE, seq_len, embedding_dim
    with torch.autocast(device_type=device, dtype=dtype, enabled=True):
        output = mlp_hyena_x_instance(random_tensor)
        print(f"Input shape:  {random_tensor.shape}")
        print(f"Output shape: {output.shape}")


    path = '/results/neurips/mlp_knowledge_transfer_1e_3/mlp_hyena_x/block_0/mse/scaled_pred=False/seed=0/checkpoints/best_val.pt'
    ckpt = torch.load(path, map_location='cuda')['model']
    print(ckpt.keys())
    msg = mlp_hyena_x_instance.load_state_dict(ckpt, strict=True)
    print(msg)



if __name__ == '__main__':
    test_mlp_hyena_x()