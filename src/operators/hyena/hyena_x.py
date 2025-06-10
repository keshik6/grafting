# Authors: Keshik, Michael
# Minimal implementation of Hyena-X operator (Sequence mixer) proposed in https://arxiv.org/abs/2506.05340

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


class CausalConv1D_filter(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(CausalConv1D_filter, self).__init__(*args, **kwargs)

    def forward(self, input):
        w = rearrange(self.weight, "d 1 w -> d w")
        custom_output = causal_conv1d_fn(input, w, bias=self.bias, activation=None, seq_idx=None)
        return custom_output


class Hyena_X(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            num_heads=1, 
            inner_factor=1,
            num_blocks=1, 
            short_filter_order=4, 
            activation="id",
        ):
        r"""
        Hyena-X operator described in the paper https://arxiv.org/abs/2506.05340
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            short_filter_order: (int): Length of the explicit convolutional filters for featurization. Defaults to 4
            activation: (str): type of act between kernel output and FF (default identity)
        """
        super().__init__()
        assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
        assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.block_dim = block_dim
        self.head_dim = head_dim
        self.short_filter_order = short_filter_order
        self.num_blocks = num_blocks
        self.activation = activation
        self.setup_projections()
        self.setup_filters()


    def setup_projections(self):
        "Initializes input and output projections"
        linear_cls = nn.Linear 
        self.out_proj = linear_cls(self.d_model * self.inner_factor, self.d_model)
        self.in_proj = linear_cls(self.d_model, (self.order + 1) * self.d_model) 
        
            
    def setup_filters(self):   
        "Initializes explicit filters"
        assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
        total_width = self.d_model * self.inner_factor * (self.order + 1)
        
        self.short_filter = CausalConv1D_filter(
            in_channels=total_width, 
            out_channels=total_width, 
            kernel_size=self.short_filter_order, 
            groups=total_width, 
            padding=self.short_filter_order - 1
        )
        # trainable_params = sum(p.numel() for p in self.short_filter.parameters() if p.requires_grad)
        
    
    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u) # Dense projection
        u = u.transpose(1, 2)
        uc = self.short_filter(u)[...,:l_filter] # Conv featurizer (Explicit)
        q, k, v = uc.split(self.d_model, dim=1)        
        v = k * v
        # v = self.inner_filter(v) # Inner filter (This line is the only difference between Hyena-SE and Hyena-X)
        y = q * v
        y = y.transpose(1, 2)
        y = self.out_proj(y)
        return y


    @property
    def d_output(self):
        return self.d_model



def test_hyena_se():
    device='cuda'
    dtype = torch.bfloat16 

    instance = Hyena_X(1152, 256).to(dtype).to(device)
    print(instance)
    num_params = count_parameters(instance)
    print(f"Total parameters: {num_params:,}")

    # Test the operator by simply passing a sequence of (1, seq_len, embedding_dim)
    seq_len = 256
    embedding_dim = 1152
    random_tensor = torch.rand(1, seq_len, embedding_dim, dtype=dtype).to(device) # BATCH_SIZE, seq_len, embedding_dim
    with torch.autocast(device_type=device, dtype=dtype, enabled=True):
        output = instance(random_tensor)
        print(f"Input shape:  {random_tensor.shape}")
        print(f"Output shape: {output.shape}")


    path = '/results/scion_training/8k/short_hyena/block_0/mae/scaled_pred=False/seed=0/checkpoints/best_train.pt'
    ckpt = torch.load(path, map_location='cuda')['model']
    print(ckpt.keys())
    msg = instance.load_state_dict(ckpt, strict=True)
    print(msg)



if __name__ == '__main__':
    test_hyena_se()