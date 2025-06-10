# Authors: Keshik, Michael
# Minimal implementation of Hyena-Y operator (Sequence mixer) proposed in https://arxiv.org/abs/2506.05340

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


class Hyena_Y(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2, 
            inner_filter_len=4,
            num_heads=1, 
            inner_factor=1,
            num_blocks=1, 
            activation="id",
        ):
        r"""
        Hyena-Y operator described in the paper https://arxiv.org/abs/2506.05340
        
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            inner_filter_len (int): Length of the explicit inner convolutional filter. Defaults to 4.
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
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
        self.inner_filter_len = inner_filter_len
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
        # total_width = self.d_model * self.inner_factor * (self.order + 1)
        
        self.inner_filter = CausalConv1D_filter(
            in_channels=self.d_model, 
            out_channels=self.d_model, 
            kernel_size=self.inner_filter_len, 
            groups=self.d_model, 
            padding=self.inner_filter_len - 1
        )
        self._initialize_inner_filter_as_identity()
        # trainable_params = sum(p.numel() for p in self.short_filter.parameters() if p.requires_grad)
        # print(self.inner_filter)

    def _initialize_inner_filter_as_identity(self):
        with torch.no_grad():
            self.inner_filter.weight.zero_()
            self.inner_filter.weight[:, 0, self.inner_filter_len-1] = 1.0
            
            # Zero out the bias
            if self.inner_filter.bias is not None:
                self.inner_filter.bias.zero_()

    
    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        uc = self.in_proj(u)
        uc = uc.transpose(1, 2)
        #uc = self.short_filter(u)[...,:l_filter] # No explicit input featurizer convs
        q, k, v = uc.split(self.d_model, dim=1)        
        v = k * v
        v = self.inner_filter(v)[...,:l_filter]
        y = q * v
        y = y.transpose(1, 2)
        y = self.out_proj(y)
        
        return y


    @property
    def d_output(self):
        return self.d_model



def test_hyena_y():
    device='cuda'
    dtype = torch.bfloat16 

    instance = Hyena_Y(1152, 256).to(dtype).to(device)
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


    path = '/results/neurips/short_hyena_y/block_0/mae/scaled_pred=False/seed=0/checkpoints/best_val.pt'
    ckpt = torch.load(path, map_location='cuda')['model']
    print(ckpt.keys())
    msg = instance.load_state_dict(ckpt, strict=True)
    print(msg)



if __name__ == '__main__':
    test_hyena_y()