# Authors: Keshik, Michael
# https://arxiv.org/abs/2503.01868?.

# Import base libraries
import os, sys

# Import scientific computing libraries
import torch

# Import custom utils
from utils import load_yaml_file


# Replacement operator class
class ReplacementFactory:
    """
    Factory class for constructing replacement operators used for grafting diffusion transformers.

    This class instantiates modules like MHA, Hyena, Mamba, MLP, etc.,
    based on a provided name and config file path.

    Attributes:
        operator_name (str): Name of the operator to load.
        config_file (str): Path to the YAML config file for the operator.
        operator (nn.Module): The instantiated operator module.
    """

    def __init__(self, operator_name, config_file_name):
        """
        Initialize the Replacement class by loading the operator based on name and config.

        Args:
            operator_name (str): Name of the operator (e.g., 'hyena_x', 'mha').
            config_file_name (str): Path to the config file describing the operator.
        """
        self.operator_name = operator_name
        self.config_file = config_file_name

        if ('mha' in self.operator_name) or ('swa' in self.operator_name):
            self.operator = self._create_mha_operator(self.config_file)
        elif 'hyena_se' == self.operator_name:
            self.operator = self._create_hyena_se_operator(self.config_file)
        elif 'hyena_y' == self.operator_name:
            self.operator = self._create_hyena_y_operator(self.config_file)
        elif 'hyena_x' == self.operator_name:
            self.operator = self._create_hyena_x_operator(self.config_file)
        elif 'mamba2' == self.operator_name:
            self.operator = self._create_mamba2_operator(self.config_file)
        elif 'mlp' == self.operator_name:
            self.operator = self._create_mlp_operator(self.config_file)
        elif 'mlp_hyena_x' == self.operator_name:
            self.operator = self._create_mlp_hyena_x_operator(self.config_file)
        else:
            raise NotImplementedError


    def _create_hyena_se_operator(self, config_file):
        sys.path.append(f"{os.getcwd()}/src/operators/hyena/") # Append paths
        config = load_yaml_file(config_file)
        from operators.hyena.hyena_se import Hyena_SE
        replacement_operator = Hyena_SE(**config)
        return replacement_operator


    def _create_hyena_y_operator(self, config_file):
        sys.path.append(f"{os.getcwd()}/src/operators/hyena/") # Append paths
        config = load_yaml_file(config_file)
        from operators.hyena.hyena_y import Hyena_Y
        replacement_operator = Hyena_Y(**config)
        return replacement_operator


    def _create_hyena_x_operator(self, config_file):
        sys.path.append(f"{os.getcwd()}/src/operators/hyena/") # Append paths
        config = load_yaml_file(config_file)
        from operators.hyena.hyena_x import Hyena_X
        replacement_operator = Hyena_X(**config)
        return replacement_operator


    def _create_mamba2_operator(self, config_file):
        config = load_yaml_file(config_file)
        from mamba_ssm import Mamba2
        mamba2_operator = Mamba2(**config)
        #print(short_hyena1D_operator)
        return mamba2_operator


    def _create_mha_operator(self, config_file):
        config = load_yaml_file(config_file)
       
        if not 'use_flash_attn' in config:
            from timm.models.vision_transformer import Attention
            attention_operator = Attention(**config)
            return attention_operator

        # Used for SWA experiments
        elif config['use_flash_attn'] == True:
            config['window_size'] = tuple(config['window_size'])
            from flash_attn.modules.mha import MHA as Attention
            attention_operator = Attention(**config)
            assert attention_operator.cross_attn == False
            return attention_operator

        else:
            raise Exception


    def _create_mlp_operator(self, config_file):
        from timm.models.vision_transformer import Mlp
        approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
        config = load_yaml_file(config_file)
        d_model, expansion_ratio, dropout = config['d_model'], config['expansion_ratio'], config['dropout']
        replacement_operator = Mlp(in_features=d_model, hidden_features=int(d_model*expansion_ratio), act_layer=approx_gelu, drop=dropout)
        return replacement_operator


    def _create_mlp_hyena_x_operator(self, config_file):
        sys.path.append(f"{os.getcwd()}/src/operators/mlp/") # Append paths
        config = load_yaml_file(config_file)
        from operators.mlp.mlp_hyena_x import MLP_Hyena_X
        replacement_operator = MLP_Hyena_X(**config)
        return replacement_operator


    def get_operator(self):
        return self.operator


    def test_operator(self, embedding_dim, device='cuda'):
        assert self.operator is not None
        self.operator = self.operator.to(device)

        # dtype = torch.bfloat16 if 'mha_flash_attention' in self.operator_name else torch.float16
        dtype = torch.bfloat16 
        random_tensor = torch.rand(1, 256, embedding_dim, dtype=dtype).to(device) # BATCH_SIZE, sequence_length, HIDDEN_SIZE
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            output = self.operator(random_tensor)
            print(random_tensor.size(), output.size())
        assert output.size() == random_tensor.size()




# Simple code for testing
if __name__ == '__main__':
    
    # Test code
    operators_to_config = {
        # 'hyena_se': './configs/sequence_mixers/hyena_se.yaml',
        'hyena_y': './configs/sequence_mixers/hyena_y.yaml',
        # 'hyena_x': './configs/sequence_mixers/hyena_x.yaml',
        # 'mamba2': './configs/sequence_mixers/mamba2.yaml',
        
        # 'mha': './configs/sequence_mixers/mha_timm.yaml',
        # 'swa': './configs/sequence_mixers/swa.yaml',
        # "mamba2": './configs/operators/mamba2/baseline.yaml',

        # 'mlp': './configs/channel_mixers/mlp_ratio_6.yaml',
        # 'mlp_hyena_x': './configs/channel_mixers/mlp_hyena_x.yaml',
        
    }

    for (key, value) in operators_to_config.items():
        #print(key, value)
        scion_object = ReplacementFactory(key, value)
        scion_operator = scion_object.get_operator()
        scion_object.test_operator(1152)