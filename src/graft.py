# Authors: Keshik
# Minimal code for creating grafts proposed in https://arxiv.org/abs/2506.05340
import copy
from replacement_factory import ReplacementFactory
import torch
from models.dit import DiT_models
from models.download import find_model


def graft_dit(dit_model_name, dit_ckpt_path, image_size,
                    operator_type, operator_name, operator_config_filepath, 
                    graft_indexes, graft_weights, 
                    run_all_unit_tests=True):
    """
    Replaces selected blocks in a pretrained Diffusion Transformer (DiT) model
    with alternative operators, using the "grafting" method introduced in
    https://arxiv.org/abs/2506.05340.

    Parameters:
        dit_model_name (str): Name of the DiT model variant (e.g., 'DiT-XL/2').
        dit_ckpt_path (str or None): Path to the checkpoint with pretrained weights,
                                     including grafted operator states (if available).
        image_size (int): Input image resolution (e.g., 256 or 512).
        operator_type (str): Type of component to replace, either 'mha' or 'mlp'.
        operator_name (str): Name of the operator being inserted (e.g., 'hyena', 'conv').
        operator_config_filepath (str): Path to the YAML config file for the new operator.
        graft_indexes (list of int or None): Indexes of blocks to graft the new operator into.
        graft_weights (dict): Optional dictionary mapping graft index to operator checkpoint path.
        run_all_unit_tests (bool): Placeholder for future tests or assertions (currently unused).

    Returns:
        torch.nn.Module: The modified DiT model with specified operators grafted.
    """
    # Load Model
    device='cpu'
    latent_size = image_size // 8
    model = DiT_models[dit_model_name](
        input_size=latent_size,
        num_classes=1000
    ).to(device)


    if dit_model_name == "DiT-XL/2":
        ckpt_path = f"DiT-XL-2-{image_size}x{image_size}.pt"
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict, strict=True)
        print(f'> Model={dit_model_name}, Image size={image_size}, loaded {ckpt_path} successfully')
    

    # Graft
    if graft_indexes is None:
        return model


    for index in graft_indexes:
        # Create new operators for each index.
        scion_obj = ReplacementFactory(operator_name, operator_config_filepath)
        scion_model = scion_obj.get_operator().to(device)

        # Load weights (if any)
        if graft_weights is not None:
            if index in graft_weights:
                scion_ckpt_path = graft_weights[index]
                scion_state_dict = torch.load(scion_ckpt_path, map_location='cuda')
                scion_model.load_state_dict(scion_state_dict['model'], strict=True)
                print(f'> Graft Index ={index}, loaded {scion_ckpt_path} successfully')

        # Swap
        if operator_type == 'mha':
            model.blocks[index].attn = scion_model
        elif operator_type == 'mlp':
            model.blocks[index].mlp = scion_model
        else:
            raise Exception

        print(f'Grafted {operator_type} Block {index} with {operator_name}; Config file@{operator_config_filepath}')


    if dit_ckpt_path is not None:
        ckpt_path = dit_ckpt_path
        state_dict = torch.load(ckpt_path, map_location='cuda')['ema']
        model.load_state_dict(state_dict, strict=True) # For grafting using heterogenous operators.
        print(f'> Model={dit_model_name}, Image size={image_size}, loaded {ckpt_path} successfully')
    

    return model



def test():
    dit_model_name = 'DiT-XL/2'
    dit_ckpt_path = None
    image_size = 256 
    operator_name = 'mha'
    operator_config_filepath = 'configs/sequence_mixers/mha_timm.yaml'
    graft_indexes = [14, 15]
    graft_weights = {
       
    }

    graft_dit(dit_model_name, dit_ckpt_path, image_size, 
              operator_name, operator_name, operator_config_filepath, 
              graft_indexes, graft_weights, 
              run_all_unit_tests=False)


if __name__ == '__main__':
    test()