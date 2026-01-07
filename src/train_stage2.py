# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# References:
# https://github.com/facebookresearch/DiT
# https://github.com/chuanyangjin/fast-DiT

"""
Grafting: Stage 2 (Lightweight finetuning)
"""
import sys, os, json, math
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import yaml

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
import torch.nn as nn
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


sys.path.append(f"{os.getcwd()}/src/datasets/") # Append paths
# Printing
from rich.console import Console
from rich.table import Table
import wandb
import torch.nn.functional as F
from utils import *
from imagenet_vae_dataset import ImageNet_VAE_WebDataset
from itertools import islice
from graft import graft_dit



#################################################################################
#                                  Training Helper Functions                    #
#################################################################################

def list_to_string(int_list):
    return f"({', '.join(map(str, int_list))})"

def list_to_underscore_string(int_list):
    return '_'.join(map(str, int_list))


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def wandb_log(metrics_dict):
    # Log wandb metrics
    wandb.log(metrics_dict)


def convert_to_thousands_string(number):
    if number % 1000 == 0:
        return f"{number // 1000}k"
    else:
        raise ValueError("Number is not in thousands")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def freeze_first_n_layers(model, n, only_attn=False):
    """
    Freeze the first `n` blocks of the model. If `only_attn` is True, fine-tune only the `attn` parameters in the last M blocks,
    and freeze all parameters in the final layer.
    
    Args:
    - model: The neural network model.
    - n: Number of blocks to freeze from the beginning.
    - only_attn: Boolean flag to indicate if only `attn` parameters should be fine-tuned in the last M blocks.
    """
    total_blocks = len(model.blocks)
    
    for i, block in enumerate(model.blocks):
        if i < n:
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True

    # Handle the final layer
    for param in model.final_layer.parameters():
        param.requires_grad = True

    for param in model.x_embedder.parameters():
        param.requires_grad = True

    for param in model.t_embedder.parameters():
        param.requires_grad = True

    for param in model.y_embedder.parameters():
        param.requires_grad = True


def calculate_average_loss(metrics):
    """
    Calculate the average loss from the accumulated metrics.

    Args:
    metrics (list or torch.Tensor): A list or tensor containing accumulated metrics.
                                    The format should be [loss1, samples1, loss2, samples2, ..., lossN, samplesN].

    Returns:
    float: The average loss.
    """
    
    total_losses = metrics[0]
    vb_losses = metrics[1]
    mse_losses = metrics[2]
    num_samples = metrics[3]

    #print(losses, num_samples)

    # Calculate the total loss and the total number of samples
    total_loss = torch.sum(total_losses)
    total_vb_loss = torch.sum(vb_losses)
    total_mse_loss = torch.sum(mse_losses)
    total_samples = torch.sum(num_samples)

    # Compute the final average loss
    average_total_loss = total_loss / total_samples
    average_vb_loss = total_vb_loss / total_samples
    average_mse_loss = total_mse_loss / total_samples

    # total train steps
    total_train_steps = torch.sum(metrics[4])

    return average_total_loss.item(), average_vb_loss.item(), average_mse_loss.item(), total_samples.item(), total_train_steps.item()


def lr_lambda(current_step):
    total_steps = 500*8 # hardcoded (128k samples @ Global batch size = 256)
    if current_step < total_steps:
        return current_step / total_steps
    else:
        return 1.0


def model_sanity_check_print_trainable_params(model, model_name, check_instance, print_full_list=True):
    console = Console()

    # Print trainable parameters (second sanity check after accelerator wrapping)
    trainable_params = [n for (n, p) in model.named_parameters() if p.requires_grad]
    
    if print_full_list:
        console.print(f'[bold green]List of trainable parameters in {model_name}, (Sanity check {check_instance})[/bold green]')
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Trainable Parameters")
        for element in trainable_params:
            table.add_row(element)
        console.print(table)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f'[bold blue]Trainable params in {model_name} => (Sanity check {check_instance}): {total_trainable_params}[/bold blue]')



def train_dit(model, ema, loader, opt, scheduler, device, accelerator, logger, checkpoint_dir, args, 
            train_steps = 0, train_images = 0):
    # Do sanity check
    #model_sanity_check_print_trainable_params(model)
    if accelerator.is_main_process:
        model_sanity_check_print_trainable_params(model, 'train', 'before training', print_full_list=True)

    # For logging
    log_steps = 0
    train_steps = train_steps
    train_images = train_images

    # Track all these losses
    local_running_total_loss =  torch.tensor(0.0, device=device)
    local_running_vb_loss =  torch.tensor(0.0, device=device)
    local_running_mse_loss =  torch.tensor(0.0, device=device)
    local_sample_count =  torch.tensor(0.0, device=device)
    local_train_steps =  torch.tensor(0.0, device=device)
    start_time = time()

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Log everything at step level
    start_epoch, end_epoch = 0, args.epochs
    save_prefix=''

    for epoch in range(start_epoch, end_epoch):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch+1}...")

        for _, x, y in loader:
            x = x.to(device)
            y = y.to(device)
            #print(x.size(),y.size())
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # Forward pass, record losses
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            #print(loss_dict.keys())
            loss = loss_dict["loss"].mean()
            vb_loss = loss_dict["vb"].mean()
            mse_loss = loss_dict["mse"].mean()

            # Update model
            opt.zero_grad()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 10.0) # You might encounter a gradient explosion problem

            opt.step()
            update_ema(ema, model)

            if scheduler is not None:
                scheduler.step() # Update the learning rate

            # Log loss values:
            #print(loss.item(), vb_loss.item(), mse_loss.item(), accelerator.num_processes)
            local_running_total_loss += loss.item()*x.size(0)
            local_running_vb_loss += vb_loss.item()*x.size(0)
            local_running_mse_loss += mse_loss.item()*x.size(0)
            local_sample_count += x.size(0)
            local_train_steps += 1
            
            # Log and reset log steps
            if (local_train_steps*accelerator.num_processes) % args.log_every == 0:
                # Training Statistics
                gathered_items_train = accelerator.gather_for_metrics((local_running_total_loss, local_running_vb_loss, local_running_mse_loss, 
                                                                    local_sample_count, local_train_steps))
                
                gathered_total_loss, gathered_vb_loss, gathered_mse_loss, gathered_total_samples_train, gathered_total_steps = calculate_average_loss(gathered_items_train)
                train_images += gathered_total_samples_train
                log_steps += 1
                train_steps += (int(gathered_total_steps)//accelerator.num_processes)

                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = gathered_total_steps / (end_time - start_time)

                metrics_dict = {
                "train_images_seen_so_far": train_images,
                "train_total_loss": gathered_total_loss,
                "train_vb_loss": gathered_vb_loss,
                "train_mse_loss": gathered_mse_loss,
                "epoch": epoch,
                "lr": get_lr(opt),
                "train_steps": train_steps,
                }

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {gathered_total_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    wandb_log(metrics_dict)
                
                # Reset monitoring variables:
                local_running_total_loss =  torch.tensor(0.0, device=device)
                local_running_vb_loss =  torch.tensor(0.0, device=device)
                local_running_mse_loss =  torch.tensor(0.0, device=device)
                local_sample_count =  torch.tensor(0.0, device=device)
                local_train_steps =  torch.tensor(0.0, device=device)
                log_steps = 0
                start_time = time()

                # Save DiT checkpoint (Save the first one and last one seperately. The last one should have the optimizer state and the train steps)
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args,
                            "train_images_seen_so_far": train_images,
                            "train_steps": train_steps,
                        }
                        
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")


    # Finish everything and save the last checkpoint
    if accelerator.is_main_process:
        checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_images_seen_so_far": train_images,
                    "train_steps": train_steps,
        }

        checkpoint_path = f"{checkpoint_dir}/{save_prefix}last.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved last checkpoint to {checkpoint_path}")

        # Print trainable parameters (for sanity check)
        model_sanity_check_print_trainable_params(model, 'train', 'after completion', print_full_list=False)

    return model, train_steps, train_images



def main(args): 
    """
    Trains a Grafted DiT model.
    """
    # Load config file
    #args.config_filepath = 'configs/finetuning/imagenet/8k_samples/self_graft.yaml'
    config_filepath = args.config_filepath
    config = load_yaml_file(config_filepath)
    
    # Get arguments
    dit_model_name = config['train_config']['dit_model_name']
    dit_ckpt_path = config['train_config']['dit_ckpt_path']
    dit_ckpt_path = None if dit_ckpt_path=="None" else dit_ckpt_path
    image_size = config['train_config']['image_size']
    graft_indexes = config['train_config']['graft_indexes']
    graft_weights = config['train_config']['graft_weights']
    operator_type = config['operator']['type']
    operator_name = config['operator']['name']
    operator_config_filepath = config['operator']['config_filepath']
    #print(dit_ckpt_path, dit_ckpt_path == None)
    print(f'training grafted {dit_model_name}')

    # Setup PyTorch:
    args.global_seed = config['train_config']['global_seed']
    torch.manual_seed(args.global_seed)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    train_steps = 0
    train_images = 0

    # Setup logging frequency
    num_samples = int(config['train']['num_samples'])
    train_steps_per_epoch = int(math.ceil(num_samples/int(config['train']['batch_size']))) # Number of steps per epoch (250 steps)
    assert args.ckpt_every % args.log_every == 0

    # Setup an experiment folder:
    logger = None
    checkpoint_dir = None
    if accelerator.is_main_process:
        num_samples_for_finetuning = convert_to_thousands_string(num_samples)
        graft_indexes_str_repr = list_to_string(graft_indexes)
        init_type = config['train_config']['init_type']
        run_name =  f'{num_samples_for_finetuning}/finetuning/{init_type}/{operator_name}_{graft_indexes_str_repr}'
        tags = ['imagenet-1k', 'DiT_XL/2', str(args.image_size), f'{num_samples_for_finetuning}_samples', 'finetuning']
        experiment_dir = os.path.join(args.result_dir, f'{operator_name}_{list_to_underscore_string(graft_indexes)}')

        #experiment_dir = os.path.join(args.result_dir, f'{operator_name}_tmp')
        os.makedirs(experiment_dir, exist_ok=True) 

        # Create checkpoint directory
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create local logs
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Setup wandb logs
        if not args.only_init:
            os.environ['WANDB_DIR'] = experiment_dir            
            wandb.init(project='grafting_diffusion_transformers_neurips25', tags=tags, name=run_name)
            wandb.config.update(args)

    model = graft_dit(dit_model_name, dit_ckpt_path, image_size, operator_type,
                    operator_name, operator_config_filepath, graft_indexes, graft_weights, run_all_unit_tests=True)
    
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.resume_training_checkpoint_path is not None:
        resume_state_dict = torch.load(args.resume_training_checkpoint_path, map_location='cuda')
        model.load_state_dict(resume_state_dict['model'], strict=True)
        ema.load_state_dict(resume_state_dict['model'], strict=True) 
        requires_grad(ema, False)
        train_images = resume_state_dict["train_images_seen_so_far"]
        train_steps = resume_state_dict["train_steps"]
        print(f"loaded {resume_state_dict_path} successfully.")

    # Create dataset
    dataset = ImageNet_VAE_WebDataset(**config['train'])
    loader = dataset.get_dataloader()

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train() 
    ema.eval()
    model_sanity_check_print_trainable_params(model, 'ema model', 'ema in eval mode', print_full_list=False)

    # Print trainable parameters (for sanity check)
    model_sanity_check_print_trainable_params(model, 'train', 'before freezing params', print_full_list=False)
    freeze_first_n_layers(model, n=0, only_attn=False) # do full finetuning

    # Print trainable parameters (for sanity check)
    model_sanity_check_print_trainable_params(model, 'train', 'after freezing params')

    # -------------- Training Hyperparameters --------------
    # Load other args
    args.epochs = int(config['train_config']['epochs'])
    args.init_lr = float(config['train_config']['initial_learning_rate'])
    args.wd = float(config['train_config']['wd'])
    #print(args.init_lr, args.end_lr)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr, weight_decay=args.wd)
    
    if args.resume_training_checkpoint_path is not None:
        try:
            #opt.load_state_dict(resume_state_dict['opt'])
            current_param_groups = opt.param_groups[0]['params']
            opt.load_state_dict({
                "state": resume_state_dict["opt"]["state"],
                "param_groups": [{
                    **resume_state_dict["opt"]["param_groups"][0],
                    "params": current_param_groups  # Use the current parameters
                }]
            })
            print('successfully loaded optimizer state')
        except ValueError as e:
            print(f"Error loading optimizer state: {e}")
            # Debugging info
            print(f"Saved param groups: {len(resume_state_dict['opt']['param_groups'])}")
            print(f"Current param groups: {len(opt.param_groups)}")
            print(f"Saved param groups: {resume_state_dict['opt']['param_groups']}")
            print(f"Current param groups: {opt.param_groups}")
            raise Exception

    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scheduler.step() # Set the learning rate of the optimizer to zero.
    print('initial lr', get_lr(opt))

    model, ema, opt, loader, scheduler  = accelerator.prepare(model, ema, opt, loader, scheduler)

    # Save the initialized checkpoint to track weights/ latent sampling
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
        wandb.config.update(args)
        wandb.watch(model, log = 'all', log_freq = 32)

        checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "config_train": config,
                    }
        checkpoint_path = f"{checkpoint_dir}/init.pt"

        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            logger.info(f"Init checkpoint already exists: {checkpoint_path}")

    if args.only_init:
        print(f'Successfully saved the initialized checkpoint, exiting now...')
        sys.exit()

    # Strictly stick to 256 batch size.
    model, train_steps, train_images = train_dit(model, ema, loader, opt, scheduler, device, accelerator, logger, checkpoint_dir, args,
                train_steps = train_steps, train_images = train_images)
    torch.cuda.synchronize()

    # ---------------------
    print(f'After grafting stage 2: train_steps = {train_steps}, train_images = {train_images}')
    
    if accelerator.is_main_process:
        logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="./tmp/")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--only-init",  action='store_true')
    parser.add_argument("--config-filepath", type=str, required=True)
    parser.add_argument("--resume-training-checkpoint-path", type=str, default=None)


    # Custom
    def parse_int_list(value):
        return [int(i) for i in value.split(',') if i.strip().isdigit()]
    

    args = parser.parse_args()
    print(args)
    main(args)
