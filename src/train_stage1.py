# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

"""
Grafting: Stage 1 (Activation Distillation)
"""
import os, sys, math, json, gc
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import logging
import yaml

 
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
import torch.nn as nn
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from itertools import islice
import torch.nn.functional as F
import wandb
sys.path.append(f"{os.getcwd()}/src/datasets/") # Append paths

# Import custom modules
from scion_fts_webdataset import Scion_Training_WebDataset
from utils import *
from replacement_factory import ReplacementFactory
from loss_functions_stage1 import get_loss_fn_for_distillation


#################################################################################
#                                  Training Helper Functions                    #
#################################################################################
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

# Define the learning rate scheduler function to warm-up
def lr_lambda(current_step, warmup_steps=2000):
    total_steps = warmup_steps
    if current_step < total_steps:
        return current_step / total_steps
    else:
        return 1


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# For training Scions start with a high learning rate and decay it by a linear factor
def calculate_decay_factor(start_lr, end_lr, num_epochs):
    # decay_factor = (1e-4 / 1e-3) ** (1 / 200)
    decay_factor = (end_lr / start_lr) ** (1 / num_epochs)
    return decay_factor



def calculate_average_loss(metrics):
    """
    Calculate the average loss from the accumulated metrics.

    Args:
    metrics (list or torch.Tensor): A list or tensor containing accumulated metrics.
                                    The format should be [loss1, samples1, loss2, samples2, ..., lossN, samplesN].

    Returns:
    float: The average loss.
    """
    # Separate the losses and the number of samples
    # losses = metrics[0::2]
    # num_samples = metrics[1::2]
    losses = metrics[0]
    losses_scaled = metrics[1]
    losses_unscaled = metrics[2]
    num_samples = metrics[3]

    #print(losses, num_samples)

    # Calculate the total loss and the total number of samples
    total_loss = torch.sum(losses)
    total_loss_scaled = torch.sum(losses_scaled)
    total_loss_unscaled = torch.sum(losses_unscaled)
    total_samples = torch.sum(num_samples)

    # Compute the final average loss
    average_loss = total_loss / total_samples
    average_loss_scaled = total_loss_scaled / total_samples
    average_loss_unscaled = total_loss_unscaled / total_samples

    return average_loss.item(), average_loss_scaled.item(), average_loss_unscaled.item(), total_samples.item()


@torch.no_grad()
def validate(model, val_loader, criterion, device, accelerator):
    model.eval() 

    # mse_loss = nn.MSELoss()
    local_loss_sum_scaled = torch.tensor(0.0, device=device)
    local_loss_sum_unscaled = torch.tensor(0.0, device=device)
    local_sample_count = torch.tensor(0, device=device)

    with torch.no_grad():
        for keys, x, y, adaln_out_fts, t in val_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)

            # Scale params
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out_fts.chunk(6, dim=1)
            gate_msa = gate_msa.to(device).float()

            #Predict
            # clamp 
            #y = torch.clamp(y, min=-10, max=10)
            pred = model(x)

            # Scale losses
            y_scaled = gate_msa.unsqueeze(1)*y
            pred_scaled = gate_msa.unsqueeze(1)*pred

            # Now reshape and calculate loss
            pred_flat = pred.view(pred.size(0), -1)
            pred_scaled_flat = pred_scaled.view(pred_scaled.size(0), -1)
            y_flat = y.view(y.size(0), -1)
            y_scaled_flat = y_scaled.view(y_scaled.size(0), -1)

            # Calculate both scaled and unscaled losses
            loss_scaled = criterion(pred_scaled_flat, y_scaled_flat)
            loss_unscaled = criterion(pred_flat, y_flat)

            # Aggregate losses
            local_sample_count += x.size(0)

            # Additional logs
            local_loss_sum_scaled += loss_scaled.item()*x.size(0)
            local_loss_sum_unscaled += loss_unscaled.item()*x.size(0)

    # Duplicate returns
    return local_loss_sum_unscaled, local_loss_sum_scaled, local_loss_sum_unscaled, local_sample_count


def train(model, opt, train_loader, criterion, device, accelerator, scaled=False, clip_norm_val=0.0):
    model.train() 
    print(f'Scaled set to {scaled}')

    clip_bool = False if int(math.ceil(clip_norm_val)) == 0 else True
    print(f'Clipping set to {clip_bool}/ clip norm value = {clip_norm_val}')

    #mse_loss = nn.MSELoss()
    local_loss_sum = torch.tensor(0.0, device=device)
    local_loss_sum_scaled = torch.tensor(0.0, device=device)
    local_loss_sum_unscaled = torch.tensor(0.0, device=device)
    local_sample_count = torch.tensor(0, device=device)


    for index, (keys, x, y, adaln_out_fts, t) in enumerate(train_loader):
        #print(x.size(), device)
        x = x.to(device).float()
        y = y.to(device).float()
        x = x.squeeze(dim=1)
        y = y.squeeze(dim=1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_out_fts.chunk(6, dim=1)
        gate_msa = gate_msa.to(device).float()

        # Predict
        # clamp 
        #y = torch.clamp(y, min=-10, max=10)
        pred = model(x)

        # Scale losses
        y_scaled = gate_msa.unsqueeze(1)*y
        pred_scaled = gate_msa.unsqueeze(1)*pred

        # Now reshape and calculate loss
        pred_flat = pred.view(pred.size(0), -1)
        pred_scaled_flat = pred_scaled.view(pred_scaled.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        y_scaled_flat = y_scaled.view(y_scaled.size(0), -1)

        # Calculate both scaled and unscaled losses
        loss_scaled = criterion(pred_scaled_flat, y_scaled_flat).detach()
        loss_unscaled = criterion(pred_flat, y_flat).detach()

        # Decide which loss to use for updating parameters
        loss = criterion(pred_scaled_flat, y_scaled_flat) if scaled else criterion(pred_flat, y_flat)

        # Update params
        opt.zero_grad()
        accelerator.backward(loss)

        if clip_bool:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), clip_norm_val) # You might encounter a gradient explosion problem.

        opt.step()

        # Aggregate losses
        local_loss_sum += loss.item() * x.size(0)
        local_sample_count += x.size(0)

        # Additional logs
        local_loss_sum_scaled += loss_scaled.item()*x.size(0)
        local_loss_sum_unscaled += loss_unscaled.item()*x.size(0)

    return local_loss_sum, local_loss_sum_scaled, local_loss_sum_unscaled, local_sample_count



# Helper function
def update_best_loss(gathered_loss_unscaled, best_loss_unscaled, gathered_loss_scaled, best_loss_scaled, scaled_predictor):
    save_best = False
    
    # Update unscaled loss
    if gathered_loss_unscaled < best_loss_unscaled:
        best_loss_unscaled = gathered_loss_unscaled
        if not scaled_predictor:  # Only set save_best if unscaled loss is used
            save_best = True
    
    # Update scaled loss
    if gathered_loss_scaled < best_loss_scaled:
        best_loss_scaled = gathered_loss_scaled
        if scaled_predictor:  # Only set save_best if scaled loss is used
            save_best = True
    
    return best_loss_unscaled, best_loss_scaled, save_best



def main(args): 
    """
    Trains Scion.
    """
    # Load config file
    config_filepath = args.config_filepath
    config = load_yaml_file(config_filepath)
    
    # block_index = config['train']['block_index']
    block_index = args.block_index
    block_name = f'block_{block_index}'
    operator_name = config['operator']['name']
    operator_config_filepath = config['operator']['config_filepath']
    print(f'distiling {block_name}')

    # Setup PyTorch (Use unique seeds for each blocks):
    args.global_seed = (config['train_config']['global_seed'] + int(block_index))
    torch.manual_seed(args.global_seed)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Loss functions
    criterion_val = nn.MSELoss()
    if args.huber_delta is not None:
        criterion_train = get_loss_fn_for_distillation({'loss': args.loss, 'delta': args.huber_delta})
        loss_seed_identifier = f'{args.loss}_delta={args.huber_delta}/scaled_pred={args.scaled_predictor}'
    else:
        criterion_train = get_loss_fn_for_distillation({'loss': args.loss})
        loss_seed_identifier = f'{args.loss}/scaled_pred={args.scaled_predictor}'

    print(criterion_train)
    
    # Setup an experiment folder:
    if accelerator.is_main_process:
        num_samples_for_scion_training = convert_to_thousands_string(int(config['train']['num_samples']))
        # run_name =  f'{num_samples_for_scion_training}/scion/{operator_name}/{block_name}'
        run_name =  f'{num_samples_for_scion_training}/scion/{operator_name}/{block_name}/{loss_seed_identifier}/seed={args.global_seed}'
        tags = ['imagenet-1k', 'DiT_XL/2', '256', f'{num_samples_for_scion_training}_samples']
        experiment_dir = os.path.join(args.result_dir, operator_name, block_name, loss_seed_identifier, f'seed={args.global_seed}')
        os.makedirs(experiment_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)

        # Create checkpoint directory
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup wandb
        os.environ['WANDB_DIR'] = experiment_dir

        # Create local and wandb logs
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project='grafting_stage1', tags=tags, name=run_name)
        wandb.config.update(args)


    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    # --------- Model Configs ---------
    # Create model (Scion)
    scion_obj = ReplacementFactory(operator_name, operator_config_filepath)
    model = scion_obj.get_operator().to(device)
    
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    #ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training (Ignore ema for stage 1)
    #requires_grad(ema, False)

    if accelerator.is_main_process:
        logger.info(f"Scion Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
        # Print trainable parameters (for sanity check)
        trainable_params = [ n for (n, p) in model.named_parameters() if p.requires_grad ]
        print(f'List of trainable parameters')
        for element in trainable_params:
            print(element)

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {total_trainable_params}')


    # --------- Dataset Configs ---------
    # Train configs
    config_train = load_yaml_file(config_filepath)['train']
    sha_key_train = config_train['sha_key']
    config_train = remove_keys(config_train, ['sha_key'])
    config_train['block_index'] = block_index

    # Val configs
    config_val = load_yaml_file(config_filepath)['val']
    sha_key_val = config_val['sha_key']
    config_val = remove_keys(config_val, ['sha_key'])
    config_val['block_index'] = block_index

    
    # Create training dataloader
    train_dataset = Scion_Training_WebDataset(**config_train)
    train_loader = train_dataset.get_dataloader()

    # Create val dataloader
    val_dataset = Scion_Training_WebDataset(**config_val)
    val_loader = val_dataset.get_dataloader()

    
    # -------------- Training Hyperparameters --------------
    # Load other args
    args.epochs = int(config['train_config']['epochs'])
    args.init_lr = float(config['train_config']['initial_learning_rate'])*accelerator.num_processes
    args.end_lr = float(config['train_config']['min_learning_rate'])*accelerator.num_processes
    args.wd = float(config['train_config']['wd'])
    args.clip_norm_val = float(config['train_config']['clip_norm_val'])

    args.shard_batch_size = float(config['train']['batch_size']) # this is per shard
    args.global_batch_size = float(config['train']['batch_size'])*8.0 # this is total shards
    args.num_workers = float(config['train']['num_workers'])

    # Set up variables.
    init_lr = args.init_lr
    end_lr = args.end_lr
    wd = args.wd
    num_epochs = args.epochs
    print(init_lr, end_lr, num_epochs, wd)

    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=wd)
    
    # Create the scheduler
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=args.end_lr, eps=1e-8, verbose=True)
    step_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 100, 150], gamma=0.5, verbose=True)
    print('initial lr', get_lr(opt))

    # Prepare model for training:
    model.train()
    model, opt, plateau_scheduler, step_scheduler, train_loader, val_loader = accelerator.prepare(model, opt, plateau_scheduler, step_scheduler, train_loader, val_loader)
    
    # Variables for monitoring/logging purposes:
    train_images = 0
    log_steps = 0
    best_val_loss_unscaled, best_val_loss_scaled = 1e9, 1e9
    best_train_loss_unscaled, best_train_loss_scaled = 1e9, 1e9 # For tracking MSE loss on the training set 
    VAL_FREQUENCY=10 # evaluate every 10 epochs (this signal is used for selecting the best.pt)
    start_time = time()
    
    # Save the initialized checkpoint to track hyena weights
    if accelerator.is_main_process:
        logger.info(f"Training for {num_epochs} epochs...")
        wandb.config.update(args)
        wandb.watch(model, log = 'all', log_freq = 10)
        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        checkpoint = {
                        # "model": model.module.state_dict(),
                        "model": model_state_dict,
                        # "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "config_train": config_train,
                        "config_val": config_val,
                    }
        checkpoint_path = f"{checkpoint_dir}/init.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


    for epoch in range(num_epochs):

        # Reset monitoring variables:
        log_steps = 0
        start_time = time()

        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch+1}...")
        
        # Training Statistics
        local_loss_sum_train, local_loss_scaled_sum_train, local_loss_unscaled_sum_train, \
                local_sample_count_train = train(model, opt, train_loader, criterion_train, device, accelerator, args.scaled_predictor, args.clip_norm_val)
        gathered_items_train = accelerator.gather_for_metrics((local_loss_sum_train, local_loss_scaled_sum_train, 
                                local_loss_unscaled_sum_train, local_sample_count_train))
        gathered_loss_train, gathered_loss_scaled_train, gathered_loss_unscaled_train, \
                    gathered_total_samples_train = calculate_average_loss(gathered_items_train)
        train_images += gathered_total_samples_train
        log_steps += 1
        
        # Measure training speed
        end_time = time()
        steps_per_sec = log_steps / (end_time - start_time)

        # Validation Statistics
        if (epoch+1)%VAL_FREQUENCY == 0 or epoch == 0:
            # Evaluate on Validation set
            local_loss_sum_val, local_loss_scaled_sum_val, local_loss_unscaled_sum_val, \
                        local_sample_count_val = validate(model, val_loader, criterion_val, device, accelerator)
            gathered_items_val = accelerator.gather_for_metrics((local_loss_sum_val, local_loss_scaled_sum_val, 
                                local_loss_unscaled_sum_val, local_sample_count_val))
            gathered_loss_val, gathered_loss_scaled_val, gathered_loss_unscaled_val, \
                         gathered_total_samples_val = calculate_average_loss(gathered_items_val)


            # Evaluate on Train set
            local_loss_sum_train_eval, local_loss_scaled_sum_train_eval, local_loss_unscaled_sum_train_eval, \
                local_sample_count_train_eval = validate(model, train_loader, criterion_val, device, accelerator)
            gathered_items_train_eval = accelerator.gather_for_metrics((local_loss_sum_train_eval, local_loss_scaled_sum_train_eval, 
                                    local_loss_unscaled_sum_train_eval, local_sample_count_train_eval))
            gathered_loss_train_eval, gathered_loss_scaled_train_eval, gathered_loss_unscaled_train_eval, \
                        gathered_total_samples_train_eval = calculate_average_loss(gathered_items_train_eval)

            plateau_scheduler.step(gathered_loss_scaled_val)
            torch.cuda.empty_cache()
            gc.collect()

        step_scheduler.step(epoch)

        if accelerator.is_main_process:
            logger.info(f"(Images seen so far: {train_images:07d}), Train Loss ({args.loss}): {gathered_loss_train:.4f}, Train Loss (Eval): {gathered_loss_train_eval:.4f}, Val Loss: {gathered_loss_val:.4f}, Epochs/Sec: {steps_per_sec:.2f}")

        # Save best checkpoints
        if accelerator.is_main_process:
            save_best_val=False
            save_best_train=False

            # For validation loss
            best_val_loss_unscaled, best_val_loss_scaled, save_best_val = update_best_loss(
                gathered_loss_unscaled_val,
                best_val_loss_unscaled,
                gathered_loss_scaled_val,
                best_val_loss_scaled,
                args.scaled_predictor
            )

            # For training loss
            best_train_loss_unscaled, best_train_loss_scaled, save_best_train = update_best_loss(
                gathered_loss_unscaled_train_eval,
                best_train_loss_unscaled,
                gathered_loss_scaled_train_eval,
                best_train_loss_scaled,
                args.scaled_predictor
            )

            model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            # Save checkpoint
            checkpoint = {
                    # "model": model.module.state_dict(),
                    "model": model_state_dict,
                    "opt": opt.state_dict(),
                    "args": args,
                    "config_train": config_train,
                    "config_val": config_val,
                    "epoch": epoch+1,
                    "train_images_seen_so_far": train_images,

                    # Record training losses (Can be MSE/ MAE/ Huber)
                    r"train/loss": gathered_loss_train,
                    r"train/y_loss": gathered_loss_unscaled_train,
                    r"train/y_alpha_loss": gathered_loss_scaled_train,
                    
                    # Record evaluation loss on Training set
                    r"train_eval/y_loss": gathered_loss_unscaled_train_eval,
                    r"train_eval/y_alpha_loss": gathered_loss_scaled_train_eval,
                    
                    # Record validation loss
                    r"val/y_loss": gathered_loss_unscaled_val,
                    r"val/y_alpha_loss": gathered_loss_scaled_val,

                    # Best loss
                    r"best/val/y_loss": best_val_loss_unscaled,
                    r"best/val/y_alpha_loss": best_val_loss_scaled,

                    r"best/train/y_loss": best_train_loss_unscaled,
                    r"best/train/y_alpha_loss": best_train_loss_scaled,

                }

            if save_best_val:
                checkpoint_path = f"{checkpoint_dir}/best_val.pt"
                torch.save(checkpoint, checkpoint_path)
                console_loss = best_val_loss_unscaled if not args.scaled_predictor else best_val_loss_scaled
                logger.info(f"Saved best checkpoint with val loss = {console_loss} to {checkpoint_path}")
            
            if save_best_train:
                checkpoint_path = f"{checkpoint_dir}/best_train.pt"
                torch.save(checkpoint, checkpoint_path)
                console_loss = best_train_loss_unscaled if not args.scaled_predictor else best_train_loss_scaled
                logger.info(f"Saved best checkpoint with train loss = {console_loss} to {checkpoint_path}")


            if (epoch+1)%args.log_every == 0:
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}_train_images_seen_so_far_{train_images:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                #logger.info(f"(step={train_images:07d}) Train Loss: {avg_mse_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")


            # Wandb log (validation loss needs to be handled at correct intervals)
            if (epoch+1)%VAL_FREQUENCY == 0 or epoch == 0:
                metrics_dict = {
                    "train_images_seen_so_far": train_images,
                    "epoch": epoch+1,
                    "lr": get_lr(opt),

                   # Record training losses (Can be MSE/ MAE/ Huber)
                    r"train/loss": gathered_loss_train,
                    r"train/y_loss": gathered_loss_unscaled_train,
                    r"train/y_alpha_loss": gathered_loss_scaled_train,
                    
                    # Record evaluation loss on Training set
                    r"train_eval/y_loss": gathered_loss_unscaled_train_eval,
                    r"train_eval/y_alpha_loss": gathered_loss_scaled_train_eval,
                    
                    # Record validation loss
                    r"val/y_loss": gathered_loss_unscaled_val,
                    r"val/y_alpha_loss": gathered_loss_scaled_val,

                    # Best loss
                    r"best/val/y_loss": best_val_loss_unscaled,
                    r"best/val/y_alpha_loss": best_val_loss_scaled,

                    r"best/train/y_loss": best_train_loss_unscaled,
                    r"best/train/y_alpha_loss": best_train_loss_scaled,

                }
            else:
                metrics_dict = {
                    "train_images_seen_so_far": train_images,
                    "epoch": epoch+1,
                    "lr": get_lr(opt),

                    # Record training losses (Can be MSE/ MAE/ Huber)
                    r"train/loss": gathered_loss_train,
                    r"train/y_loss": gathered_loss_unscaled_train,
                    r"train/y_alpha_loss": gathered_loss_scaled_train,

                    # Best loss
                    r"best/val/y_loss": best_val_loss_unscaled,
                    r"best/val/y_alpha_loss": best_val_loss_scaled,

                    r"best/train/y_loss": best_train_loss_unscaled,
                    r"best/train/y_alpha_loss": best_train_loss_scaled
                }
            wandb_log(metrics_dict)
            
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="./DiT_results/")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--config-filepath", type=str, required=True)
    parser.add_argument('--block-index', type=int, required=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--loss", type=str, choices=["l2", "l1", "huber"], required=True)
    parser.add_argument('--huber_delta', type=float, default=None, help='delta value for Huber Loss')
    parser.add_argument('--scaled_predictor', action='store_true', help='A boolean flag')

    # Custom
    def parse_int_list(value):
        return [int(i) for i in value.split(',') if i.strip().isdigit()]

    args = parser.parse_args()
    main(args)
