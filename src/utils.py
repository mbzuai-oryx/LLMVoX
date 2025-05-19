"""
Utility functions for speech prediction model training.
"""
import os
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from contextlib import nullcontext
import transformers
from typing import Dict, Optional, Sequence, List
from src.data import get_batch

def setup_environment(config):
    """Set up the training environment including DDP if applicable."""
    # Check if we're running with distributed data parallel
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        
        # Adjust gradient accumulation steps proportionally 
        gradient_accumulation_steps = config['gradient_accumulation_steps']
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # Single GPU or CPU setup
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        device = config['device']
        gradient_accumulation_steps = config['gradient_accumulation_steps']
    
    # Calculate tokens per iteration for reporting
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Create output directory
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    
    # Set up random seed
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision for faster training on A100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine device type and precision
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Return updated config along with environment variables
    config['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    return {
        'device': device,
        'device_type': device_type,
        'ctx': ctx,
        'master_process': master_process,
        'ddp': ddp,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'tokens_per_iter': tokens_per_iter,
        'gradient_accumulation_steps': gradient_accumulation_steps
    }


@torch.no_grad()
def estimate_loss(model, train_dataloader,context_length,eval_iters, ctx, device, device_type, llm_model):
    """Estimate loss over evaluation iterations."""
    out = {}
    model.eval()
    for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_dataloader,context_length,device,device_type, llm_model)
            with ctx:
                try:
                    logits, loss, _ = model(X, Y)
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(config, it):
    """Get learning rate for current iteration based on schedule."""
    if not config['decay_lr']:
        return config['learning_rate']
    
    # Linear warmup phase
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    
    # After decay iterations, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    
    # In between, use cosine decay
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])


def setup_wandb(config, master_process):
    """Initialize wandb logging if enabled."""
    if config['wandb_log'] and master_process:
        try:
            import wandb
            print(f"Initializing wandb with project: {config['wandb_project']}, run name: {config['wandb_run_name']}")
            wandb.init(project=config['wandb_project'], name=config['wandb_run_name'], config=config)
            return wandb
        except ImportError:
            print("wandb not installed, skipping wandb initialization")
            return None
    return None


def save_checkpoint(config, model, optimizer, iter_num, model_args):
    """Save model checkpoint."""
    raw_model = model.module if isinstance(model, DDP) else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'config': config,
    }
    
    # Save to specified filename or default
    checkpoint_path = os.path.join(config['out_dir'], config.get('checkpoint_filename', 'ckpt.pt'))
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    # Optionally save a backup copy
    if config.get('always_save_checkpoint', False):
        backup_path = os.path.join(config['out_dir'], f'ckpt_{iter_num}.pt')
        print(f"Saving backup checkpoint to {backup_path}")
        torch.save(checkpoint, backup_path)
        
        # Keep only the latest 3 checkpoints, remove older ones
        checkpoint_files = sorted(glob(os.path.join(config['out_dir'], 'ckpt_*.pt')), 
                                  key=os.path.getctime, reverse=True)  # Sort by creation time
        if len(checkpoint_files) > 3:
            old_checkpoints = checkpoint_files[3:]
            for ckpt in old_checkpoints:
                print(f"Removing old checkpoint: {ckpt}")
                os.remove(ckpt)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        
