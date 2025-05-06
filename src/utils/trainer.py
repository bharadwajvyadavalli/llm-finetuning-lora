import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from src.utils.logging import setup_logging

logger = setup_logging()

def setup_wandb(project_name="llm-lora"):
    try:
        wandb.init(project=project_name)
        return wandb
    except Exception as e:
        logger.error(f"WandB init failed: {e}")
        raise

def checkpoint(model, optimizer, epoch, path):
    try:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict() if optimizer else None
        }, path)
        logger.info(f"Checkpoint saved at {path}")
    except Exception as e:
        logger.error(f"Checkpoint failed: {e}")

def enable_ddp(model):
    try:
        torch.distributed.init_process_group("nccl")
        return DDP(model.cuda(), device_ids=[torch.cuda.current_device()])
    except Exception as e:
        logger.error(f"DDP setup failed: {e}")
        raise
