# Training Utils (Member C) - Pure Jittor Implementation
import os
import sys
import time
import datetime
import random
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

import jittor as jt
from jittor import nn
from PIL import Image

from .config import TrainingConfig


def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.seed(seed)


def save_model(model: nn.Module, optimizer: jt.optim.Optimizer, scheduler,
               epoch: int, loss: float, output_dir: str, name: str = "model"):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss
    }
    
    filename = os.path.join(output_dir, f"{name}_{epoch:04d}.pkl")
    jt.save(checkpoint, filename)
    
    # Also save as latest
    latest_filename = os.path.join(output_dir, f"{name}_latest.pkl")
    jt.save(checkpoint, latest_filename)
    
    print(f"Saved checkpoint to {filename}")
    return filename


def load_model(model: nn.Module, optimizer: jt.optim.Optimizer = None, scheduler = None,
               checkpoint_path: str = None, resume_training: bool = True):
    """Load model checkpoint"""
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and resume_training:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and resume_training:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    if resume_training:
        start_epoch += 1  # Start from next epoch
    
    print(f"Loaded checkpoint from epoch {start_epoch - 1}")
    return start_epoch, loss


def get_lr(optimizer: jt.optim.Optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    try:
        return jt.in_mpi
    except:
        return False


def get_world_size():
    """Get world size"""
    if not is_dist_avail_and_initialized():
        return 1
    return jt.world_size


def get_rank():
    """Get rank"""
    if not is_dist_avail_and_initialized():
        return 0
    return jt.rank


def is_main_process():
    """Check if current process is the main process"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save on master process"""
    if is_main_process():
        jt.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """Setup for distributed training"""
    import logging
    if is_master:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)


def init_distributed_mode(args):
    """Initialize distributed training for Jittor"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    
    # Jittor distributed initialization
    jt.distributed.init(args.rank, args.world_size)
    setup_for_distributed(args.rank == 0)


def collate_fn(batch):
    """Collate function for dataloader"""
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    """Get max value along each axis"""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """Container for tensors with masks"""
    def __init__(self, tensors, mask: Optional[jt.Var]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[jt.Var]):
    """Create nested tensor from list of tensors"""
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = jt.zeros(batch_shape, dtype=dtype)
        mask = jt.ones((b, h, w), dtype=jt.bool)
        for i, img in enumerate(tensor_list):
            tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
            mask[i, : img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('Not supported')
    return NestedTensor(tensor, mask)


class MetricLogger(object):
    """Logger for tracking metrics"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jt.Var):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters.setdefault(k, SmoothedValue(window_size=20, fmt="{value:.4f} (avg: {avg:.4f})"))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque_len = window_size
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        while len(self.deque) > self.deque_len:
            self.deque.pop(0)
        self.total += value * n
        self.count += n

    def synchronize_between_processes(self):
        """Synchronize between distributed processes"""
        if not is_dist_avail_and_initialized():
            return
        t = jt.array([self.count, self.total])
        # Jittor distributed all_reduce
        if hasattr(jt, 'mpi_all_reduce'):
            t = jt.mpi_all_reduce(t)
        t = t.numpy()
        self.count = int(t[0])
        self.total = float(t[1])

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0
        d = sorted(self.deque)
        return d[len(d) // 2]

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def log_images(images: jt.Var, targets: List[Dict], outputs: Dict, output_dir: str, 
               iteration: int, prefix: str = "", max_images: int = 8):
    """Log images with predictions and ground truth"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy for visualization
    images_np = (images.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    
    for i in range(min(images_np.shape[0], max_images)):
        img = images_np[i]
        img_pil = Image.fromarray(img)
        
        # Draw ground truth boxes
        if i < len(targets) and 'boxes' in targets[i]:
            boxes = targets[i]['boxes']
            if isinstance(boxes, jt.Var):
                boxes = boxes.numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                img_pil = draw_box(img_pil, (x1, y1, x2, y2), color=(0, 255, 0), width=2)
        
        # Draw prediction boxes
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes'][i]
            pred_logits = outputs['pred_logits'][i]
            
            if isinstance(pred_boxes, jt.Var):
                pred_boxes = pred_boxes.numpy()
            if isinstance(pred_logits, jt.Var):
                pred_logits = pred_logits.numpy()
            
            # Apply sigmoid and threshold
            pred_scores = 1 / (1 + np.exp(-pred_logits))  # sigmoid
            max_scores = pred_scores.max(axis=-1)
            
            # Keep only predictions with high confidence
            keep = max_scores > 0.3
            
            for j in range(len(pred_boxes)):
                if keep[j]:
                    box = pred_boxes[j]
                    score = max_scores[j]
                    x1, y1, x2, y2 = box.astype(int)
                    img_pil = draw_box(img_pil, (x1, y1, x2, y2), color=(255, 0, 0), width=2)
                    text = f"{score:.2f}"
                    img_pil = draw_text(img_pil, (x1, y1 - 5), text, color=(255, 0, 0))
        
        # Save image
        img_path = os.path.join(output_dir, f"{prefix}iter_{iteration:06d}_{i}.jpg")
        img_pil.save(img_path)


def draw_box(img: Image.Image, box: Tuple[int, int, int, int], color=(255, 0, 0), width=1):
    """Draw bounding box on image"""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=color, width=width)
    return img


def draw_text(img: Image.Image, position: Tuple[int, int], text: str, color=(255, 0, 0)):
    """Draw text on image"""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(img)
    draw.text(position, text, fill=color, font=font)
    return img


def create_logger(log_dir: str, name: str = "training"):
    """Create logger for training"""
    import logging
    from logging.handlers import RotatingFileHandler
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Create file handler
    fh = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


def log_metrics(metrics: Dict[str, float], step: int, logger=None, use_wandb=False):
    """Log metrics"""
    if logger:
        log_str = "Step {}: ".format(step)
        for k, v in metrics.items():
            log_str += f"{k}: {v:.4f} "
        logger.info(log_str)
    
    if use_wandb:
        import wandb
        wandb.log(metrics, step=step)


def adjust_learning_rate(optimizer, epoch: int, config: TrainingConfig):
    """Adjust learning rate based on epoch"""
    if epoch < config.warmup_steps:
        # Linear warmup
        lr = config.lr * (epoch + 1) / config.warmup_steps
    elif epoch < config.lr_drop:
        lr = config.lr
    else:
        lr = config.lr * 0.1  # Decay by 10x
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def get_parameter_groups(model: nn.Module, config: TrainingConfig):
    """Get parameter groups for optimizer with different learning rates"""
    # Backbone parameters
    backbone_params = []
    
    # Linear projection parameters
    linear_proj_params = []
    
    # Other parameters
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if this is a backbone parameter
        is_backbone = any(name.startswith(bp) for bp in config.lr_backbone_names)
        
        # Check if this is a linear projection parameter
        is_linear_proj = any(name.startswith(lp) for lp in config.lr_linear_proj_names)
        
        if is_backbone:
            backbone_params.append(param)
        elif is_linear_proj:
            linear_proj_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    parameter_groups = [
        {"params": other_params, "lr": config.lr},
        {"params": backbone_params, "lr": config.lr_backbone},
        {"params": linear_proj_params, "lr": config.lr * config.lr_linear_proj_mult},
    ]
    
    return parameter_groups


def create_optimizer(model: nn.Module, config: TrainingConfig):
    """Create optimizer"""
    parameter_groups = get_parameter_groups(model, config)
    
    optimizer = jt.optim.AdamW(
        parameter_groups,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def create_scheduler(optimizer: jt.optim.Optimizer, config: TrainingConfig):
    """Create learning rate scheduler"""
    if config.lr_scheduler == "step":
        scheduler = jt.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[config.lr_drop],
            gamma=0.1
        )
    elif config.lr_scheduler == "cosine":
        scheduler = jt.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    else:
        scheduler = None
    
    return scheduler
