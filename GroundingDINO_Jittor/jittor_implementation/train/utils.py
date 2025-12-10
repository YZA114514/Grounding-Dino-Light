# Training Utils (Member C)
import os
import sys
import time
import datetime
import random
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

import jittor as jt
from jittor import nn

import torch
import torchvision.transforms as T
from PIL import Image

from .config import TrainingConfig


def seed_everything(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    jt.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: nn.Module, optimizer: jt.optim.Optimizer, scheduler: jt.optim.LRScheduler,
               epoch: int, loss: float, output_dir: str, name: str = "model"):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss
    }
    
    filename = os.path.join(output_dir, f"{name}_{epoch:04d}.pth")
    jt.save(checkpoint, filename)
    
    # Also save as latest
    latest_filename = os.path.join(output_dir, f"{name}_latest.pth")
    jt.save(checkpoint, latest_filename)
    
    return filename


def load_model(model: nn.Module, optimizer: jt.optim.Optimizer = None, scheduler: jt.optim.LRScheduler = None,
               checkpoint_path: str = None, resume_training: bool = True):
    """Load model checkpoint"""
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return 0, float('inf')
    
    checkpoint = jt.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and resume_training:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and resume_training:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss'] if 'loss' in checkpoint else float('inf')
    
    if resume_training:
        start_epoch += 1  # Start from next epoch
    
    return start_epoch, loss


def get_lr(optimizer: jt.optim.Optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not jt.distributed:
        return False
    return jt.distributed.is_initialized()


def get_world_size():
    """Get world size"""
    if not is_dist_avail_and_initialized():
        return 1
    return jt.distributed.get_world_size()


def get_rank():
    """Get rank"""
    if not is_dist_avail_and_initialized():
        return 0
    return jt.distributed.get_rank()


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
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def collate_fn(batch):
    """Collate function for dataloader"""
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[jt.Var]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[jt.Var]):
    # TODO: make this more general
    if tensor_list[0].ndim == 3:
        # TODO: make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = jt.zeros(batch_shape, dtype=dtype)
        mask = jt.ones((b, h, w), dtype=jt.bool)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('Not supported')
    return NestedTensor(tensor, mask)


class MetricLogger(object):
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

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average.
    """

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
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = jt.array([self.count, self.total])
        jt.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = jt.array(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = jt.array(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
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
            for box in targets[i]['boxes'].numpy():
                x1, y1, x2, y2 = box.astype(int)
                img_pil = draw_box(img_pil, (x1, y1, x2, y2), color=(0, 255, 0), width=2)
        
        # Draw prediction boxes
        if 'pred_boxes' in outputs:
            pred_boxes = outputs['pred_boxes'][i]
            pred_logits = outputs['pred_logits'][i]
            
            # Apply sigmoid and threshold
            pred_scores = pred_logits.sigmoid().max(dim=-1)[0]
            pred_labels = pred_logits.sigmoid().argmax(dim=-1)[0]
            
            # Keep only predictions with high confidence
            keep = pred_scores > 0.3
            
            for box, score, label in zip(pred_boxes[keep], pred_scores[keep], pred_labels[keep]):
                x1, y1, x2, y2 = box.numpy().astype(int)
                img_pil = draw_box(img_pil, (x1, y1, x2, y2), color=(255, 0, 0), width=2)
                
                # Add label and score
                text = f"{label.item()}: {score.item():.2f}"
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
        scheduler = jt.optim.MultiStepLR(
            optimizer,
            milestones=[config.lr_drop],
            gamma=0.1
        )
    elif config.lr_scheduler == "cosine":
        scheduler = jt.optim.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    else:
        scheduler = None
    
    return scheduler


def convert_to_jittor_format(data: Dict):
    """Convert data from PyTorch to Jittor format"""
    jittor_data = {}
    
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            jittor_data[key] = jt.array(value.numpy())
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            jittor_data[key] = [jt.array(item.numpy()) for item in value]
        elif isinstance(value, dict):
            jittor_data[key] = {k: (jt.array(v.numpy()) if isinstance(v, torch.Tensor) else v) 
                                for k, v in value.items()}
        else:
            jittor_data[key] = value
    
    return jittor_data