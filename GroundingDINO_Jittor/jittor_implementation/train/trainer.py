# Trainer (Member C)
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Union, Tuple

import jittor as jt
from jittor import nn

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Import project modules
from .config import TrainingConfig, create_train_config, get_args_parser
from .utils import (
    seed_everything, save_model, load_model, get_lr, MetricLogger, 
    collate_fn, nested_tensor_from_tensor_list, create_logger, 
    log_metrics, adjust_learning_rate, create_optimizer, create_scheduler,
    log_images, convert_to_jittor_format, SmoothedValue
)
from ..data import build_dataset, get_dataloader
from ..losses import GroundingLoss, SetCriterion
from ..eval import evaluate_lvis
from ..models import GroundingDINO
from ..models.text_encoder import BERTWrapper
from ..models.fusion import FeatureFusion
from ..models.query import LanguageGuidedQuery


class Trainer:
    """Trainer for GroundingDINO model"""
    
    def __init__(self, model: nn.Module, text_encoder: BERTWrapper, 
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 criterion: nn.Module, optimizer: jt.optim.Optimizer,
                 scheduler: Optional[jt.optim.LRScheduler],
                 config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            model: GroundingDINO model
            text_encoder: BERT text encoder
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
        """
        self.model = model
        self.text_encoder = text_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # Set device
        self.device = jt.device(config.device)
        self.model.to(self.device)
        self.text_encoder.to(self.device)
        self.criterion.to(self.device)
        
        # Initialize logger
        self.logger = create_logger(config.output_dir, "training")
        
        # Initialize Weights & Biases if needed
        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.to_dict(),
                name=config.model_name
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_metric = 0.0
    
    def train_one_epoch(self, epoch: int):
        """Train model for one epoch"""
        self.model.train()
        self.text_encoder.train()
        self.criterion.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = self.config.log_interval
        
        # Adjust learning rate
        if self.config.lr_scheduler == "step":
            if epoch < self.config.warmup_steps:
                # Linear warmup
                lr = self.config.lr * (epoch + 1) / self.config.warmup_steps
            elif epoch < self.config.lr_drop:
                lr = self.config.lr
            else:
                lr = self.config.lr * 0.1  # Decay by 10x
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        for i, (samples, targets, texts) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            # Convert to Jittor tensors if needed
            if isinstance(samples, torch.Tensor):
                samples = jt.array(samples.numpy())
            
            # Convert targets to Jittor format
            jittor_targets = []
            for target in targets:
                jittor_target = {}
                for key, value in target.items():
                    if isinstance(value, torch.Tensor):
                        jittor_target[key] = jt.array(value.numpy())
                    else:
                        jittor_target[key] = value
                jittor_targets.append(jittor_target)
            
            # Move to device
            samples = samples.to(self.device)
            for target in jittor_targets:
                for key, value in target.items():
                    if isinstance(value, jt.Var):
                        target[key] = value.to(self.device)
            
            # Process text
            text_dict = self.text_encoder(texts)
            
            # Move text features to device
            for key, value in text_dict.items():
                if isinstance(value, jt.Var):
                    text_dict[key] = value.to(self.device)
            
            # Forward pass
            outputs = self.model(samples, text_dict)
            
            # Compute loss
            loss_dict = self.criterion(outputs, jittor_targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Backward pass
            self.optimizer.step(losses)
            
            # Log metrics
            metric_logger.update(loss=losses.item(), **loss_dict)
            metric_logger.update(lr=get_lr(self.optimizer))
            
            # Update global step
            self.global_step += 1
            
            # Log images if needed
            if i % self.config.log_image_interval == 0 and self.config.debug:
                log_images(
                    samples, 
                    jittor_targets, 
                    outputs,
                    os.path.join(self.config.output_dir, "debug_images"),
                    self.global_step,
                    prefix="train_"
                )
            
            # Log metrics to wandb if needed
            if self.use_wandb and i % print_freq == 0:
                import wandb
                wandb.log({
                    "train/loss": losses.item(),
                    "train/lr": get_lr(self.optimizer),
                    **{f"train/{k}": v.item() for k, v in loss_dict.items()}
                }, step=self.global_step)
        
        # Sync metrics
        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def evaluate(self, epoch: int):
        """Evaluate model on validation set"""
        with jt.no_grad():
            if self.val_loader is None:
                return {}
        
        self.model.eval()
        self.text_encoder.eval()
        self.criterion.eval()
        
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        
        print_freq = 20
        all_outputs = []
        all_targets = []
        
        for i, (samples, targets, texts) in enumerate(metric_logger.log_every(self.val_loader, print_freq, header)):
            # Convert to Jittor tensors if needed
            if isinstance(samples, torch.Tensor):
                samples = jt.array(samples.numpy())
            
            # Convert targets to Jittor format
            jittor_targets = []
            for target in targets:
                jittor_target = {}
                for key, value in target.items():
                    if isinstance(value, torch.Tensor):
                        jittor_target[key] = jt.array(value.numpy())
                    else:
                        jittor_target[key] = value
                jittor_targets.append(jittor_target)
            
            # Move to device
            samples = samples.to(self.device)
            for target in jittor_targets:
                for key, value in target.items():
                    if isinstance(value, jt.Var):
                        target[key] = value.to(self.device)
            
            # Process text
            text_dict = self.text_encoder(texts)
            
            # Move text features to device
            for key, value in text_dict.items():
                if isinstance(value, jt.Var):
                    text_dict[key] = value.to(self.device)
            
            # Forward pass
            outputs = self.model(samples, text_dict)
            
            # Compute loss
            loss_dict = self.criterion(outputs, jittor_targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Log metrics
            metric_logger.update(loss=losses.item(), **loss_dict)
            
            # Store outputs and targets for evaluation
            all_outputs.append(outputs)
            all_targets.append(jittor_targets)
        
        # Sync metrics
        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        
        # Compute evaluation metrics
        eval_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        # Run LVIS evaluation if configured
        if self.config.eval and self.val_loader is not None:
            # Convert outputs and targets to format expected by evaluator
            # This is a simplified version - actual implementation may require more complex conversion
            lvis_metrics = evaluate_lvis(
                self.model,
                self.val_loader,
                self.text_encoder,
                ann_file=os.path.join(self.config.data_path, "lvis_val.json"),
                output_dir=os.path.join(self.config.output_dir, "eval_results")
            )
            eval_metrics.update(lvis_metrics)
        
        # Log metrics to wandb if needed
        if self.use_wandb:
            import wandb
            wandb_dict = {f"val/{k}": v for k, v in eval_metrics.items()}
            wandb.log(wandb_dict, step=self.global_step)
        
        return eval_metrics
    
    def train(self):
        """Run complete training loop"""
        # Load checkpoint if specified
        if self.config.resume:
            start_epoch, _ = load_model(
                self.model, self.optimizer, self.scheduler,
                self.config.resume, resume_training=True
            )
            self.current_epoch = start_epoch
            self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        # Main training loop
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_one_epoch(epoch)
            
            # Adjust learning rate if needed
            if self.scheduler is not None and self.config.lr_scheduler != "step":
                self.scheduler.step()
            
            # Evaluate if needed
            val_metrics = {}
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(epoch)
            
            # Save checkpoint if needed
            if (epoch + 1) % self.config.save_interval == 0:
                save_model(
                    self.model, self.optimizer, self.scheduler,
                    epoch, train_metrics.get('loss', float('inf')),
                    self.config.checkpoint_dir, "groundingdino"
                )
            
            # Save best model if needed
            if 'AP' in val_metrics and val_metrics['AP'] > self.best_metric:
                self.best_metric = val_metrics['AP']
                save_model(
                    self.model, self.optimizer, self.scheduler,
                    epoch, train_metrics.get('loss', float('inf')),
                    self.config.checkpoint_dir, "groundingdino_best"
                )
            elif train_metrics.get('loss', float('inf')) < self.best_loss:
                self.best_loss = train_metrics.get('loss', float('inf'))
                save_model(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_loss,
                    self.config.checkpoint_dir, "groundingdino_best_loss"
                )
        
        # Save final model
        save_model(
            self.model, self.optimizer, self.scheduler,
            self.config.epochs - 1, train_metrics.get('loss', float('inf')),
            self.config.checkpoint_dir, "groundingdino_final"
        )
        
        self.logger.info("Training completed")


def build_model(config: TrainingConfig):
    """Build model based on configuration"""
    # Initialize model components
    text_encoder = BERTWrapper(
        model_name=config.text_encoder_type,
        max_text_len=config.max_text_len
    )
    
    # Initialize main model
    model = GroundingDINO(
        backbone=None,  # Will be initialized later based on config
        transformer=None,  # Will be initialized later based on config
        num_queries=config.num_queries,
        num_feature_levels=config.num_feature_levels,
        nheads=config.nheads,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        activation=config.activation
    )
    
    # Set feature map for text encoder
    text_encoder.set_feat_map(model.text_feat_map)
    
    return model, text_encoder


def build_criterion(config: TrainingConfig):
    """Build loss criterion based on configuration"""
    return SetCriterion(
        num_classes=config.num_classes,
        weight_dict={"loss_ce": config.loss_ce, "loss_bbox": config.loss_bbox, "loss_giou": config.loss_giou},
        losses=["labels", "boxes", "giou"],
        focal_alpha=config.matcher.get("focal_alpha", 0.25)
    )


def main(args: argparse.Namespace):
    """Main training function"""
    # Create configuration
    config = create_train_config(args)
    
    # Set random seed
    if config.seed is not None:
        seed_everything(config.seed)
    
    # Print configuration
    print(f"Training configuration: {config.to_dict()}")
    
    # Initialize distributed training if needed
    if config.world_size > 1:
        jt.distributed.init(config.rank, config.world_size, config.dist_url)
    
    # Build model
    model, text_encoder = build_model(config)
    
    # Build criterion
    criterion = build_criterion(config)
    
    # Build datasets and dataloaders
    train_dataset = build_dataset('train', config)
    val_dataset = build_dataset('val', config)
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler_type='distributed' if config.world_size > 1 else 'lvis',
        sampler_kwargs={'samples_per_epoch': 1000} if config.dataset_file == 'lvis' else {}
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler_type='distributed' if config.world_size > 1 else 'default',
        sampler_kwargs={}
    )
    
    # Build optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)