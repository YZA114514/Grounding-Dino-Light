# Trainer (Member C) - Pure Jittor Implementation
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Union, Tuple

import jittor as jt
from jittor import nn

# Import project modules
from .config import TrainingConfig, create_train_config, get_args_parser
from .utils import (
    seed_everything, save_model, load_model, get_lr, MetricLogger, 
    collate_fn, nested_tensor_from_tensor_list, create_logger, 
    log_metrics, adjust_learning_rate, create_optimizer, create_scheduler,
    log_images, SmoothedValue
)
from ..data import build_dataset, get_dataloader
from ..losses import GroundingLoss, SetCriterion
from ..eval import evaluate_lvis
from ..models import GroundingDINO
from ..models.text_encoder import BERTWrapper
from ..models.fusion import FeatureFusion
from ..models.query import LanguageGuidedQuery


class Trainer:
    """Trainer for GroundingDINO model (Pure Jittor)"""
    
    def __init__(self, model: nn.Module, text_encoder: BERTWrapper, 
                 train_loader, val_loader,
                 criterion: nn.Module, optimizer: jt.optim.Optimizer,
                 scheduler,
                 config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            model: GroundingDINO model
            text_encoder: BERT text encoder
            train_loader: Training data loader (Jittor Dataset)
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
        
        # Set CUDA flag for Jittor
        if config.device == "cuda":
            jt.flags.use_cuda = 1
        
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
        
        batch_idx = 0
        for samples, targets, texts in self.train_loader:
            # samples and targets are already Jittor Vars from the dataset
            
            # Process text
            text_dict = self.text_encoder(texts)
            
            # Forward pass
            outputs = self.model(samples, text_dict)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Backward pass
            self.optimizer.step(losses)
            
            # Log metrics
            loss_values = {k: v.item() if isinstance(v, jt.Var) else v for k, v in loss_dict.items()}
            metric_logger.update(loss=losses.item(), **loss_values)
            metric_logger.update(lr=get_lr(self.optimizer))
            
            # Update global step
            self.global_step += 1
            
            # Log images if needed
            if batch_idx % self.config.log_image_interval == 0 and self.config.debug:
                log_images(
                    samples, 
                    targets, 
                    outputs,
                    os.path.join(self.config.output_dir, "debug_images"),
                    self.global_step,
                    prefix="train_"
                )
            
            # Log metrics to wandb if needed
            if self.use_wandb and batch_idx % print_freq == 0:
                import wandb
                wandb.log({
                    "train/loss": losses.item(),
                    "train/lr": get_lr(self.optimizer),
                    **{f"train/{k}": v for k, v in loss_values.items()}
                }, step=self.global_step)
            
            # Print progress
            if batch_idx % print_freq == 0:
                print(f"{header} [{batch_idx}/{len(self.train_loader)}] {metric_logger}")
            
            batch_idx += 1
        
        print(f"Averaged stats: {metric_logger}")
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @jt.no_grad()
    def evaluate(self, epoch: int):
        """Evaluate model on validation set"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.text_encoder.eval()
        
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        
        print_freq = 20
        all_outputs = []
        all_targets = []
        
        batch_idx = 0
        for samples, targets, texts in self.val_loader:
            # Process text
            text_dict = self.text_encoder(texts)
            
            # Forward pass
            outputs = self.model(samples, text_dict)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # Log metrics
            loss_values = {k: v.item() if isinstance(v, jt.Var) else v for k, v in loss_dict.items()}
            metric_logger.update(loss=losses.item(), **loss_values)
            
            # Store outputs and targets for evaluation
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            if batch_idx % print_freq == 0:
                print(f"{header} [{batch_idx}/{len(self.val_loader)}] {metric_logger}")
            
            batch_idx += 1
        
        print(f"Averaged stats: {metric_logger}")
        
        # Compute evaluation metrics
        eval_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        # Run LVIS evaluation if configured
        if self.config.eval and self.val_loader is not None:
            try:
                lvis_metrics = evaluate_lvis(
                    self.model,
                    self.val_loader,
                    self.text_encoder,
                    ann_file=os.path.join(self.config.data_path, "lvis_val.json"),
                    output_dir=os.path.join(self.config.output_dir, "eval_results")
                )
                eval_metrics.update(lvis_metrics)
            except Exception as e:
                print(f"Warning: LVIS evaluation failed: {e}")
        
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
    # Import backbone
    from ..models.backbone import SwinTransformer
    
    # Initialize backbone
    backbone = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False
    )
    
    # Initialize text encoder
    text_encoder = BERTWrapper(
        model_name=config.text_encoder_type,
        max_text_len=config.max_text_len
    )
    
    # Initialize main model
    model = GroundingDINO(
        backbone=backbone,
        num_queries=config.num_queries,
        hidden_dim=config.hidden_dim,
        num_feature_levels=config.num_feature_levels,
        nheads=config.nheads,
        max_text_len=config.max_text_len,
        two_stage_type="standard",
        dec_pred_bbox_embed_share=False,
        two_stage_bbox_embed_share=False,
    )
    
    return model, text_encoder


def build_criterion(config: TrainingConfig):
    """Build loss criterion based on configuration"""
    return SetCriterion(
        num_classes=config.num_classes,
        weight_dict={"loss_ce": config.loss_coefs.get("loss_ce", 2.0), 
                     "loss_bbox": config.loss_coefs.get("loss_bbox", 5.0), 
                     "loss_giou": config.loss_coefs.get("loss_giou", 2.0)},
        losses=["labels", "boxes", "giou"],
        eos_coef=0.1
    )


def main(args: argparse.Namespace):
    """Main training function"""
    # Create configuration
    config = create_train_config(args)
    
    # Set random seed
    if config.seed is not None:
        seed_everything(config.seed)
    
    # Set CUDA
    if config.device == "cuda":
        jt.flags.use_cuda = 1
    
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
    train_dataset = build_dataset('train', config.to_dict())
    val_dataset = build_dataset('val', config.to_dict())
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler_type='lvis' if config.dataset_file == 'lvis' else None,
        sampler_kwargs={'samples_per_epoch': 1000} if config.dataset_file == 'lvis' else {}
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler_type=None,
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
