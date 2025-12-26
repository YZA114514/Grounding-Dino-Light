# Training Config (Member C)
import argparse
import os
from typing import Dict, List, Optional, Union


class TrainingConfig:
    """Configuration for training GroundingDINO"""
    
    def __init__(self, args=None):
        # Initialize with default values
        self.setup_default()
        
        # Update with arguments if provided
        if args is not None:
            self.update_from_args(args)
    
    def setup_default(self):
        """Setup default configuration values"""
        # Model parameters
        self.model_name = "groundingdino_swin-t"
        self.hidden_dim = 256
        self.num_queries = 900
        self.num_feature_levels = 4
        self.nheads = 8
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.activation = "relu"
        
        # Text encoder parameters
        self.text_encoder_type = "bert-base-uncased"
        self.max_text_len = 256
        
        # Training parameters
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.batch_size = 4
        self.epochs = 40
        self.lr_drop = 35
        self.lr_backbone_names = ["backbone.0", "backbone.1", "backbone.2", "backbone.3"]
        self.lr_backbone = 1e-5
        self.lr_linear_proj_names = ["reference_points", "sampling_offsets"]
        self.lr_linear_proj_mult = 0.1
        self.lr_criterion = 1e-4
        self.dropout = 0.1
        self.clip_max_norm = 0.1
        
        # Data parameters
        self.dataset_file = "lvis"
        self.data_path = "/path/to/dataset"
        self.remove_difficult = False
        self.frozen_weights = None
        
        # Loss coefficients
        self.loss_coefs = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0
        }
        self.aux_loss = True
        self.set_cost_class = 2.0
        self.set_cost_bbox = 5.0
        self.set_cost_giou = 2.0
        
        # Matcher parameters
        self.matcher = {
            "cost_class": 1,
            "cost_bbox": 5,
            "cost_giou": 2,
            "focal_alpha": 0.25
        }
        
        # Evaluation parameters
        self.eval_interval = 5
        self.eval = True
        
        # Output and checkpoint parameters
        self.output_dir = "./outputs"
        self.checkpoint_dir = "./checkpoints"
        self.resume = ""
        self.start_epoch = 0
        self.save_interval = 5
        self.log_interval = 50
        
        # Device parameters
        self.device = "cuda"
        self.seed = 42
        self.num_workers = 4
        
        # Distributed training parameters
        self.world_size = 1
        self.dist_url = "env://"
        self.rank = 0
        
        # Other parameters
        self.use_wandb = False
        self.wandb_project = "groundingdino"
        self.wandb_entity = None
        
        # Debug parameters
        self.debug = False
        self.log_image_interval = 100
        
        # FPN parameters
        self.position_encoding = "sine"
        self.position_embedding_scale = 6.283185307179586  # 2 * pi
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.num_feature_levels = 4
        self.dec_n_points = 4
        self.enc_n_points = 4
        self.two_stage = False
        self.two_stage_num_proposals = 300
        
        # Backbone parameters
        self.backbone = "resnet50"
        self.dilation = False
        self.position_embedding = "sine"
        
        # Transformer parameters
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 100
        self.pre_norm = False
        
        # Loss parameters
        self.no_aux_loss = False
        self.bbox_loss_type = "l1"
        self.giou_loss_type = "linear"
        self.focal_alpha = 0.25
        
        # Dataset parameters
        self.num_classes = 2001  # LVIS
        self.masks = False
        self.aux_loss = True
        
        # Text parameters
        self.text_encoder_type = "bert-base-uncased"
        self.sub_sentence_present = True
        
        # Optimization parameters
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 2
        self.weight_decay = 1e-4
        self.epochs = 50
        self.lr_drop = 40
        self.clip_max_norm = 0.1
        
        # Scheduler parameters
        self.lr_scheduler = "step"
        self.min_lr = 1e-6
        self.warmup_steps = 1000
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key in dir(self):
            if not key.startswith('_'):
                value = getattr(self, key)
                if not callable(value):
                    config_dict[key] = value
        return config_dict
    
    def from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def load(self, path: str):
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        self.from_dict(config_dict)


def get_args_parser():
    """Get argument parser for training configuration"""
    parser = argparse.ArgumentParser('GroundingDINO training script', add_help=False)
    
    # Model parameters
    parser.add_argument('--model_name', default='groundingdino_swin-t', type=str,
                        help='Name of model to use')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='Hidden dimension of transformer')
    parser.add_argument('--num_queries', default=900, type=int,
                        help='Number of query slots')
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help='Number of feature levels')
    parser.add_argument('--nheads', default=8, type=int,
                        help='Number of attention heads')
    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help='Dimension of feedforward network')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--activation', default='relu', type=str,
                        help='Activation function')
    
    # Text encoder parameters
    parser.add_argument('--text_encoder_type', default='bert-base-uncased', type=str,
                        help='Type of text encoder')
    parser.add_argument('--max_text_len', default=256, type=int,
                        help='Maximum text length')
    
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    parser.add_argument('--lr_backbone_names', default=["backbone.0", "backbone.1", "backbone.2", "backbone.3"], nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=["reference_points", "sampling_offsets"], nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_criterion', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    
    # Data parameters
    parser.add_argument('--dataset_file', default='lvis', type=str)
    parser.add_argument('--data_path', default='/path/to/dataset', type=str,
                        help='Path to dataset')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # Loss coefficients
    parser.add_argument('--loss_ce', default=2.0, type=float)
    parser.add_argument('--loss_bbox', default=5.0, type=float)
    parser.add_argument('--loss_giou', default=2.0, type=float)
    parser.add_argument('--aux_loss', action='store_true')
    parser.add_argument('--set_cost_class', default=2.0, type=float)
    parser.add_argument('--set_cost_bbox', default=5.0, type=float)
    parser.add_argument('--set_cost_giou', default=2.0, type=float)
    
    # Matcher parameters
    parser.add_argument('--matcher_cost_class', default=1.0, type=float)
    parser.add_argument('--matcher_cost_bbox', default=5.0, type=float)
    parser.add_argument('--matcher_cost_giou', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Evaluation parameters
    parser.add_argument('--eval_interval', default=5, type=int)
    parser.add_argument('--eval', action='store_true')
    
    # Output and checkpoint parameters
    parser.add_argument('--output_dir', default='./outputs', type=str,
                        help='Path to save outputs')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch')
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    
    # Device parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='URL used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='Rank of current process')
    
    # Other parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', default='groundingdino', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='Weights & Biases entity name')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--log_image_interval', default=100, type=int,
                        help='Interval for logging images')
    
    return parser


def create_train_config(args=None):
    """Create training configuration"""
    if args is None:
        # Create default arguments
        parser = get_args_parser()
        args = parser.parse_args([])
    
    config = TrainingConfig(args)
    
    # Create directories if they don't exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    return config


# Predefined configurations for different models and settings
def get_swin_t_config():
    """Get configuration for Swin-T backbone"""
    config = TrainingConfig()
    config.model_name = "groundingdino_swin-t"
    config.backbone = "swin_T_224_1k"
    config.hidden_dim = 256
    config.num_queries = 900
    config.lr = 1e-4
    config.lr_backbone = 1e-5
    config.epochs = 40
    return config


def get_swin_b_config():
    """Get configuration for Swin-B backbone"""
    config = TrainingConfig()
    config.model_name = "groundingdino_swin-b"
    config.backbone = "swin_B_224_22k"
    config.hidden_dim = 256
    config.num_queries = 900
    config.lr = 1e-4
    config.lr_backbone = 5e-6
    config.epochs = 50
    return config


def get_debug_config():
    """Get configuration for debugging"""
    config = TrainingConfig()
    config.batch_size = 1
    config.epochs = 5
    config.eval_interval = 1
    config.log_interval = 10
    config.save_interval = 1
    config.debug = True
    return config