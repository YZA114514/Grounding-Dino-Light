# VLM Comparison Experiment (Member C)
import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional, Union, Tuple

import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import project modules
from ..train.config import create_train_config
from ..models import GroundingDINO
from ..models.text_encoder import BERTWrapper
from ..data import build_dataset, get_dataloader
from ..train.utils import seed_everything, convert_to_jittor_format


class VLMComparator:
    """Comparator for Vision-Language Models"""
    
    def __init__(self, model: GroundingDINO, text_encoder: BERTWrapper, 
                 config, output_dir: str = "./comparison_results"):
        """
        Initialize VLM Comparator
        
        Args:
            model: GroundingDINO model
            text_encoder: BERT text encoder
            config: Configuration
            output_dir: Output directory for results
        """
        self.model = model
        self.text_encoder = text_encoder
        self.config = config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        self.device = jt.device(config.device)
        self.model.to(self.device)
        self.text_encoder.to(self.device)
        
        # Set models to eval mode
        self.model.eval()
        self.text_encoder.eval()
        
        # Initialize results storage
        self.results = {
            "model_name": config.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images": []
        }
    
    @jt.no_grad
    def process_image(self, image_path: str, text: str, threshold: float = 0.3):
        """
        Process a single image with text prompt
        
        Args:
            image_path: Path to image
            text: Text prompt
            threshold: Confidence threshold for detections
            
        Returns:
            Dict containing detection results
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (width, height)
        
        # Preprocess image
        transform = T.Compose([
            T.Resize((800, 1333)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = jt.array(image_tensor.numpy()).to(self.device)
        
        # Process text
        text_dict = self.text_encoder([text])
        
        # Move text features to device
        for key, value in text_dict.items():
            if isinstance(value, jt.Var):
                text_dict[key] = value.to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor, text_dict)
        
        # Extract predictions
        pred_logits = outputs["pred_logits"][0]  # (N, num_classes)
        pred_boxes = outputs["pred_boxes"][0]  # (N, 4)
        
        # Apply sigmoid to get scores
        pred_scores = pred_logits.sigmoid()
        
        # Get max scores for each detection
        max_scores, _ = pred_scores.max(dim=-1)  # (N,)
        
        # Filter by threshold
        keep = max_scores > threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = max_scores[keep]
        pred_labels = pred_scores.argmax(dim=-1)
        
        # Convert from relative [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        w, h = orig_size
        boxes = pred_boxes.numpy()
        boxes[:, 0::2] *= w  # cx -> x1, x2
        boxes[:, 1::2] *= h  # cy -> y1, y2
        
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3]  # y2
        
        # Clip to image bounds
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        
        # Return results
        return {
            "image_path": image_path,
            "text": text,
            "boxes": boxes.tolist(),
            "scores": pred_scores.numpy().tolist(),
            "labels": pred_labels.numpy().tolist()
        }
    
    def visualize_results(self, image_path: str, result: Dict, save_path: str = None):
        """
        Visualize detection results on image
        
        Args:
            image_path: Path to original image
            result: Detection results
            save_path: Path to save visualization
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Convert to numpy for matplotlib
        image_np = np.array(image)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)
        
        # Draw boxes
        boxes = np.array(result["boxes"])
        scores = np.array(result["scores"])
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, color='red', linewidth=2
                )
            )
            
            # Add score
            plt.text(
                x1, y1 - 10,
                f"{score:.2f}",
                color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Add title
        plt.title(f"Text prompt: {result['text']}")
        plt.axis('off')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
    
    def run_comparison(self, image_list: List[str], text_prompts: List[str], 
                      save_visualizations: bool = True):
        """
        Run comparison on a list of images and text prompts
        
        Args:
            image_list: List of image paths
            text_prompts: List of text prompts
            save_visualizations: Whether to save visualizations
        """
        for image_path in image_list:
            for text in text_prompts:
                print(f"Processing {image_path} with prompt: '{text}'")
                
                # Process image
                result = self.process_image(image_path, text)
                
                # Add to results
                self.results["images"].append(result)
                
                # Visualize if needed
                if save_visualizations:
                    # Create save path
                    image_name = os.path.basename(image_path).split('.')[0]
                    text_name = text.replace(" ", "_").replace(".", "")
                    save_path = os.path.join(
                        self.output_dir, 
                        f"{image_name}_{text_name}.jpg"
                    )
                    
                    # Save visualization
                    self.visualize_results(image_path, result, save_path)
        
        # Save results
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        return self.results
    
    def compare_with_baseline(self, baseline_results_path: str, 
                             output_comparison_path: str = None):
        """
        Compare results with baseline
        
        Args:
            baseline_results_path: Path to baseline results JSON
            output_comparison_path: Path to save comparison results
        """
        # Load baseline results
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        
        # Create comparison
        comparison = {
            "model": self.results["model_name"],
            "baseline": baseline_results["model_name"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparisons": []
        }
        
        # Group results by image and text
        current_results_by_key = {}
        baseline_results_by_key = {}
        
        for item in self.results["images"]:
            key = (item["image_path"], item["text"])
            current_results_by_key[key] = item
        
        for item in baseline_results["images"]:
            key = (item["image_path"], item["text"])
            baseline_results_by_key[key] = item
        
        # Compare common items
        common_keys = set(current_results_by_key.keys()) & set(baseline_results_by_key.keys())
        
        for key in common_keys:
            current = current_results_by_key[key]
            baseline = baseline_results_by_key[key]
            
            # Count detections
            current_count = len(current["boxes"])
            baseline_count = len(baseline["boxes"])
            
            # Average confidence
            current_avg_conf = np.mean(current["scores"]) if current["scores"] > 0 else 0
            baseline_avg_conf = np.mean(baseline["scores"]) if baseline["scores"] > 0 else 0
            
            comparison_item = {
                "image_path": key[0],
                "text": key[1],
                "current_model": {
                    "num_detections": current_count,
                    "avg_confidence": float(current_avg_conf)
                },
                "baseline_model": {
                    "num_detections": baseline_count,
                    "avg_confidence": float(baseline_avg_conf)
                },
                "difference": {
                    "num_detections": current_count - baseline_count,
                    "avg_confidence": current_avg_conf - baseline_avg_conf
                }
            }
            
            comparison["comparisons"].append(comparison_item)
        
        # Save comparison
        if output_comparison_path is None:
            output_comparison_path = os.path.join(self.output_dir, "comparison.json")
        
        with open(output_comparison_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        return comparison


def load_model(config, checkpoint_path: str):
    """Load model from checkpoint"""
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
    
    # Load checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = jt.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Could not load checkpoint from {checkpoint_path}")
    
    # Set feature map for text encoder
    text_encoder.set_feat_map(model.text_feat_map)
    
    return model, text_encoder


def main(args):
    """Main function for VLM comparison"""
    # Create configuration
    config = create_train_config(args)
    
    # Set random seed
    if config.seed is not None:
        seed_everything(config.seed)
    
    # Load model
    model, text_encoder = load_model(config, args.checkpoint_path)
    
    # Initialize comparator
    comparator = VLMComparator(
        model=model,
        text_encoder=text_encoder,
        config=config,
        output_dir=args.output_dir
    )
    
    # Prepare image list and text prompts
    image_list = args.image_list
    text_prompts = args.text_prompts
    
    # Run comparison
    results = comparator.run_comparison(
        image_list=image_list,
        text_prompts=text_prompts,
        save_visualizations=args.save_visualizations
    )
    
    # Compare with baseline if provided
    if args.baseline_results:
        comparator.compare_with_baseline(args.baseline_results)
    
    print(f"Comparison results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VLM Comparison Experiment")
    
    # Model arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="groundingdino_swin-t",
                        help="Model name")
    
    # Data arguments
    parser.add_argument("--image_list", type=str, nargs='+', required=True,
                        help="List of image paths")
    parser.add_argument("--text_prompts", type=str, nargs='+', required=True,
                        help="List of text prompts")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                        help="Output directory for results")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualization images")
    
    # Comparison arguments
    parser.add_argument("--baseline_results", type=str, default=None,
                        help="Path to baseline results JSON")
    
    # Training config arguments (subset)
    parser.add_argument("--text_encoder_type", type=str, default="bert-base-uncased",
                        help="Text encoder type")
    parser.add_argument("--max_text_len", type=int, default=256,
                        help="Maximum text length")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)