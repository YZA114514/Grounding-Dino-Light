# LVIS Evaluator (Member B)
import os
import json
import numpy as np
import jittor as jt
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import tempfile
import subprocess


class LVISEvaluator:
    """
    LVIS Evaluator for evaluating object detection performance
    
    Args:
        ann_file: Path to LVIS annotation file
        output_dir: Directory to save evaluation results
        iou_types: List of IoU types to evaluate (default: ['bbox'])
    """
    
    def __init__(
        self,
        ann_file: str,
        output_dir: str = './eval_results',
        iou_types: List[str] = ['bbox']
    ):
        self.ann_file = ann_file
        self.output_dir = output_dir
        self.iou_types = iou_types
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco_gt = json.load(f)
        
        # Create category mapping
        self.categories = {cat['id']: cat for cat in self.coco_gt['categories']}
        self.cat_ids = [cat['id'] for cat in self.coco_gt['categories']]
        self.num_classes = len(self.cat_ids)
        
        # Create image mapping
        self.images = {img['id']: img for img in self.coco_gt['images']}
        
        # Group annotations by image
        self.img_to_anns = defaultdict(list)
        for ann in self.coco_gt['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
    
    def prepare_for_coco_detection(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert predictions to COCO format
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            List of predictions in COCO format
        """
        coco_results = []
        
        for pred in predictions:
            image_id = pred['image_id']
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            # Convert boxes from [cx, cy, w, h] to [x, y, w, h]
            if len(boxes) > 0:
                boxes = boxes.numpy()
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x = cx - w / 2
                y = cy - h / 2
                
                for i in range(len(boxes)):
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(labels[i]),
                        'bbox': [float(x[i]), float(y[i]), float(w[i]), float(h[i])],
                        'score': float(scores[i])
                    })
        
        return coco_results
    
    def evaluate(self, model, dataloader) -> Dict[str, float]:
        """
        Evaluate model on LVIS dataset
        
        Args:
            model: Model to evaluate
            dataloader: Dataloader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Collect predictions
        predictions = []
        
        with jt.no_grad():
            for images, targets in dataloader:
                # Get model predictions
                outputs = model(images)
                
                # Process each image in the batch
                for i in range(len(targets)):
                    image_id = targets[i]['image_id']
                    
                    # Extract predictions
                    pred_logits = outputs['pred_logits'][i]
                    pred_boxes = outputs['pred_boxes'][i]
                    
                    # Apply sigmoid to get probabilities
                    pred_probs = jt.sigmoid(pred_logits)
                    
                    # Get top-k predictions
                    scores, labels = jt.max(pred_probs, dim=-1)
                    
                    # Filter by confidence threshold
                    conf_threshold = 0.1
                    mask = scores > conf_threshold
                    
                    if jt.sum(mask) > 0:
                        filtered_scores = scores[mask]
                        filtered_labels = labels[mask]
                        filtered_boxes = pred_boxes[mask]
                        
                        predictions.append({
                            'image_id': int(image_id),
                            'boxes': filtered_boxes,
                            'scores': filtered_scores,
                            'labels': filtered_labels
                        })
                    else:
                        # No predictions for this image
                        predictions.append({
                            'image_id': int(image_id),
                            'boxes': jt.array([]).reshape(0, 4),
                            'scores': jt.array([]),
                            'labels': jt.array([])
                        })
        
        # Convert to COCO format
        coco_results = self.prepare_for_coco_detection(predictions)
        
        # Save predictions to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_results, f)
            results_file = f.name
        
        # Run COCO evaluation
        metrics = self._run_coco_evaluation(results_file)
        
        # Clean up temporary file
        os.remove(results_file)
        
        return metrics
    
    def _run_coco_evaluation(self, results_file: str) -> Dict[str, float]:
        """
        Run COCO evaluation using pycocotools
        
        Args:
            results_file: Path to results file
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Try to use pycocotools if available
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            
            # Initialize COCO ground truth
            coco_gt = COCO(self.ann_file)
            
            # Initialize COCO predictions
            coco_dt = coco_gt.loadRes(results_file)
            
            # Initialize COCOeval
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            stats = coco_eval.stats
            
            # Map to LVIS metrics
            metrics = {
                'AP': stats[0],  # AP at IoU=0.50:0.95
                'AP50': stats[1],  # AP at IoU=0.50
                'AP75': stats[2],  # AP at IoU=0.75
                'APs': stats[3],  # AP for small area
                'APm': stats[4],  # AP for medium area
                'APl': stats[5],  # AP for large area
                'AR1': stats[6],  # AR with 1 detection per image
                'AR10': stats[7],  # AR with 10 detections per image
                'AR100': stats[8],  # AR with 100 detections per image
                'ARs': stats[9],  # AR for small area
                'ARm': stats[10],  # AR for medium area
                'ARl': stats[11],  # AR for large area
            }
            
            return metrics
            
        except ImportError:
            # Fallback to simple evaluation if pycocotools is not available
            print("pycocotools not available, using simple evaluation")
            return self._simple_evaluation(results_file)
    
    def _simple_evaluation(self, results_file: str) -> Dict[str, float]:
        """
        Simple evaluation without pycocotools
        
        Args:
            results_file: Path to results file
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load predictions
        with open(results_file, 'r') as f:
            predictions = json.load(f)
        
        # Group predictions by image
        img_to_preds = defaultdict(list)
        for pred in predictions:
            img_to_preds[pred['image_id']].append(pred)
        
        # Calculate simple metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_id, anns in self.img_to_anns.items():
            # Get ground truth annotations
            gt_boxes = [ann['bbox'] for ann in anns]
            gt_cats = [ann['category_id'] for ann in anns]
            
            # Get predictions
            preds = img_to_preds.get(img_id, [])
            pred_boxes = [pred['bbox'] for pred in preds]
            pred_cats = [pred['category_id'] for pred in preds]
            pred_scores = [pred['score'] for pred in preds]
            
            # Sort predictions by score
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes = [pred_boxes[i] for i in sorted_indices]
            pred_cats = [pred_cats[i] for i in sorted_indices]
            
            # Calculate IoU and matches
            matches = self._calculate_matches(gt_boxes, gt_cats, pred_boxes, pred_cats)
            
            tp = len(matches)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate precision and recall
        precision = total_tp / (total_tp + total_fp + 1e-7)
        recall = total_tp / (total_tp + total_fn + 1e-7)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'AP': f1,  # Use F1 as a simple approximation of AP
            'precision': precision,
            'recall': recall
        }
    
    def _calculate_matches(
        self,
        gt_boxes: List[List[float]],
        gt_cats: List[int],
        pred_boxes: List[List[float]],
        pred_cats: List[int],
        iou_threshold: float = 0.5
    ) -> List[int]:
        """
        Calculate matches between ground truth and predictions
        
        Args:
            gt_boxes: List of ground truth boxes [x, y, w, h]
            gt_cats: List of ground truth category IDs
            pred_boxes: List of predicted boxes [x, y, w, h]
            pred_cats: List of predicted category IDs
            iou_threshold: IoU threshold for matching
            
        Returns:
            List of matched prediction indices
        """
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return []
        
        # Convert to numpy arrays
        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        
        # Convert to [x1, y1, x2, y2] format
        gt_boxes_xyxy = np.column_stack([
            gt_boxes[:, 0],
            gt_boxes[:, 1],
            gt_boxes[:, 0] + gt_boxes[:, 2],
            gt_boxes[:, 1] + gt_boxes[:, 3]
        ])
        
        pred_boxes_xyxy = np.column_stack([
            pred_boxes[:, 0],
            pred_boxes[:, 1],
            pred_boxes[:, 0] + pred_boxes[:, 2],
            pred_boxes[:, 1] + pred_boxes[:, 3]
        ])
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(gt_boxes_xyxy, pred_boxes_xyxy)
        
        # Find matches
        matches = []
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)
        
        # Greedy matching
        for i in range(len(pred_boxes)):
            # Find best match for this prediction
            best_iou = 0
            best_gt_idx = -1
            
            for j in range(len(gt_boxes)):
                if not gt_matched[j] and iou_matrix[j, i] > best_iou:
                    best_iou = iou_matrix[j, i]
                    best_gt_idx = j
            
            # Check if match is valid
            if best_gt_idx >= 0 and best_iou >= iou_threshold and gt_cats[best_gt_idx] == pred_cats[i]:
                matches.append(i)
                gt_matched[best_gt_idx] = True
                pred_matched[i] = True
        
        return matches
    
    def _calculate_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate IoU matrix between two sets of boxes
        
        Args:
            boxes1: Array of boxes in [x1, y1, x2, y2] format
            boxes2: Array of boxes in [x1, y1, x2, y2] format
            
        Returns:
            IoU matrix of shape (len(boxes1), len(boxes2))
        """
        # Calculate intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        return iou


def evaluate_lvis(
    model,
    dataloader,
    ann_file: str,
    output_dir: str = './eval_results'
) -> Dict[str, float]:
    """
    Evaluate model on LVIS dataset
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
        ann_file: Path to LVIS annotation file
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = LVISEvaluator(ann_file, output_dir)
    metrics = evaluator.evaluate(model, dataloader)
    
    # Print results
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics

