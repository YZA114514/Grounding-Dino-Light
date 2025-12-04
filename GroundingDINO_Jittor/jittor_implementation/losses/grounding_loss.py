# Grounding Loss (Member B)
import jittor as jt
from jittor import nn
import numpy as np
from .focal_loss import SigmoidFocalLoss
from .giou_loss import GIoULoss
from .l1_loss import L1Loss


class GroundingLoss(nn.Module):
    """
    Grounding DINO Loss function that combines classification and bounding box regression losses
    
    Args:
        weight_dict (dict): Dictionary of loss weights
        losses (list): List of losses to compute
        eos_coef (float): Weight for no-object class (default: 0.1)
    """
    
    def __init__(
        self,
        weight_dict: dict = None,
        losses: list = None,
        eos_coef: float = 0.1
    ):
        super().__init__()
        
        # Default weights
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 2.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0
            }
        
        self.weight_dict = weight_dict
        
        # Default losses
        if losses is None:
            losses = ['labels', 'boxes', 'giou']
        
        self.losses = losses
        self.eos_coef = eos_coef
        
        # Initialize loss functions
        self.focal_loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
        self.giou_loss = GIoULoss()
        self.l1_loss = L1Loss()
    
    def loss_labels(self, pred_logits: jt.Var, targets: list, indices: list) -> dict:
        """
        Compute classification loss
        
        Args:
            pred_logits: Predicted logits of shape (batch_size, num_queries, num_classes)
            targets: List of target dictionaries
            indices: List of tuples of matched indices
            
        Returns:
            Dictionary with classification loss
        """
        # Prepare target classes
        target_classes = jt.full(
            (pred_logits.shape[0], pred_logits.shape[1]),
            self.eos_coef,
            dtype=pred_logits.dtype
        )
        
        # Set positive samples
        for i, (_, target_classes_i) in enumerate(zip(indices, targets)):
            target_classes[i, indices[i][0]] = jt.array(target_classes_i['labels'])
        
        # Compute focal loss
        loss_ce = self.focal_loss(pred_logits, target_classes)
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(self, pred_boxes: jt.Var, targets: list, indices: list) -> dict:
        """
        Compute L1 bounding box loss
        
        Args:
            pred_boxes: Predicted boxes of shape (batch_size, num_queries, 4)
            targets: List of target dictionaries
            indices: List of tuples of matched indices
            
        Returns:
            Dictionary with L1 box loss
        """
        if 'boxes' not in self.losses:
            return {}
        
        # Get predicted boxes for matched indices
        src_boxes = []
        target_boxes = []
        
        for i, (idx_i, _) in enumerate(indices):
            src_boxes.append(pred_boxes[i, idx_i])
            target_boxes.append(jt.array(targets[i]['boxes'][idx_i]))
        
        if len(src_boxes) == 0:
            return {'loss_bbox': jt.array(0.0)}
        
        src_boxes = jt.concat(src_boxes, dim=0)
        target_boxes = jt.concat(target_boxes, dim=0)
        
        # Compute L1 loss
        loss_bbox = self.l1_loss(src_boxes, target_boxes)
        
        return {'loss_bbox': loss_bbox}
    
    def loss_giou(self, pred_boxes: jt.Var, targets: list, indices: list) -> dict:
        """
        Compute GIoU bounding box loss
        
        Args:
            pred_boxes: Predicted boxes of shape (batch_size, num_queries, 4)
            targets: List of target dictionaries
            indices: List of tuples of matched indices
            
        Returns:
            Dictionary with GIoU loss
        """
        if 'giou' not in self.losses:
            return {}
        
        # Get predicted boxes for matched indices
        src_boxes = []
        target_boxes = []
        
        for i, (idx_i, _) in enumerate(indices):
            src_boxes.append(pred_boxes[i, idx_i])
            target_boxes.append(jt.array(targets[i]['boxes'][idx_i]))
        
        if len(src_boxes) == 0:
            return {'loss_giou': jt.array(0.0)}
        
        src_boxes = jt.concat(src_boxes, dim=0)
        target_boxes = jt.concat(target_boxes, dim=0)
        
        # Compute GIoU loss
        loss_giou = self.giou_loss(src_boxes, target_boxes)
        
        return {'loss_giou': loss_giou}
    
    def get_loss(self, loss_name: str, outputs: dict, targets: list, indices: list, **kwargs) -> dict:
        """
        Get specific loss by name
        
        Args:
            loss_name: Name of the loss
            outputs: Dictionary of model outputs
            targets: List of target dictionaries
            indices: List of tuples of matched indices
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with the specified loss
        """
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'giou': self.loss_giou
        }
        
        if loss_name not in loss_map:
            raise ValueError(f"Unknown loss name: {loss_name}")
        
        return loss_map[loss_name](outputs, targets, indices, **kwargs)
    
    def execute(self, outputs: dict, targets: list) -> dict:
        """
        Compute all losses
        
        Args:
            outputs: Dictionary of model outputs
            targets: List of target dictionaries
            
        Returns:
            Dictionary with all computed losses
        """
        # Extract outputs
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Compute matching between predictions and targets
        indices = self._compute_matching(pred_logits, pred_boxes, targets)
        
        # Compute all requested losses
        losses = {}
        for loss_name in self.losses:
            losses.update(self.get_loss(loss_name, outputs, targets, indices))
        
        # Apply weights
        weighted_losses = {}
        for k in losses.keys():
            if k in self.weight_dict:
                weighted_losses[k] = losses[k] * self.weight_dict[k]
            else:
                weighted_losses[k] = losses[k]
        
        return weighted_losses
    
    def _compute_matching(self, pred_logits: jt.Var, pred_boxes: jt.Var, targets: list) -> list:
        """
        Compute matching between predictions and targets using Hungarian algorithm
        
        Args:
            pred_logits: Predicted logits of shape (batch_size, num_queries, num_classes)
            pred_boxes: Predicted boxes of shape (batch_size, num_queries, 4)
            targets: List of target dictionaries
            
        Returns:
            List of tuples of matched indices
        """
        # This is a simplified version of the matching algorithm
        # In a full implementation, this would use the Hungarian algorithm
        # to find the optimal matching between predictions and targets
        
        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]
        
        indices = []
        for i in range(batch_size):
            # For simplicity, we'll just use the first N predictions
            # where N is the number of target objects
            num_targets = len(targets[i]['labels'])
            
            # Create dummy indices (pred_idx, target_idx)
            pred_idx = jt.arange(min(num_queries, num_targets))
            target_idx = jt.arange(num_targets)
            
            indices.append((pred_idx, target_idx))
        
        return indices


class SetCriterion(nn.Module):
    """
    Set criterion for DETR-style models
    
    Args:
        num_classes (int): Number of object classes
        matcher (nn.Module): Module for matching predictions and targets
        weight_dict (dict): Dictionary of loss weights
        eos_coef (float): Weight for no-object class
        losses (list): List of losses to compute
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module = None,
        weight_dict: dict = None,
        eos_coef: float = 0.1,
        losses: list = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Initialize grounding loss
        self.grounding_loss = GroundingLoss(
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef
        )
    
    def execute(self, outputs: dict, targets: list) -> dict:
        """
        Compute losses
        
        Args:
            outputs: Dictionary of model outputs
            targets: List of target dictionaries
            
        Returns:
            Dictionary with all computed losses
        """
        # Compute losses using grounding loss
        losses = self.grounding_loss(outputs, targets)
        
        return losses

