# Focal Loss (Member B)
import jittor as jt
from jittor import nn
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var) -> jt.Var:
        """
        Calculate focal loss
        
        Args:
            pred: Predicted logits of shape (N, C) where C is the number of classes
            target: Ground truth labels of shape (N,) where each value is in [0, C-1]
            
        Returns:
            Calculated focal loss
        """
        # Convert target to one-hot encoding
        num_classes = pred.shape[-1]
        target_one_hot = jt.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate probability
        prob = nn.sigmoid(pred)
        
        # Calculate focal weight
        pt = prob * target_one_hot + (1 - prob) * (1 - target_one_hot)
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
        
        # Calculate binary cross entropy
        bce = -jt.log(pt + 1e-8)
        
        # Apply weights
        loss = alpha_weight * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid-based Focal Loss for multi-label classification
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var) -> jt.Var:
        """
        Calculate sigmoid-based focal loss
        
        Args:
            pred: Predicted logits of shape (N, C) where C is the number of classes
            target: Ground truth labels of shape (N, C) with values 0 or 1
            
        Returns:
            Calculated focal loss
        """
        # Calculate probability
        prob = nn.sigmoid(pred)
        
        # Calculate focal weight
        pt = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate binary cross entropy
        bce = -jt.log(pt + 1e-8)
        
        # Apply weights
        loss = alpha_weight * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

