# L1 Loss (Member B)
import jittor as jt
from jittor import nn
import numpy as np


class L1Loss(nn.Module):
    """
    L1 Loss (Mean Absolute Error) for bounding box regression
    
    Args:
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var) -> jt.Var:
        """
        Calculate L1 loss
        
        Args:
            pred: Predicted values of shape (N, ...)
            target: Target values of shape (N, ...)
            
        Returns:
            Calculated L1 loss
        """
        loss = jt.abs(pred - target)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss) for bounding box regression
    
    Args:
        beta (float): Threshold at which to change from L1 to L2 loss (default: 1.0)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var) -> jt.Var:
        """
        Calculate Smooth L1 loss
        
        Args:
            pred: Predicted values of shape (N, ...)
            target: Target values of shape (N, ...)
            
        Returns:
            Calculated Smooth L1 loss
        """
        diff = jt.abs(pred - target)
        cond = diff < self.beta
        
        # L2 loss for small differences, L1 loss for large differences
        loss = jt.where(
            cond,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for bounding box regression
    
    Args:
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var, weights: jt.Var) -> jt.Var:
        """
        Calculate weighted L1 loss
        
        Args:
            pred: Predicted values of shape (N, ...)
            target: Target values of shape (N, ...)
            weights: Weights for each element of shape (N, ...)
            
        Returns:
            Calculated weighted L1 loss
        """
        loss = jt.abs(pred - target) * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum() / (weights.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted Smooth L1 Loss for bounding box regression
    
    Args:
        beta (float): Threshold at which to change from L1 to L2 loss (default: 1.0)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def execute(self, pred: jt.Var, target: jt.Var, weights: jt.Var) -> jt.Var:
        """
        Calculate weighted Smooth L1 loss
        
        Args:
            pred: Predicted values of shape (N, ...)
            target: Target values of shape (N, ...)
            weights: Weights for each element of shape (N, ...)
            
        Returns:
            Calculated weighted Smooth L1 loss
        """
        diff = jt.abs(pred - target)
        cond = diff < self.beta
        
        # L2 loss for small differences, L1 loss for large differences
        loss = jt.where(
            cond,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply weights
        loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum() / (weights.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

