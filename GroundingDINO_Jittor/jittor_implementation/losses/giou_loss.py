# GIoU Loss (Member B)
import jittor as jt
from jittor import nn
import numpy as np


def box_cxcywh_to_xyxy(x: jt.Var) -> jt.Var:
    """
    Convert bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    
    Args:
        x: Bounding boxes of shape (N, 4) in (cx, cy, w, h) format
        
    Returns:
        Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
    """
    x_c, y_c, w, h = jt.unbind(x, dim=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: jt.Var) -> jt.Var:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height)
    
    Args:
        x: Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        
    Returns:
        Bounding boxes of shape (N, 4) in (cx, cy, w, h) format
    """
    x0, y0, x1, y1 = jt.unbind(x, dim=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)


def box_area(boxes: jt.Var) -> jt.Var:
    """
    Compute area of bounding boxes
    
    Args:
        boxes: Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        
    Returns:
        Area of each box of shape (N,)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: jt.Var, boxes2: jt.Var) -> jt.Var:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Bounding boxes of shape (M, 4) in (x1, y1, x2, y2) format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    # Compute intersection
    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    
    wh = (rb - lt).clamp(min_v=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)
    
    # Compute union
    union = area1[:, None] + area2 - inter
    
    # Compute IoU
    iou = inter / (union + 1e-7)
    
    return iou


def box_giou(boxes1: jt.Var, boxes2: jt.Var) -> jt.Var:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes
    
    Args:
        boxes1: Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        
    Returns:
        GIoU values of shape (N,)
    """
    # Compute IoU
    iou = box_iou(boxes1, boxes2).diag()
    
    # Compute convex hull (smallest enclosing box)
    lt = jt.minimum(boxes1[:, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min_v=0)
    area_c = wh[:, 0] * wh[:, 1]
    
    # Compute union area
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1 + area2 - iou * union
    
    # Compute GIoU
    giou = iou - (area_c - union) / (area_c + 1e-7)
    
    return giou


class GIoULoss(nn.Module):
    """
    Generalized IoU (GIoU) Loss for bounding box regression
    
    Args:
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def execute(self, pred_boxes: jt.Var, target_boxes: jt.Var) -> jt.Var:
        """
        Calculate GIoU loss
        
        Args:
            pred_boxes: Predicted boxes of shape (N, 4) in (cx, cy, w, h) format
            target_boxes: Target boxes of shape (N, 4) in (cx, cy, w, h) format
            
        Returns:
            Calculated GIoU loss
        """
        # Convert to (x1, y1, x2, y2) format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Calculate GIoU
        giou = box_giou(pred_boxes_xyxy, target_boxes_xyxy)
        
        # GIoU loss is 1 - GIoU
        loss = 1 - giou
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DIoULoss(nn.Module):
    """
    Distance IoU (DIoU) Loss for bounding box regression
    
    Args:
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def execute(self, pred_boxes: jt.Var, target_boxes: jt.Var) -> jt.Var:
        """
        Calculate DIoU loss
        
        Args:
            pred_boxes: Predicted boxes of shape (N, 4) in (cx, cy, w, h) format
            target_boxes: Target boxes of shape (N, 4) in (cx, cy, w, h) format
            
        Returns:
            Calculated DIoU loss
        """
        # Convert to (x1, y1, x2, y2) format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Calculate center points
        pred_cx = (pred_boxes_xyxy[:, 0] + pred_boxes_xyxy[:, 2]) / 2
        pred_cy = (pred_boxes_xyxy[:, 1] + pred_boxes_xyxy[:, 3]) / 2
        target_cx = (target_boxes_xyxy[:, 0] + target_boxes_xyxy[:, 2]) / 2
        target_cy = (target_boxes_xyxy[:, 1] + target_boxes_xyxy[:, 3]) / 2
        
        # Calculate diagonal of the smallest enclosing box
        c_x = (jt.maximum(pred_boxes_xyxy[:, 0], target_boxes_xyxy[:, 0]) - 
               jt.minimum(pred_boxes_xyxy[:, 2], target_boxes_xyxy[:, 2]))
        c_y = (jt.maximum(pred_boxes_xyxy[:, 1], target_boxes_xyxy[:, 1]) - 
               jt.minimum(pred_boxes_xyxy[:, 3], target_boxes_xyxy[:, 3]))
        c_squared = c_x ** 2 + c_y ** 2 + 1e-7
        
        # Calculate squared distance between center points
        rho_squared = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Calculate IoU
        iou = box_iou(pred_boxes_xyxy, target_boxes_xyxy).diag()
        
        # Calculate DIoU
        diou = iou - rho_squared / c_squared
        
        # DIoU loss is 1 - DIoU
        loss = 1 - diou
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class CIoULoss(nn.Module):
    """
    Complete IoU (CIoU) Loss for bounding box regression
    
    Args:
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def execute(self, pred_boxes: jt.Var, target_boxes: jt.Var) -> jt.Var:
        """
        Calculate CIoU loss
        
        Args:
            pred_boxes: Predicted boxes of shape (N, 4) in (cx, cy, w, h) format
            target_boxes: Target boxes of shape (N, 4) in (cx, cy, w, h) format
            
        Returns:
            Calculated CIoU loss
        """
        # Convert to (x1, y1, x2, y2) format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Calculate width and height
        pred_w = pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]
        pred_h = pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1]
        target_w = target_boxes_xyxy[:, 2] - target_boxes_xyxy[:, 0]
        target_h = target_boxes_xyxy[:, 3] - target_boxes_xyxy[:, 1]
        
        # Calculate aspect ratio consistency term
        v = (4 / (jt.pi ** 2)) * jt.pow(
            jt.atan(target_w / (target_h + 1e-7)) - jt.atan(pred_w / (pred_h + 1e-7)), 2
        )
        
        # Calculate alpha
        alpha = v / (1 - box_iou(pred_boxes_xyxy, target_boxes_xyxy).diag() + v + 1e-7)
        
        # Calculate center points
        pred_cx = (pred_boxes_xyxy[:, 0] + pred_boxes_xyxy[:, 2]) / 2
        pred_cy = (pred_boxes_xyxy[:, 1] + pred_boxes_xyxy[:, 3]) / 2
        target_cx = (target_boxes_xyxy[:, 0] + target_boxes_xyxy[:, 2]) / 2
        target_cy = (target_boxes_xyxy[:, 1] + target_boxes_xyxy[:, 3]) / 2
        
        # Calculate diagonal of the smallest enclosing box
        c_x = (jt.maximum(pred_boxes_xyxy[:, 0], target_boxes_xyxy[:, 0]) - 
               jt.minimum(pred_boxes_xyxy[:, 2], target_boxes_xyxy[:, 2]))
        c_y = (jt.maximum(pred_boxes_xyxy[:, 1], target_boxes_xyxy[:, 1]) - 
               jt.minimum(pred_boxes_xyxy[:, 3], target_boxes_xyxy[:, 3]))
        c_squared = c_x ** 2 + c_y ** 2 + 1e-7
        
        # Calculate squared distance between center points
        rho_squared = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Calculate IoU
        iou = box_iou(pred_boxes_xyxy, target_boxes_xyxy).diag()
        
        # Calculate CIoU
        ciou = iou - (rho_squared / c_squared + alpha * v)
        
        # CIoU loss is 1 - CIoU
        loss = 1 - ciou
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

