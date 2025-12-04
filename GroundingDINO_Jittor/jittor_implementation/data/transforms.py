# Data Transforms (Member B)
import random
import PIL
import numpy as np
import jittor as jt
from jittor import transform
from typing import Tuple, List, Optional, Union, Dict, Any


def crop(image: PIL.Image.Image, target: Dict[str, Any], region: Tuple[int, int, int, int]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    """
    Crop the image and adjust target accordingly
    
    Args:
        image: PIL Image
        target: Dictionary containing annotations
        region: (i, j, h, w) region to crop
        
    Returns:
        Cropped image and adjusted target
    """
    cropped_image = image.crop((region[1], region[0], region[1] + region[3], region[0] + region[2]))
    
    target = target.copy()
    i, j, h, w = region
    
    # Update size
    target["size"] = jt.Var([h, w])
    
    fields = ["labels", "area", "iscrowd", "positive_map"]
    
    if "boxes" in target:
        boxes = jt.Var(target["boxes"])
        max_size = jt.Var([w, h], dtype=jt.float)
        cropped_boxes = boxes - jt.Var([j, i, j, i])
        cropped_boxes = jt.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = jt.clamp(cropped_boxes, min_v=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4).numpy().tolist()
        target["area"] = area.numpy().tolist()
        fields.append("boxes")
    
    if "masks" in target:
        # Update masks
        masks = jt.Var(target["masks"])
        target["masks"] = masks[:, i:i+h, j:j+w].numpy().tolist()
        fields.append("masks")
    
    # Remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # Favor boxes selection when defining which elements to keep
        if "boxes" in target:
            cropped_boxes = jt.Var(target["boxes"]).reshape(-1, 2, 2)
            keep = jt.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            masks = jt.Var(target["masks"])
            keep = masks.flatten(1).any(dim=1)
        
        keep_indices = jt.where(keep)[0].numpy().tolist()
        
        for field in fields:
            if field in target:
                target[field] = [target[field][i] for i in keep_indices]
    
    return cropped_image, target


def hflip(image: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    """
    Horizontally flip the image and adjust target accordingly
    
    Args:
        image: PIL Image
        target: Dictionary containing annotations
        
    Returns:
        Flipped image and adjusted target
    """
    flipped_image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    
    w, h = image.size
    
    target = target.copy()
    if "boxes" in target:
        boxes = jt.Var(target["boxes"])
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x2, y2 = x1 + w, y1 + h
        
        # Flip horizontally: new_x1 = w - old_x2, new_x2 = w - old_x1
        new_x1 = w - x2
        new_x2 = w - x1
        
        # Convert back to [x, y, w, h]
        new_boxes = jt.stack([new_x1, y1, new_x2 - new_x1, h], dim=1)
        target["boxes"] = new_boxes.numpy().tolist()
    
    if "masks" in target:
        masks = jt.Var(target["masks"])
        target["masks"] = jt.flip(masks, dims=[2]).numpy().tolist()
    
    return flipped_image, target


def resize(image: PIL.Image.Image, target: Optional[Dict[str, Any]], size: Union[int, Tuple[int, int]], 
           max_size: Optional[int] = None) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
    """
    Resize the image and adjust target accordingly
    
    Args:
        image: PIL Image
        target: Dictionary containing annotations
        size: Target size (int for shorter side, or (h, w) tuple)
        max_size: Maximum size for longer side
        
    Returns:
        Resized image and adjusted target
    """
    def get_size_with_aspect_ratio(image_size: Tuple[int, int], size: int, max_size: Optional[int] = None) -> Tuple[int, int]:
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        
        return (oh, ow)
    
    def get_size(image_size: Tuple[int, int], size: Union[int, Tuple[int, int]], 
                max_size: Optional[int] = None) -> Tuple[int, int]:
        if isinstance(size, (list, tuple)):
            return size
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    size = get_size(image.size, size, max_size)
    rescaled_image = image.resize((size[1], size[0]), PIL.Image.BILINEAR)
    
    if target is None:
        return rescaled_image, None
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    
    target = target.copy()
    if "boxes" in target:
        boxes = jt.Var(target["boxes"])
        scaled_boxes = boxes * jt.Var([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes.numpy().tolist()
    
    if "area" in target:
        area = jt.Var(target["area"])
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area.numpy().tolist()
    
    h, w = size
    target["size"] = jt.Var([h, w]).numpy().tolist()
    
    if "masks" in target:
        masks = jt.Var(target["masks"])
        # Use bilinear interpolation for masks
        resized_masks = jt.interpolate(masks.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=False)
        target["masks"] = (resized_masks > 0.5).squeeze(1).numpy().tolist()
    
    return rescaled_image, target


def pad(image: PIL.Image.Image, target: Optional[Dict[str, Any]], 
        padding: Tuple[int, int]) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
    """
    Pad the image and adjust target accordingly
    
    Args:
        image: PIL Image
        target: Dictionary containing annotations
        padding: (pad_x, pad_y) padding for right and bottom
        
    Returns:
        Padded image and adjusted target
    """
    padded_image = transform.pad(image, (padding[0], padding[1]))
    if target is None:
        return padded_image, None
    
    target = target.copy()
    # Update size
    w, h = padded_image.size
    target["size"] = jt.Var([h, w]).numpy().tolist()
    
    if "masks" in target:
        masks = jt.Var(target["masks"])
        padded_masks = jt.nn.functional.pad(masks, (0, padding[0], 0, padding[1]))
        target["masks"] = padded_masks.numpy().tolist()
    
    return padded_image, target


class RandomCrop:
    """Random crop transformation"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
    
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        # Get random crop parameters
        i, j, h, w = transform.RandomCrop.get_params(img, self.size)
        return crop(img, target, (i, j, h, w))


class RandomSizeCrop:
    """Random crop with variable size"""
    
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes
    
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        init_boxes = len(target.get("boxes", []))
        max_patience = 10
        
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = transform.RandomCrop.get_params(img, [h, w])
            result_img, result_target = crop(img, target, region)
            
            if (
                not self.respect_boxes
                or len(result_target.get("boxes", [])) == init_boxes
                or i == max_patience - 1
            ):
                return result_img, result_target
        
        return result_img, result_target


class CenterCrop:
    """Center crop transformation"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
    
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    """Random horizontal flip transformation"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    """Random resize transformation"""
    
    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, img: PIL.Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class ToTensor:
    """Convert PIL Image to tensor"""
    
    def __call__(self, img: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[jt.Var, Dict[str, Any]]:
        img_tensor = transform.to_tensor(img)
        return img_tensor, target


class Normalize:
    """Normalize tensor with mean and std"""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: jt.Var, target: Optional[Dict[str, Any]] = None) -> Tuple[jt.Var, Optional[Dict[str, Any]]]:
        image = transform.image_normalize(image, mean=self.mean, std=self.std)
        
        if target is None:
            return image, None
        
        target = target.copy()
        h, w = image.shape[-2:]
        
        if "boxes" in target:
            boxes = jt.Var(target["boxes"])
            # Convert from [x, y, w, h] to [cx, cy, w, h]
            x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx, cy = x + w/2, y + h/2
            boxes = jt.stack([cx, cy, w, h], dim=1)
            # Normalize by image size
            boxes = boxes / jt.Var([w, h, w, h], dtype=jt.float)
            target["boxes"] = boxes.numpy().tolist()
        
        return image, target


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: PIL.Image.Image, target: Dict[str, Any]) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def build_transforms(is_train: bool = True) -> Compose:
    """
    Build data transforms for training or evaluation
    
    Args:
        is_train: Whether to build training transforms (with augmentation)
        
    Returns:
        Composed transforms
    """
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if is_train:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        
        return Compose([
            RandomHorizontalFlip(),
            RandomResize(scales, max_size=1333),
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize((800, 1333)),
            ToTensor(),
            normalize,
        ])


class Resize:
    """Resize transformation"""
    
    def __init__(self, size: Union[int, Tuple[int, int]], max_size: Optional[int] = None):
        self.size = size
        self.max_size = max_size
    
    def __call__(self, img: PIL.Image.Image, target: Optional[Dict[str, Any]] = None) -> Tuple[PIL.Image.Image, Optional[Dict[str, Any]]]:
        return resize(img, target, self.size, self.max_size)

