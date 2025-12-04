# Dataset Loader (Member B)
import os
import json
import random
import numpy as np
import PIL
from typing import Dict, List, Any, Optional, Tuple, Callable
import jittor as jt
from jittor.dataset import Dataset
from .transforms import Compose, build_transforms


class LVISDataset(Dataset):
    """
    LVIS Dataset for Jittor
    
    Args:
        anno_path: Path to LVIS annotation file
        image_dir: Directory containing images
        transforms: Data transforms to apply
        is_train: Whether this is a training dataset
        filter_empty: Whether to filter out images with no annotations
    """
    
    def __init__(
        self,
        anno_path: str,
        image_dir: str,
        transforms: Optional[Compose] = None,
        is_train: bool = True,
        filter_empty: bool = True
    ):
        super().__init__()
        
        self.anno_path = anno_path
        self.image_dir = image_dir
        self.is_train = is_train
        self.filter_empty = filter_empty
        
        # Load annotations
        with open(anno_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter out images with no annotations if required
        if self.filter_empty:
            self.data = [item for item in self.data if len(item.get('bboxes', [])) > 0]
        
        # Set transforms
        self.transforms = transforms if transforms else build_transforms(is_train)
        
        # Set dataset attributes for Jittor
        self.total_len = len(self.data)
        
        # Print dataset info
        print(f"Loaded {self.total_len} images from {anno_path}")
    
    def __getitem__(self, idx: int) -> Tuple[jt.Var, Dict[str, Any]]:
        """
        Get item by index
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, target_dict)
        """
        # Get annotation data
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['file_name'])
        image = PIL.Image.open(image_path).convert('RGB')
        
        # Prepare target
        target = {
            'image_id': item.get('image_id', idx),
            'file_name': item['file_name'],
            'height': item.get('height', image.height),
            'width': item.get('width', image.width),
            'boxes': item.get('bboxes', []),
            'labels': item.get('labels', []),
            'categories': item.get('categories', []),
            'text': item.get('text', ''),
        }
        
        # Convert boxes to numpy array if needed
        if target['boxes']:
            target['boxes'] = np.array(target['boxes'], dtype=np.float32)
        
        # Apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
        
        # Convert target to proper format
        if isinstance(target['boxes'], list):
            target['boxes'] = np.array(target['boxes'], dtype=np.float32) if target['boxes'] else np.zeros((0, 4), dtype=np.float32)
        
        # Convert to Jittor tensors
        if not isinstance(image, jt.Var):
            image = jt.array(np.array(image))
        
        # Convert target fields to Jittor tensors
        for key in ['boxes']:
            if key in target and isinstance(target[key], np.ndarray):
                target[key] = jt.array(target[key])
        
        return image, target
    
    def __len__(self) -> int:
        """Get dataset length"""
        return self.total_len
    
    def collate_fn(self, batch: List[Tuple[jt.Var, Dict[str, Any]]]) -> Tuple[jt.Var, List[Dict[str, Any]]]:
        """
        Collate function for batching
        
        Args:
            batch: List of (image, target) tuples
            
        Returns:
            Batched images and list of targets
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Stack images
        batched_images = jt.stack(images, dim=0)
        
        return batched_images, targets


class ODVGDataset(Dataset):
    """
    ODVG (Object Detection + Visual Grounding) Dataset for Jittor
    
    Args:
        anno_path: Path to ODVG annotation file
        image_dir: Directory containing images
        transforms: Data transforms to apply
        is_train: Whether this is a training dataset
        filter_empty: Whether to filter out images with no annotations
    """
    
    def __init__(
        self,
        anno_path: str,
        image_dir: str,
        transforms: Optional[Compose] = None,
        is_train: bool = True,
        filter_empty: bool = True
    ):
        super().__init__()
        
        self.anno_path = anno_path
        self.image_dir = image_dir
        self.is_train = is_train
        self.filter_empty = filter_empty
        
        # Load annotations
        with open(anno_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter out images with no annotations if required
        if self.filter_empty:
            self.data = [item for item in self.data if len(item.get('bboxes', [])) > 0]
        
        # Set transforms
        self.transforms = transforms if transforms else build_transforms(is_train)
        
        # Set dataset attributes for Jittor
        self.total_len = len(self.data)
        
        # Print dataset info
        print(f"Loaded {self.total_len} images from {anno_path}")
    
    def __getitem__(self, idx: int) -> Tuple[jt.Var, Dict[str, Any]]:
        """
        Get item by index
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, target_dict)
        """
        # Get annotation data
        item = self.data[idx]
        
        # Load image
        image_path = item.get('image_path', os.path.join(self.image_dir, item['file_name']))
        image = PIL.Image.open(image_path).convert('RGB')
        
        # Prepare target
        target = {
            'image_id': item.get('image_id', idx),
            'file_name': item['file_name'],
            'height': item.get('height', image.height),
            'width': item.get('width', image.width),
            'boxes': item.get('bboxes', []),
            'labels': item.get('labels', []),
            'categories': item.get('categories', []),
            'text': item.get('text', ''),
        }
        
        # Convert boxes to numpy array if needed
        if target['boxes']:
            target['boxes'] = np.array(target['boxes'], dtype=np.float32)
        
        # Apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
        
        # Convert target to proper format
        if isinstance(target['boxes'], list):
            target['boxes'] = np.array(target['boxes'], dtype=np.float32) if target['boxes'] else np.zeros((0, 4), dtype=np.float32)
        
        # Convert to Jittor tensors
        if not isinstance(image, jt.Var):
            image = jt.array(np.array(image))
        
        # Convert target fields to Jittor tensors
        for key in ['boxes']:
            if key in target and isinstance(target[key], np.ndarray):
                target[key] = jt.array(target[key])
        
        return image, target
    
    def __len__(self) -> int:
        """Get dataset length"""
        return self.total_len
    
    def collate_fn(self, batch: List[Tuple[jt.Var, Dict[str, Any]]]) -> Tuple[jt.Var, List[Dict[str, Any]]]:
        """
        Collate function for batching
        
        Args:
            batch: List of (image, target) tuples
            
        Returns:
            Batched images and list of targets
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Stack images
        batched_images = jt.stack(images, dim=0)
        
        return batched_images, targets


def build_dataset(
    image_set: str,
    args: Dict[str, Any]
) -> Dataset:
    """
    Build dataset based on image set and arguments
    
    Args:
        image_set: 'train', 'val', or 'test'
        args: Dictionary of arguments
        
    Returns:
        Dataset instance
    """
    if args['dataset_file'] == 'lvis':
        # LVIS dataset
        paths = {
            'train': ('lvis_train.json', 'lvis_train'),
            'val': ('lvis_val.json', 'lvis_val'),
            'test': ('lvis_test.json', 'lvis_test')
        }
        
        json_file, img_dir = paths[image_set]
        anno_path = os.path.join(args['data_path'], json_file)
        img_dir = os.path.join(args['data_path'], img_dir)
        
        dataset = LVISDataset(
            anno_path=anno_path,
            image_dir=img_dir,
            transforms=build_transforms(is_train=(image_set == 'train')),
            is_train=(image_set == 'train')
        )
    elif args['dataset_file'] == 'odvg':
        # ODVG dataset
        paths = {
            'train': ('odvg_train.json', 'train'),
            'val': ('odvg_val.json', 'val'),
            'test': ('odvg_test.json', 'test')
        }
        
        json_file, img_dir = paths[image_set]
        anno_path = os.path.join(args['data_path'], json_file)
        img_dir = os.path.join(args['data_path'], img_dir)
        
        dataset = ODVGDataset(
            anno_path=anno_path,
            image_dir=img_dir,
            transforms=build_transforms(is_train=(image_set == 'train')),
            is_train=(image_set == 'train')
        )
    else:
        raise ValueError(f"Unsupported dataset file: {args['dataset_file']}")
    
    return dataset

