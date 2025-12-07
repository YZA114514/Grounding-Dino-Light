# -*- coding: utf-8 -*-
"""
LVIS Dataset - 支持原始 COCO/LVIS 格式的数据加载器

LVIS 数据集格式 (COCO-style):
{
    "images": [{"id": int, "file_name": str, "height": int, "width": int}, ...],
    "annotations": [{"id": int, "image_id": int, "category_id": int, "bbox": [x,y,w,h], "area": float}, ...],
    "categories": [{"id": int, "name": str, "synset": str}, ...]
}
"""

import os
import json
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import jittor as jt
from jittor.dataset import Dataset


class LVISDetectionDataset(Dataset):
    """
    LVIS 数据集加载器 - 支持原始 COCO/LVIS JSON 格式
    
    用于 Grounding DINO 的目标检测训练和评估
    """
    
    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        transforms=None,
        is_train: bool = True,
        max_text_len: int = 256,
        use_category_names: bool = True,
        category_name_file: Optional[str] = None,
    ):
        """
        Args:
            ann_file: LVIS 标注文件路径 (JSON)
            image_dir: 图像目录路径
            transforms: 数据变换
            is_train: 是否为训练集
            max_text_len: 文本最大长度
            use_category_names: 是否使用类别名称作为文本 prompt
            category_name_file: 自定义类别名称文件 (可选)
        """
        super().__init__()
        
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        self.max_text_len = max_text_len
        self.use_category_names = use_category_names
        
        # 加载标注
        print(f"Loading LVIS annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            self.lvis_data = json.load(f)
        
        # 解析数据结构
        self.images = {img['id']: img for img in self.lvis_data['images']}
        self.categories = {cat['id']: cat for cat in self.lvis_data['categories']}
        
        # 按图像分组标注
        self.img_to_anns = defaultdict(list)
        for ann in self.lvis_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # 获取有效图像列表 (有标注的图像)
        if is_train:
            # 训练时只使用有标注的图像
            self.image_ids = [img_id for img_id in self.images.keys() 
                            if len(self.img_to_anns[img_id]) > 0]
        else:
            # 评估时使用所有图像
            self.image_ids = list(self.images.keys())
        
        # 构建类别ID到索引的映射
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.categories.keys())}
        self.idx_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}
        
        # 构建类别名称列表
        self.category_names = [self.categories[cat_id]['name'] 
                              for cat_id in sorted(self.categories.keys())]
        
        # 如果提供了自定义类别名称文件
        if category_name_file and os.path.exists(category_name_file):
            with open(category_name_file, 'r') as f:
                custom_names = json.load(f)
                self.category_names = custom_names
        
        self.total_len = len(self.image_ids)
        self.num_classes = len(self.categories)
        
        print(f"  Total images: {len(self.images)}")
        print(f"  Images with annotations: {len(self.image_ids)}")
        print(f"  Total annotations: {len(self.lvis_data['annotations'])}")
        print(f"  Number of categories: {self.num_classes}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx: int) -> Tuple[jt.Var, Dict[str, Any]]:
        """获取数据项"""
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 获取标注
        anns = self.img_to_anns[image_id]
        
        # 准备边界框和标签
        boxes = []
        labels = []
        category_ids = []
        
        for ann in anns:
            # LVIS bbox 格式: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # 转换为归一化的 [cx, cy, w, h] 格式
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            
            # 确保边界框在有效范围内
            if nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                cat_id = ann['category_id']
                labels.append(self.cat_id_to_idx[cat_id])
                category_ids.append(cat_id)
        
        # 构建文本 prompt
        if self.use_category_names:
            # 使用图像中出现的类别名称
            unique_cats = list(set(category_ids))
            cat_names = [self.categories[cid]['name'] for cid in unique_cats]
            caption = '. '.join(cat_names) + '.'
        else:
            caption = "object."
        
        # 转换为张量
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        # 构建目标字典
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'category_ids': category_ids,
            'caption': caption,
            'orig_size': np.array([orig_h, orig_w]),
            'size': np.array([orig_h, orig_w]),
        }
        
        # 应用变换
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # 转换为 Jittor 张量
        if not isinstance(image, jt.Var):
            image = jt.array(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        target['boxes'] = jt.array(target['boxes'])
        target['labels'] = jt.array(target['labels'])
        
        return image, target
    
    def get_category_names(self) -> List[str]:
        """获取所有类别名称"""
        return self.category_names
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return self.num_classes
    
    def collate_fn(self, batch):
        """批次整理函数"""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # 填充图像到相同尺寸
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        padded_images = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            if pad_h > 0 or pad_w > 0:
                padded = jt.zeros((3, max_h, max_w))
                padded[:, :img.shape[1], :img.shape[2]] = img
                padded_images.append(padded)
            else:
                padded_images.append(img)
        
        batched_images = jt.stack(padded_images, dim=0)
        
        return batched_images, targets


def build_lvis_dataset(
    ann_file: str,
    image_dir: str,
    is_train: bool = True,
    **kwargs
) -> LVISDetectionDataset:
    """构建 LVIS 数据集"""
    from .transforms import build_transforms
    transforms = build_transforms(is_train=is_train)
    
    return LVISDetectionDataset(
        ann_file=ann_file,
        image_dir=image_dir,
        transforms=transforms,
        is_train=is_train,
        **kwargs
    )

