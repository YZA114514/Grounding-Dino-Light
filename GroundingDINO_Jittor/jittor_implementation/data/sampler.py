# Data Sampler (Member B)
import numpy as np
import random
from typing import List, Dict, Any, Optional
import jittor as jt
from jittor.dataset import Dataset


class LVISSampler:
    """
    LVIS dataset sampler that handles long-tailed distribution
    
    Args:
        dataset: LVIS dataset instance
        samples_per_epoch: Number of samples to draw per epoch
        repeat_factor: Repeat factor for rare categories
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_epoch: int = 1000,
        repeat_factor: float = 0.5,
        seed: int = 42
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.repeat_factor = repeat_factor
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Calculate category frequencies
        self._calc_category_freq()
        
        # Calculate repeat factors for each image
        self._calc_repeat_factors()
        
        # Create sampling indices
        self._create_sampling_indices()
    
    def _calc_category_freq(self) -> None:
        """Calculate frequency of each category in the dataset"""
        self.category_freq = {}
        
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            categories = target.get('categories', [])
            
            for cat in categories:
                if cat not in self.category_freq:
                    self.category_freq[cat] = 0
                self.category_freq[cat] += 1
        
        # Sort categories by frequency
        self.sorted_categories = sorted(
            self.category_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate category-wise repeat factors
        self.category_repeat_factors = {}
        max_freq = max(self.category_freq.values()) if self.category_freq else 1
        
        for cat, freq in self.category_freq.items():
            # Use exponential decay for repeat factor
            repeat_factor = (freq / max_freq) ** self.repeat_factor
            self.category_repeat_factors[cat] = repeat_factor
    
    def _calc_repeat_factors(self) -> None:
        """Calculate repeat factor for each image based on its categories"""
        self.image_repeat_factors = []
        
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            categories = target.get('categories', [])
            
            if not categories:
                # If no categories, use minimum repeat factor
                repeat_factor = 0.1
            else:
                # Use maximum repeat factor among all categories in the image
                repeat_factor = max(
                    [self.category_repeat_factors.get(cat, 0.1) for cat in categories]
                )
            
            self.image_repeat_factors.append(repeat_factor)
    
    def _create_sampling_indices(self) -> None:
        """Create sampling indices based on repeat factors"""
        self.sampling_indices = []
        
        for i, repeat_factor in enumerate(self.image_repeat_factors):
            # Number of times to include this image in the sampling pool
            repeat_count = max(1, int(repeat_factor * 10))
            self.sampling_indices.extend([i] * repeat_count)
        
        # Shuffle the sampling indices
        random.shuffle(self.sampling_indices)
    
    def __iter__(self):
        """Iterator for sampling indices"""
        # Reset and shuffle indices for each epoch
        random.shuffle(self.sampling_indices)
        
        # Sample indices for this epoch
        epoch_indices = []
        for _ in range(self.samples_per_epoch):
            idx = random.choice(self.sampling_indices)
            epoch_indices.append(idx)
        
        return iter(epoch_indices)
    
    def __len__(self) -> int:
        """Return the number of samples per epoch"""
        return self.samples_per_epoch


class BalancedSampler:
    """
    Balanced sampler that ensures equal representation of categories
    
    Args:
        dataset: Dataset instance
        samples_per_class: Number of samples to draw per class
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_class: int = 10,
        seed: int = 42
    ):
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Group images by category
        self._group_images_by_category()
        
        # Calculate total samples per epoch
        self.total_samples = len(self.category_to_images) * samples_per_class
    
    def _group_images_by_category(self) -> None:
        """Group image indices by their categories"""
        self.category_to_images = {}
        
        for i in range(len(self.dataset)):
            _, target = self.dataset[i]
            categories = target.get('categories', [])
            
            for cat in categories:
                if cat not in self.category_to_images:
                    self.category_to_images[cat] = []
                self.category_to_images[cat].append(i)
    
    def __iter__(self):
        """Iterator for sampling indices"""
        epoch_indices = []
        
        for cat, image_indices in self.category_to_images.items():
            # Sample images for this category
            if len(image_indices) >= self.samples_per_class:
                # If we have enough images, sample without replacement
                sampled_indices = random.sample(image_indices, self.samples_per_class)
            else:
                # If not enough images, sample with replacement
                sampled_indices = random.choices(image_indices, k=self.samples_per_class)
            
            epoch_indices.extend(sampled_indices)
        
        # Shuffle the indices
        random.shuffle(epoch_indices)
        
        return iter(epoch_indices)
    
    def __len__(self) -> int:
        """Return the number of samples per epoch"""
        return self.total_samples


class DistributedSampler:
    """
    Distributed sampler for multi-GPU training
    
    Args:
        dataset: Dataset instance
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process within num_replicas
        shuffle: Whether to shuffle the indices
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        if num_replicas is None:
            # Default to the number of available GPUs
            from jittor.misc import get_world_size
            num_replicas = get_world_size()
        
        if rank is None:
            # Default to the current GPU rank
            from jittor.misc import get_rank
            rank = get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate the number of samples per process
        self.num_samples = int(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas
        
        # Create indices
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Iterator for sampling indices"""
        # Determine the indices for this process
        if self.shuffle:
            # Shuffle indices
            random.seed(self.seed)
            random.shuffle(self.indices)
        
        # Calculate start and end indices for this process
        start = self.rank * self.num_samples
        end = start + self.num_samples
        
        # Get indices for this process
        process_indices = self.indices[start:end]
        
        return iter(process_indices)
    
    def __len__(self) -> int:
        """Return the number of samples for this process"""
        return self.num_samples


def build_sampler(
    sampler_type: str,
    dataset: Dataset,
    **kwargs
) -> Any:
    """
    Build sampler based on type and arguments
    
    Args:
        sampler_type: Type of sampler ('lvis', 'balanced', 'distributed')
        dataset: Dataset instance
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Sampler instance
    """
    if sampler_type == 'lvis':
        return LVISSampler(dataset, **kwargs)
    elif sampler_type == 'balanced':
        return BalancedSampler(dataset, **kwargs)
    elif sampler_type == 'distributed':
        return DistributedSampler(dataset, **kwargs)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler_type: Optional[str] = None,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Create a dataloader with optional sampler
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        sampler_type: Type of sampler to use
        sampler_kwargs: Arguments for the sampler
        **kwargs: Additional arguments for the dataloader
        
    Returns:
        Dataloader instance
    """
    # Create sampler if specified
    sampler = None
    if sampler_type:
        sampler_kwargs = sampler_kwargs or {}
        sampler = build_sampler(sampler_type, dataset, **sampler_kwargs)
        # If using a sampler, disable shuffle
        shuffle = False
    
    # Create dataloader
    dataloader = dataset.set_attrs(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        **kwargs
    )
    
    return dataloader

