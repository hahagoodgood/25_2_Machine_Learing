"""
COVID-19 이미지 분류를 위한 데이터셋 로더
데이터 로딩, 증강 및 훈련/검증/테스트 데이터 분할 처리
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

import config


class COVID19Dataset(Dataset):
    """COVID-19 흉부 X-ray 이미지를 위한 커스텀 데이터셋"""
    
    def __init__(self, root_dir, transform=None, is_test=False):
        """
        Args:
            root_dir (str): 클래스별로 정리된 모든 이미지가 있는 디렉토리
            transform (callable, optional): 샘플에 적용할 선택적 변환
            is_test (bool): 테스트 데이터셋 여부
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
        # 모든 이미지 경로와 레이블 가져오기
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # 하위 디렉토리에서 클래스 자동 감지
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
        
        print(f'Loaded {len(self.image_paths)} images from {root_dir}')
        for class_name, idx in self.class_to_idx.items():
            count = self.labels.count(idx)
            print(f'  {class_name}: {count} images')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """클래스별 샘플 분포 가져오기"""
        distribution = {}
        for idx, class_name in self.idx_to_class.items():
            distribution[class_name] = self.labels.count(idx)
        return distribution


def get_transforms(augment=True):
    """
    훈련/검증을 위한 데이터 변환 가져오기
    
    Args:
        augment (bool): 데이터 증강 적용 여부
        
    Returns:
        transforms.Compose: 구성된 변환
    """
    if augment:
        # 강력한 증강이 적용된 훈련 변환
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        # 검증/테스트 변환 (증강 없음)
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    
    return transform


def calculate_class_weights(dataset):
    """
    불균형 데이터셋 처리를 위한 클래스 가중치 계산
    
    Args:
        dataset (COVID19Dataset): 데이터셋 객체
        
    Returns:
        torch.Tensor: 클래스 가중치
    """
    class_counts = [dataset.labels.count(i) for i in range(len(dataset.class_to_idx))]
    total_samples = len(dataset)
    
    # 가중치 = 전체_샘플 / (클래스_수 * 클래스_샘플_수)
    num_classes = len(dataset.class_to_idx)
    weights = [total_samples / (num_classes * count) for count in class_counts]
    
    weights_tensor = torch.FloatTensor(weights)
    
    print(f'\nClass weights for imbalanced dataset:')
    for idx, (class_name, weight) in enumerate(zip(dataset.idx_to_class.values(), weights)):
        print(f'  {class_name}: {weight:.4f} (count: {class_counts[idx]})')
    
    return weights_tensor


def get_data_loaders(dataset_dir, batch_size=None, validation_split=None, num_workers=None):
    """
    훈련, 검증 및 테스트를 위한 데이터 로더 생성
    
    Args:
        dataset_dir (str): train/ 및 test/ 하위 디렉토리를 포함하는 루트 디렉토리
        batch_size (int, optional): 데이터 로더를 위한 배치 크기
        validation_split (float, optional): 검증을 위한 훈련 데이터의 비율
        num_workers (int, optional): 데이터 로딩을 위한 워커 수
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if validation_split is None:
        validation_split = config.VALIDATION_SPLIT
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    # 전체 훈련 데이터셋 로드
    train_transform = get_transforms(augment=True)
    full_train_dataset = COVID19Dataset(
        root_dir=os.path.join(dataset_dir, 'train'),
        transform=train_transform,
        is_test=False
    )
    
    # 훈련 데이터를 훈련 및 검증으로 분할
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 재현성을 위해 고정된 시드로 random_split 사용
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset_temp = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # 다른 변환을 사용하여 검증 데이터셋 생성 (증강 없음)
    val_transform = get_transforms(augment=False)
    val_dataset_full = COVID19Dataset(
        root_dir=os.path.join(dataset_dir, 'train'),
        transform=val_transform,
        is_test=False
    )
    
    # 분할에서 검증 인덱스 가져오기
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    # 테스트 데이터셋 로드
    test_transform = get_transforms(augment=False)
    test_dataset = COVID19Dataset(
        root_dir=os.path.join(dataset_dir, 'test'),
        transform=test_transform,
        is_test=True
    )
    
    # 전체 훈련 데이터셋에서 클래스 가중치 계산
    class_weights = calculate_class_weights(full_train_dataset)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f'\nDataset splits:')
    print(f'  Training samples: {len(train_dataset)}')
    print(f'  Validation samples: {len(val_dataset)}')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  Batch size: {batch_size}')
    print(f'  Training batches: {len(train_loader)}')
    print(f'  Validation batches: {len(val_loader)}')
    print(f'  Test batches: {len(test_loader)}')
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == '__main__':
    # 데이터셋 로더 테스트
    print('Testing dataset loader...\n')
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        config.DATASET_DIR,
        batch_size=4
    )
    
    # 훈련 로더에서 배치 가져오기
    images, labels = next(iter(train_loader))
    print(f'\nBatch test:')
    print(f'  Images shape: {images.shape}')
    print(f'  Labels shape: {labels.shape}')
    print(f'  Labels: {labels}')
    print(f'\nDataset loader test completed successfully!')
