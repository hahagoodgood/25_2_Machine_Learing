"""
VGG16 모델 구조
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models

import config


class VGG16Model(nn.Module):
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.5):
        """
        Args:
            num_classes (int): 출력 클래스 수
            pretrained (bool): 사전학습 가중치 사용 여부
            dropout (float): Dropout 비율
        """
        super(VGG16Model, self).__init__()
        
        # 사전학습된 VGG16 로드
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # 분류기의 입력 특징 수 가져오기
        num_features = self.vgg16.classifier[0].in_features
        
        # 분류기를 커스텀 헤드로 교체
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
        
        # 선택적으로 백본 동결(사전 학습된 파라미터를 그대로 가지고 가기 위해)
        if config.FREEZE_BACKBONE:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.vgg16(x)


def get_vgg16_model(num_classes=None, pretrained=True, dropout=None):
    """
    Args:
        num_classes (int, optional): 출력 클래스 수
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부
        dropout (float, optional): Dropout 비율
        
    Returns:
        VGG16Model: 모델 인스턴스
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout is None:
        dropout = config.VGG16_DROPOUT
    
    model = VGG16Model(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    
    print(f'\nVGG16 모델 초기화 완료:')
    print(f'  클래스 수: {num_classes}')
    print(f'  사전학습 모델 사용: {pretrained}')
    print(f'  Dropout: {dropout}')
    print(f'  백본 동결: {config.FREEZE_BACKBONE}')
    
    return model


if __name__ == '__main__':
    # 모델 테스트
    print('VGG16 모델 테스트 중...')
    model = get_vgg16_model(num_classes=3, pretrained=False)
    
    # 순전파 테스트
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f'\n모델 테스트:')
    print(f'  입력 shape: {x.shape}')
    print(f'  출력 shape: {output.shape}')
    
    # 파라미터 개수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  전체 파라미터: {total_params:,}')
    print(f'  학습 가능한 파라미터: {trainable_params:,}')
    
    print('\nVGG16 모델 테스트 성공적으로 완료!')
