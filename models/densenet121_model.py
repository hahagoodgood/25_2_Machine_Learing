"""
DenseNet121 모델 구조
"""
import torch
import torch.nn as nn
from torchvision import models

import config


class DenseNet121Model(nn.Module):
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes (int): 출력 클래스 수
            pretrained (bool): 사전학습 가중치 사용 여부
            dropout (float): Dropout 비율
        """
        super(DenseNet121Model, self).__init__()
        
        # 사전학습된 DenseNet121 로드
        self.densenet121 = models.densenet121(pretrained=pretrained)
        
        # 분류기의 입력 특징 수 가져오기
        num_features = self.densenet121.classifier.in_features
        
        # 분류기를 커스텀 헤드로 교체
        self.densenet121.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        # 선택적으로 백본 동결(사전 학습된 파라미터를 그대로 가지고 가기 위해)
        if config.FREEZE_BACKBONE:
            for name, param in self.densenet121.named_parameters():
                if 'classifier' not in name:  # 최종 분류기는 동결하지 않음
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass"""
        return self.densenet121(x)


def get_densenet121_model(num_classes=None, pretrained=True, dropout=None):
    """
    Args:
        num_classes (int, optional): 출력 클래스 수
        pretrained (bool): 사전학습 가중치 사용 여부
        dropout (float, optional): Dropout 비율
        
    Returns:
        DenseNet121Model: 모델 인스턴스
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout is None:
        dropout = config.DENSENET121_DROPOUT
    
    model = DenseNet121Model(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    
    print(f'\nDenseNet121 Model initialized:')
    print(f'  Number of classes: {num_classes}')
    print(f'  Pretrained: {pretrained}')
    print(f'  Dropout: {dropout}')
    print(f'  Freeze backbone: {config.FREEZE_BACKBONE}')
    
    return model


if __name__ == '__main__':
    # 모델 테스트
    print('DenseNet121 모델 테스트 중...')
    model = get_densenet121_model(num_classes=3, pretrained=False)
    
    # 순전파 테스트
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f'\nModel test:')
    print(f'  Input shape: {x.shape}')
    print(f'  Output shape: {output.shape}')
    
    # 파라미터 개수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    
    print('\nDenseNet121 model test completed successfully!')
