"""
ResNet50 모델 구조
"""
import torch
import torch.nn as nn
from torchvision import models

import config


class ResNet50Model(nn.Module):
    
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes (int): 출력 클래스 수
            pretrained (bool): 사전학습 가중치 사용 여부
            dropout (float): Dropout 비율
        """
        super(ResNet50Model, self).__init__()
        
        # 사전학습된 ResNet50 로드
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # 최종 레이어의 입력 특징 수 가져오기
        num_features = self.resnet50.fc.in_features
        
        # 최종 fully connected 레이어를 커스텀 분류기로 교체
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        # 선택적으로 백본 동결(사전 학습된 파라미터를 그대로 가지고 가기 위해)
        if config.FREEZE_BACKBONE:
            for name, param in self.resnet50.named_parameters():
                if 'fc' not in name:  # 최종 분류기는 동결하지 않음
                    param.requires_grad = False
    
    def forward(self, x):
        return self.resnet50(x)


def get_resnet50_model(num_classes=None, pretrained=True, dropout=None):
    """
    Args:
        num_classes (int, optional): 출력 클래스 수
        pretrained (bool): 사전학습 가중치 사용 여부
        dropout (float, optional): Dropout 비율
        
    Returns:
        ResNet50Model: 모델 인스턴스
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if dropout is None:
        dropout = config.RESNET50_DROPOUT
    
    model = ResNet50Model(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    
    print(f'\nResNet50 모델 초기화 완료:')
    print(f'  클래스 수: {num_classes}')
    print(f'  사전학습 모델 사용: {pretrained}')
    print(f'  Dropout: {dropout}')
    print(f'  백본 동결: {config.FREEZE_BACKBONE}')
    
    return model


if __name__ == '__main__':
    # 모델 테스트
    print('ResNet50 모델 테스트 중...')
    model = get_resnet50_model(num_classes=3, pretrained=False)
    
    # 순전파 테스트
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f'\nModel test:')
    print(f'  Input shape: {x.shape}')
    print(f'  Output shape: {output.shape}')
    
    # 파라미터 개수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  전체 파라미터: {total_params:,}')
    print(f'  학습 가능한 파라미터: {trainable_params:,}')
    
    print('\nResNet50 모델 테스트 성공적으로 완료!')

