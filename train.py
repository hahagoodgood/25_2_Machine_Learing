"""
COVID-19 흉부 X-ray 이미지 분류 모델 학습 스크립트

이 스크립트는 COVID-19 진단을 위한 딥러닝 모델을 학습합니다.
지원하는 모델 아키텍처: VGG16, ResNet50, DenseNet121

주요 기능:
- 전이 학습(Transfer Learning)을 활용한 모델 학습
- 클래스 불균형 처리를 위한 가중치 적용
- 조기 종료(Early Stopping)를 통한 과적합 방지
- 학습률 스케줄러를 통한 동적 학습률 조정
- TensorBoard를 활용한 학습 과정 시각화
- 체크포인트 저장 및 최적 모델 선택

사용 예시:
    python train.py --model vgg16 --epochs 100 --batch_size 32 --lr 0.001
"""

import os
import argparse
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

import config
from dataset import get_data_loaders
from models import get_model
from utils import set_seed, get_device, save_checkpoint, count_parameters, CheckpointManager


class EarlyStopping:
    """
    조기 종료(Early Stopping) 클래스
    
    검증 성능이 일정 에포크 동안 개선되지 않으면 학습을 조기에 종료합니다.
    이를 통해 과적합(Overfitting)을 방지하고 학습 시간을 절약할 수 있습니다.
    
    Attributes:
        patience (int): 성능 개선이 없어도 기다리는 최대 에포크 수
        min_delta (float): 성능 개선으로 인정되는 최소 변화량
        mode (str): 'max'는 높을수록 좋음(정확도), 'min'은 낮을수록 좋음(손실)
        counter (int): 성능 개선 없이 경과한 에포크 수
        best_score (float): 지금까지의 최고 성능 점수
        early_stop (bool): 조기 종료 여부 플래그
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        EarlyStopping 초기화
        
        Args:
            patience (int): 성능 개선이 없을 때 기다리는 에포크 수.
                           기본값 10은 10번의 에포크 동안 개선이 없으면 학습을 종료함을 의미합니다.
            min_delta (float): 성능 개선으로 인정하기 위한 최소 변화량.
                              너무 작은 개선은 노이즈일 수 있으므로 이 값으로 필터링합니다.
            mode (str): 성능 측정 방향 설정.
                       'max': 값이 클수록 좋음 (예: 정확도)
                       'min': 값이 작을수록 좋음 (예: 손실값)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        현재 에포크의 성능 점수를 기반으로 조기 종료 여부를 결정합니다.
        
        Args:
            score (float): 현재 에포크의 검증 성능 점수 (정확도 또는 손실값)
        
        동작 방식:
            1. 첫 번째 호출 시 현재 점수를 최고 점수로 설정
            2. 이후 호출에서 개선 여부를 확인
            3. 개선되지 않으면 카운터 증가, 개선되면 카운터 리셋
            4. 카운터가 patience에 도달하면 early_stop 플래그를 True로 설정
        """
        if self.best_score is None:
            # 첫 번째 에포크: 현재 점수를 기준으로 설정
            self.best_score = score
        elif self.mode == 'max':
            # 정확도 등 값이 클수록 좋은 경우
            if score < self.best_score + self.min_delta:
                # 성능이 개선되지 않음 - 카운터 증가
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                # 성능이 개선됨 - 최고 점수 갱신 및 카운터 리셋
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            # 손실값 등 값이 작을수록 좋은 경우
            if score > self.best_score - self.min_delta:
                # 성능이 개선되지 않음 - 카운터 증가
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                # 성능이 개선됨 - 최고 점수 갱신 및 카운터 리셋
                self.best_score = score
                self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """
    한 에포크 동안 모델을 학습시키는 함수 (모든 평가 지표 포함)
    
    이 함수에서는 순전파(Forward Pass), 손실 계산, 역전파(Backward Pass),
    가중치 업데이트의 전체 학습 과정을 수행합니다.
    
    Args:
        model (nn.Module): 학습할 신경망 모델
        train_loader (DataLoader): 학습 데이터를 배치 단위로 제공하는 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device (torch.device): 연산을 수행할 장치 (CPU 또는 GPU)
        epoch (int): 현재 에포크 번호
        writer (SummaryWriter, optional): TensorBoard 로깅을 위한 writer 객체
        
    Returns:
        dict: 모든 평가 지표를 포함하는 딕셔너리
            - loss: 평균 손실값
            - accuracy: 정확도 (%)
            - recall: Macro Recall (%)
            - precision: Macro Precision (%)
            - f1_score: Macro F1-Score (%)
    """
    model.train()
    
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    # 배치별 메트릭 기록
    batch_metrics = {
        'loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1_score': []
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        if config.GRADIENT_CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        # 배치별 메트릭 계산 및 기록
        batch_labels = np.array(all_labels)
        batch_preds = np.array(all_predictions)
        
        batch_loss = running_loss / (batch_idx + 1)
        batch_acc = 100. * (batch_preds == batch_labels).mean()
        batch_recall = 100. * recall_score(batch_labels, batch_preds, average='macro', zero_division=0)
        batch_precision = 100. * precision_score(batch_labels, batch_preds, average='macro', zero_division=0)
        batch_f1 = 100. * f1_score(batch_labels, batch_preds, average='macro', zero_division=0)
        
        batch_metrics['loss'].append(batch_loss)
        batch_metrics['accuracy'].append(batch_acc)
        batch_metrics['recall'].append(batch_recall)
        batch_metrics['precision'].append(batch_precision)
        batch_metrics['f1_score'].append(batch_f1)
        
        pbar.set_postfix({
            'loss': batch_loss,
            'acc': batch_acc,
            'recall': batch_recall
        })
    
    # 에포크 전체 통계 (마지막 배치 기준)
    avg_loss = batch_metrics['loss'][-1]
    accuracy = batch_metrics['accuracy'][-1]
    recall = batch_metrics['recall'][-1]
    precision = batch_metrics['precision'][-1]
    f1 = batch_metrics['f1_score'][-1]
    
    # TensorBoard에 학습 메트릭 로깅
    if writer:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('Recall/train', recall, epoch)
        writer.add_scalar('Precision/train', precision, epoch)
        writer.add_scalar('F1-Score/train', f1, epoch)
    
    # 에포크 요약 및 배치별 메트릭 반환
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'batch_metrics': batch_metrics  # 배치별 상세 기록
    }
    
    return metrics


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """
    검증 데이터셋에서 모델 성능을 평가하는 함수 (의료 AI 평가 지표 포함)
    
    의료 AI에서 중요한 평가 지표들을 계산합니다:
    - Recall (주 지표): 실제 환자를 놓치지 않는 것이 가장 중요
    - Precision: 오진률 최소화
    - Macro F1-Score: 전체 클래스 균형 성능
    - AUC-ROC: 임계값 독립적 성능 평가
    - Accuracy: 전체 정확도
    
    Args:
        model (nn.Module): 평가할 신경망 모델
        val_loader (DataLoader): 검증 데이터를 배치 단위로 제공하는 데이터 로더
        criterion: 손실 함수. 검증 손실을 계산하는 데 사용됩니다.
        device (torch.device): 연산을 수행할 장치 (CPU 또는 GPU)
        epoch (int): 현재 에포크 번호. 로깅에 사용됩니다.
        writer (SummaryWriter, optional): TensorBoard 로깅을 위한 writer 객체
        
    Returns:
        dict: 모든 평가 지표를 포함하는 딕셔너리
            - loss: 평균 손실값
            - accuracy: 정확도 (%)
            - recall: Macro Recall (%) - 주 평가 지표
            - precision: Macro Precision (%)
            - f1_score: Macro F1-Score (%)
            - auc_roc: Macro AUC-ROC (0~1)
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 통계 추적을 위한 변수 초기화
    running_loss = 0.0
    
    # 전체 예측값과 레이블 저장 (메트릭 계산용)
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 손실 누적
            running_loss += loss.item()
            
            # 예측 확률과 클래스 추출
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # 결과 저장
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 진행률 표시
            current_acc = 100. * (np.array(all_predictions) == np.array(all_labels)).mean()
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': current_acc
            })
    
    # numpy 배열로 변환
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # 평가 지표 계산
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * (all_predictions == all_labels).mean()
    
    # Macro 평균 (모든 클래스 동등하게 취급)
    recall = 100. * recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = 100. * precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # AUC-ROC 계산 (다중 클래스)
    try:
        # One-vs-Rest 방식으로 AUC 계산
        n_classes = all_probabilities.shape[1]
        all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
        auc_roc = roc_auc_score(all_labels_bin, all_probabilities, average='macro', multi_class='ovr')
    except ValueError:
        # 일부 클래스가 없는 경우 예외 처리
        auc_roc = 0.0
    
    # TensorBoard에 메트릭 로깅
    if writer:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('F1-Score/val', f1, epoch)
        writer.add_scalar('AUC-ROC/val', auc_roc, epoch)
    
    # 모든 메트릭을 딕셔너리로 반환
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'recall': recall,          # 주 평가 지표
        'precision': precision,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    
    return metrics


def test(model, test_loader, device):
    """
    테스트 데이터셋에서 최종 모델 성능을 평가하는 함수 (의료 AI 평가 지표 포함)
    
    학습이 완료된 후 모델의 일반화 성능을 측정하기 위해
    학습에 사용되지 않은 테스트 데이터셋에서 최종 평가를 수행합니다.
    
    Args:
        model (nn.Module): 평가할 학습된 신경망 모델
        test_loader (DataLoader): 테스트 데이터를 배치 단위로 제공하는 데이터 로더
        device (torch.device): 연산을 수행할 장치 (CPU 또는 GPU)
        
    Returns:
        dict: 모든 평가 지표를 포함하는 딕셔너리
            - accuracy: 정확도 (%)
            - recall: Macro Recall (%) - 주 평가 지표
            - precision: Macro Precision (%)
            - f1_score: Macro F1-Score (%)
            - auc_roc: Macro AUC-ROC (0~1)
            - labels: 실제 레이블 리스트
            - predictions: 예측 레이블 리스트
            - probabilities: 예측 확률 리스트
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            current_acc = 100. * (np.array(all_predictions) == np.array(all_labels)).mean()
            pbar.set_postfix({'acc': current_acc})
    
    # numpy 배열로 변환
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # 평가 지표 계산
    accuracy = 100. * (all_predictions == all_labels).mean()
    recall = 100. * recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = 100. * precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = 100. * f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # AUC-ROC 계산
    try:
        n_classes = all_probabilities.shape[1]
        all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
        auc_roc = roc_auc_score(all_labels_bin, all_probabilities, average='macro', multi_class='ovr')
    except ValueError:
        auc_roc = 0.0
    
    # 모든 메트릭을 딕셔너리로 반환
    test_metrics = {
        'accuracy': accuracy,
        'recall': recall,           # 주 평가 지표
        'precision': precision,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'labels': all_labels,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }
    
    return test_metrics


def train_model(model_name, epochs=None, batch_size=None, learning_rate=None, device=None):
    """
    모델 학습의 전체 파이프라인을 관리하는 메인 학습 함수
    
    이 함수는 다음과 같은 전체 학습 과정을 관리합니다:
    1. 하이퍼파라미터 및 환경 설정
    2. 데이터 로딩 및 전처리
    3. 모델 생성 및 설정
    4. 학습 루프 실행
    5. 최적 모델 저장 및 테스트
    
    Args:
        model_name (str): 학습할 모델의 이름.
                         'vgg16', 'resnet50', 'densenet121' 중 하나를 선택합니다.
        epochs (int, optional): 학습할 총 에포크 수.
                               None이면 config.NUM_EPOCHS 값을 사용합니다.
        batch_size (int, optional): 한 번에 처리할 이미지 수.
                                   None이면 config.BATCH_SIZE 값을 사용합니다.
        learning_rate (float, optional): 학습률 (가중치 업데이트 크기).
                                        None이면 config.LEARNING_RATE 값을 사용합니다.
        device (torch.device, optional): 학습에 사용할 장치.
                                        None이면 자동으로 GPU/CPU를 감지합니다.
    
    Returns:
        tuple: (학습된 모델, 학습 기록, 테스트 정확도)
            - 학습된 모델 (nn.Module): 최적의 검증 성능을 보인 모델
            - 학습 기록 (dict): 에포크별 손실값과 정확도 기록
            - 테스트 정확도 (float): 테스트 데이터셋에서의 최종 정확도
    
    Note:
        학습 과정에서 다음 파일들이 생성됩니다:
        - {model_name}_best.pth: 최고 검증 성능의 체크포인트
        - {model_name}_final.pth: 최종 학습 완료 후 체크포인트
        - TensorBoard 로그 파일
    """
    # 하이퍼파라미터 기본값 설정
    # 인자가 전달되지 않으면 config 파일의 값을 사용
    if epochs is None:
        epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if device is None:
        device = get_device()
    
    # 재현성을 위한 랜덤 시드 설정
    # 동일한 시드를 사용하면 동일한 결과를 재현할 수 있음
    set_seed(config.RANDOM_SEED)
    
    # 학습 설정 정보 출력
    print('='*80)
    print(f'{model_name.upper()} 모델 학습 시작')
    print('='*80)
    print(f'학습 설정:')
    print(f'  에포크: {epochs}')
    print(f'  배치 크기: {batch_size}')
    print(f'  학습률: {learning_rate}')
    print(f'  장치: {device}')
    print(f'  조기 종료 인내심: {config.EARLY_STOPPING_PATIENCE}')
    print('='*80)
    
    # 데이터 로딩
    # 학습, 검증, 테스트 데이터 로더와 클래스 가중치를 반환받음
    print('\n데이터 로딩 중...')
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        config.DATASET_DIR,
        batch_size=batch_size
    )
    
    # 모델 생성 및 GPU로 이동
    # pretrained=True로 ImageNet 사전학습 가중치를 사용
    print(f'\nCreating {model_name} model...')
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # 모델 파라미터 수 출력 (학습 가능 파라미터 / 전체 파라미터)
    count_parameters(model)
    
    # 손실 함수 정의
    # 클래스 불균형을 처리하기 위해 클래스 가중치를 적용한 CrossEntropyLoss 사용
    # 레이블 스무딩: 과적합 방지 및 일반화 성능 향상
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING  # 레이블 스무딩 적용
    )
    print(f'  레이블 스무딩: {config.LABEL_SMOOTHING}')
    
    # 옵티마이저 정의
    # config.OPTIMIZER에 따라 적절한 옵티마이저 선택
    optimizer_name = config.OPTIMIZER.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=True
        )
    else:
        raise ValueError(f'지원하지 않는 옵티마이저: {config.OPTIMIZER}')
    
    print(f'옵티마이저: {config.OPTIMIZER.upper()} (weight_decay={config.WEIGHT_DECAY})')
    
    # 학습률 스케줄러 정의
    scheduler_type = config.LR_SCHEDULER_TYPE.lower()
    if scheduler_type == 'plateau':
        # ReduceLROnPlateau: 검증 성능이 정체되면 학습률 감소
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR,
            verbose=True
        )
    elif scheduler_type == 'cosine':
        # CosineAnnealingLR: 코사인 함수 형태로 학습률 감소
        t_max = config.LR_COSINE_T_MAX if config.LR_COSINE_T_MAX else epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config.LR_SCHEDULER_MIN_LR
        )
    elif scheduler_type == 'step':
        # StepLR: 일정 에포크마다 학습률 감소
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
    elif scheduler_type == 'none':
        # 스케줄러 없음
        scheduler = None
    else:
        raise ValueError(f'지원하지 않는 스케줄러: {config.LR_SCHEDULER_TYPE}')
    
    print(f'학습률 스케줄러: {config.LR_SCHEDULER_TYPE.upper()}')
    if config.LR_WARMUP_EPOCHS > 0:
        print(f'웜업 에포크: {config.LR_WARMUP_EPOCHS}')
    if config.GRADIENT_CLIP_NORM:
        print(f'그래디언트 클리핑: {config.GRADIENT_CLIP_NORM}')
    
    # TensorBoard writer 초기화
    # 학습 과정을 시각적으로 모니터링할 수 있음
    writer = SummaryWriter(os.path.join(config.TENSORBOARD_DIR, model_name))
    
    # 조기 종료 객체 초기화
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')
    
    # 학습 기록 저장을 위한 딕셔너리
    # 에포크별 손실값과 정확도를 저장하여 학습 과정을 분석할 수 있음
    history = {
        'train_loss': [],   # 학습 손실 기록
        'train_acc': [],    # 학습 정확도 기록
        'val_loss': [],     # 검증 손실 기록
        'val_acc': []       # 검증 정확도 기록
    }
    
    # 체크포인트 관리자 초기화 (설정에 따른 체크포인트 수 유지)
    checkpoint_manager = CheckpointManager(
        save_dir=config.CHECKPOINT_DIR,
        model_name=model_name,
        max_checkpoints=config.MAX_CHECKPOINTS
    )
    
    # 최고 모델 추적을 위한 변수
    best_val_recall = 0.0  # 최고 검증 Recall 추적
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print('\n' + '='*80)
    print('학습 시작...')
    print('='*80)
    
    # 학습 시간 측정 시작
    start_time = time.time()
    
    # 전체 에포크 프로그래스 바
    epoch_pbar = tqdm(range(1, epochs + 1), desc='전체 진행', position=0)
    
    # 메인 학습 루프
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f'전체 진행 [{epoch}/{epochs}]')
        
        # 학습 단계 (모든 평가 지표 포함)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 학습 메트릭 추출 (에포크 요약)
        train_loss = train_metrics['loss']
        train_acc = train_metrics['accuracy']
        train_recall = train_metrics['recall']
        train_precision = train_metrics['precision']
        train_f1 = train_metrics['f1_score']
        train_batch_metrics = train_metrics['batch_metrics']  # 배치별 상세 기록
        
        # 검증 단계 (의료 AI 평가 지표 포함)
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # 검증 메트릭 추출
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_recall = val_metrics['recall']      # 주 평가 지표
        val_precision = val_metrics['precision']
        val_f1 = val_metrics['f1_score']
        val_auc = val_metrics['auc_roc']
        
        # 학습 기록 업데이트
        # 에포크 단위 요약값 (val과 비교용)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 추가 메트릭 기록 (history에 키가 없으면 생성)
        if 'train_recall' not in history:
            history['train_recall'] = []
            history['train_precision'] = []
            history['train_f1'] = []
            history['train_batch'] = []  # 배치별 상세 기록
            history['val_recall'] = []
            history['val_precision'] = []
            history['val_f1'] = []
            history['val_auc'] = []
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)
        history['train_f1'].append(train_f1)
        history['train_batch'].append(train_batch_metrics)  # 에포크별 배치 기록
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # 학습률 스케줄러 업데이트
        # 웜업 기간에는 선형 웜업 적용
        if epoch <= config.LR_WARMUP_EPOCHS:
            # 웜업: 학습률을 선형으로 증가
            warmup_lr = learning_rate * (epoch / config.LR_WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        elif scheduler is not None:
            # 스케줄러 타입에 따라 다른 방식으로 업데이트
            if config.LR_SCHEDULER_TYPE.lower() == 'plateau':
                # ReduceLROnPlateau: 검증 지표 기반
                scheduler.step(val_recall)
            else:
                # Cosine, Step 등: 에포크 기반
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 에포크 결과 요약 출력 (모든 평가 지표)
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'  [Primary] Recall: {val_recall:.2f}% | Precision: {val_precision:.2f}%')
        print(f'  F1-Score: {val_f1:.2f}% | AUC-ROC: {val_auc:.4f}')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # 최고 성능 모델 업데이트 (Recall 기준 - 주 평가 지표)
        if val_recall > best_val_recall:
            print(f'  [BEST] Recall improved from {best_val_recall:.2f}% to {val_recall:.2f}%')
            best_val_recall = val_recall
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # 체크포인트 관리자를 통한 저장 (Recall 기준, 상위 3개만 유지)
        checkpoint_manager.save({
            'epoch': epoch,
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_recall': val_recall,           # 주 평가 지표
            'val_metrics': val_metrics,         # 모든 평가 지표
            'history': history
        }, val_recall, epoch)  # Recall 기준으로 저장
        
        # 조기 종료 체크 (Recall 기준)
        early_stopping(val_recall)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break
    
    # 학습 완료
    training_time = time.time() - start_time
    print('\n' + '='*80)
    print('학습 완료!')
    print(f'총 학습 시간: {training_time / 60:.2f}분')
    print(f'최고 검증 Recall: {best_val_recall:.2f}%')
    print('='*80)
    
    # 최적 가중치로 모델 복원
    model.load_state_dict(best_model_wts)
    
    # 테스트 데이터셋에서 최종 평가
    print('\n테스트 데이터셋 평가 중...')
    test_metrics = test(model, test_loader, device)
    
    # 테스트 결과 출력
    print(f'\n=== 테스트 결과 ===')
    print(f'[주 지표] Recall: {test_metrics["recall"]:.2f}%')
    print(f'Precision: {test_metrics["precision"]:.2f}%')
    print(f'F1-Score: {test_metrics["f1_score"]:.2f}%')
    print(f'Accuracy: {test_metrics["accuracy"]:.2f}%')
    print(f'AUC-ROC: {test_metrics["auc_roc"]:.4f}')
    
    # 최종 체크포인트 저장 (타임스탬프와 성능 포함, 세션 폴더에 저장)
    final_filename = f'{model_name}_{checkpoint_manager.session_timestamp}_final_recall{best_val_recall:.2f}.pth'
    final_checkpoint_path = os.path.join(checkpoint_manager.save_dir, final_filename)
    save_checkpoint({
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'best_val_recall': best_val_recall,
        'test_metrics': test_metrics,  # 모든 테스트 평가 지표
        'history': history,
        'training_time': training_time
    }, final_checkpoint_path)
    
    # 학습 로그를 JSON 형식으로 저장 (시각화용)
    import json
    training_log = {
        'model_name': model_name,
        'session_timestamp': checkpoint_manager.session_timestamp,
        'training_time_minutes': training_time / 60,
        'epochs_trained': len(history['train_loss']),
        'best_val_recall': best_val_recall,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'recall': test_metrics['recall'],
            'precision': test_metrics['precision'],
            'f1_score': test_metrics['f1_score'],
            'auc_roc': test_metrics['auc_roc']
        },
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': config.OPTIMIZER,
            'weight_decay': config.WEIGHT_DECAY,
            'lr_scheduler': config.LR_SCHEDULER_TYPE,
            'warmup_epochs': config.LR_WARMUP_EPOCHS,
            'label_smoothing': config.LABEL_SMOOTHING,
            'gradient_clip': config.GRADIENT_CLIP_NORM
        },
        'history': {
            # 메타정보: 배치 수를 알면 에포크별 데이터 복원 가능
            'batches_per_epoch': len(train_loader),
            # train: 배치 단위 (모든 배치를 flat 리스트로 저장)
            # 에포크 N의 데이터: train[metric][(N-1)*batches_per_epoch : N*batches_per_epoch]
            'train': {
                'loss': [v for epoch_batch in history.get('train_batch', []) for v in epoch_batch['loss']],
                'acc': [v for epoch_batch in history.get('train_batch', []) for v in epoch_batch['accuracy']],
                'recall': [v for epoch_batch in history.get('train_batch', []) for v in epoch_batch['recall']],
                'precision': [v for epoch_batch in history.get('train_batch', []) for v in epoch_batch['precision']],
                'f1': [v for epoch_batch in history.get('train_batch', []) for v in epoch_batch['f1_score']]
            },
            # val: 에포크 단위
            'val': {
                'loss': history['val_loss'],
                'acc': history['val_acc'],
                'recall': history.get('val_recall', []),
                'precision': history.get('val_precision', []),
                'f1': history.get('val_f1', []),
                'auc': history.get('val_auc', [])
            }
        }
    }
    
    log_filename = f'{model_name}_{checkpoint_manager.session_timestamp}_training_log.json'
    log_path = os.path.join(checkpoint_manager.save_dir, log_filename)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    print(f'학습 로그 저장됨: {log_path}')
    
    # TensorBoard writer 종료
    writer.close()
    
    return model, history, test_metrics


def main():
    """
    스크립트의 진입점(Entry Point) 함수
    
    명령줄 인터페이스(CLI)를 통해 학습 파라미터를 받아
    모델 학습을 시작합니다.
    
    사용 예시:
        # VGG16 모델 학습 (기본 설정)
        python train.py --model vgg16
        
        # ResNet50 모델, 커스텀 설정으로 학습
        python train.py --model resnet50 --epochs 50 --batch_size 16 --lr 0.0001
        
        # DenseNet121 모델 학습
        python train.py --model densenet121 --epochs 100
    
    Arguments:
        --model: 학습할 모델 아키텍처 (필수)
                 선택지: vgg16, resnet50, densenet121
        --epochs: 학습 에포크 수 (기본값: config.NUM_EPOCHS)
        --batch_size: 배치 크기 (기본값: config.BATCH_SIZE)
        --lr: 학습률 (기본값: config.LEARNING_RATE)
    """
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(description='Train COVID-19 Classification Model')
    
    # 모델 아키텍처 선택 (기본값: vgg16)
    parser.add_argument('--model', type=str, default='vgg16',
                       choices=['vgg16', 'resnet50', 'densenet121'],
                       help='Model architecture to train (default: vgg16)')
    
    # 선택적 인자: 학습 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs (default: {config.NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=None,
                       help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                       help=f'Learning rate (default: {config.LEARNING_RATE})')
    
    # 명령줄 인자 파싱
    args = parser.parse_args()
    
    # 모델 학습 실행
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


# 스크립트가 직접 실행될 때만 main() 함수 호출
# 다른 모듈에서 import할 때는 실행되지 않음
if __name__ == '__main__':
    main()
