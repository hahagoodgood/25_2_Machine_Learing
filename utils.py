"""
COVID-19 이미지 분류 프로젝트를 위한 유틸리티 함수 모음

이 모듈은 다음과 같은 기능을 제공합니다:
- 랜덤 시드 설정 (재현성 보장)
- 장치(GPU/CPU) 감지
- 체크포인트 저장 및 로드
- 혼동 행렬 시각화
- 학습 곡선 시각화
- 분류 메트릭 계산
- ROC 곡선 시각화
"""
import os
import random
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json


def set_seed(seed=42):
    """
    재현성을 위한 랜덤 시드 설정
    
    동일한 시드를 설정하면 매번 동일한 결과를 얻을 수 있습니다.
    Python, NumPy, PyTorch의 모든 랜덤 생성기에 동일한 시드를 적용합니다.
    
    Args:
        seed (int): 랜덤 시드 값 (기본값: 42)
    
    Note:
        - torch.backends.cudnn.deterministic = True: GPU 연산의 결정론적 동작 보장
        - torch.backends.cudnn.benchmark = False: 최적화 비활성화로 재현성 확보
    """
    random.seed(seed)                    # Python 내장 random 모듈
    np.random.seed(seed)                 # NumPy 랜덤
    torch.manual_seed(seed)              # PyTorch CPU 랜덤
    torch.cuda.manual_seed(seed)         # PyTorch GPU 랜덤 (단일 GPU)
    torch.cuda.manual_seed_all(seed)     # PyTorch GPU 랜덤 (다중 GPU)
    torch.backends.cudnn.deterministic = True  # cuDNN 결정론적 모드
    torch.backends.cudnn.benchmark = False     # cuDNN 벤치마크 비활성화


def get_device():
    """
    사용 가능한 장치(GPU 또는 CPU) 감지 및 반환
    
    CUDA가 사용 가능한 경우 GPU를 반환하고, 그렇지 않으면 CPU를 반환합니다.
    GPU 사용 시 장치 이름과 메모리 정보를 출력합니다.
    
    Returns:
        torch.device: 사용할 장치 객체 ('cuda' 또는 'cpu')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'사용 GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)')
    else:
        print('CPU 사용')
    return device


def save_checkpoint(state, filepath):
    """
    모델 체크포인트 저장
    
    학습 중간 상태(모델 가중치, 옵티마이저 상태, 에포크 등)를 파일로 저장합니다.
    학습이 중단되어도 저장된 체크포인트에서 재개할 수 있습니다.
    
    Args:
        state (dict): 저장할 상태 딕셔너리
                     - model_state_dict: 모델 가중치
                     - optimizer_state_dict: 옵티마이저 상태
                     - epoch: 현재 에포크
                     - best_val_acc: 최고 검증 정확도
        filepath (str): 체크포인트를 저장할 파일 경로 (.pth)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f'체크포인트 저장됨: {filepath}')


class CheckpointManager:
    """
    상위 N개의 체크포인트만 유지하는 체크포인트 관리 클래스
    
    검증 Recall 기준으로 가장 성능이 좋은 N개의 체크포인트만 저장하고,
    N개를 초과하면 가장 성능이 낮은 체크포인트를 자동으로 삭제합니다.
    
    Attributes:
        save_dir (str): 체크포인트 저장 디렉토리
        model_name (str): 모델 이름 (파일명에 사용)
        max_checkpoints (int): 유지할 최대 체크포인트 수
        checkpoints (list): 저장된 체크포인트 정보 리스트 [(Recall, 경로), ...]
    """
    
    def __init__(self, save_dir, model_name, max_checkpoints=3):
        """
        CheckpointManager 초기화
        
        Args:
            save_dir (str): 체크포인트를 저장할 기본 디렉토리 경로
            model_name (str): 모델 이름 (예: 'vgg16', 'resnet50')
            max_checkpoints (int): 유지할 최대 체크포인트 수 (기본값: 3)
        """
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []  # (val_recall, filepath) 튜플 리스트
        
        # 세션별 폴더 생성 (날짜_시간 형식)
        # 예: checkpoints/session_20231207_235126/
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = f'session_{self.session_timestamp}'
        self.save_dir = os.path.join(save_dir, session_folder)
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'체크포인트 저장 디렉토리: {self.save_dir}')
    
    def save(self, state, val_recall, epoch):
        """
        체크포인트 저장 및 관리
        
        새 체크포인트를 저장하고, max_checkpoints를 초과하면
        가장 성능이 낮은 체크포인트를 삭제합니다.
        
        Args:
            state (dict): 저장할 상태 딕셔너리 (모델, 옵티마이저 등)
            val_recall (float): 현재 검증 Recall (%)
            epoch (int): 현재 에포크
            
        Returns:
            bool: 저장 여부 (상위 N개에 포함되면 True)
        """
        # 저장 대상인지 확인 (현재 체크포인트가 N개 미만이거나, 최저 성능보다 좋은 경우)
        should_save = len(self.checkpoints) < self.max_checkpoints
        
        if not should_save and self.checkpoints:
            # 현재 저장된 체크포인트 중 최저 성능과 비교
            min_recall = min(self.checkpoints, key=lambda x: x[0])[0]
            should_save = val_recall > min_recall
        
        if should_save:
            # 파일 경로 생성 (세션 타임스탬프, 에포크, Recall을 파일명에 포함)
            filename = f'{self.model_name}_{self.session_timestamp}_epoch{epoch:03d}_recall{val_recall:.2f}.pth'
            filepath = os.path.join(self.save_dir, filename)
            
            # 체크포인트 저장
            torch.save(state, filepath)
            self.checkpoints.append((val_recall, filepath))
            print(f'체크포인트 저장됨: {filepath}')
            
            # 최대 개수 초과 시 가장 낮은 성능의 체크포인트 삭제
            if len(self.checkpoints) > self.max_checkpoints:
                self._remove_worst_checkpoint()
            
            return True
        
        return False
    
    def _remove_worst_checkpoint(self):
        """
        가장 성능이 낮은 체크포인트 삭제 (내부 메서드)
        """
        if not self.checkpoints:
            return
        
        # 가장 낮은 Recall의 체크포인트 찾기
        worst_idx = min(range(len(self.checkpoints)), 
                       key=lambda i: self.checkpoints[i][0])
        worst_recall, worst_path = self.checkpoints[worst_idx]
        
        # 파일 삭제
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f'체크포인트 삭제됨 (Recall {worst_recall:.2f}%): {worst_path}')
        
        # 리스트에서 제거
        self.checkpoints.pop(worst_idx)
    
    def get_best_checkpoint(self):
        """
        가장 성능이 좋은 체크포인트 경로 반환
        
        Returns:
            str or None: 최고 성능 체크포인트 경로, 없으면 None
        """
        if not self.checkpoints:
            return None
        
        best_recall, best_path = max(self.checkpoints, key=lambda x: x[0])
        return best_path
    
    def get_all_checkpoints(self):
        """
        모든 저장된 체크포인트 정보 반환 (Recall 내림차순)
        
        Returns:
            list: [(Recall, 경로), ...] 형태의 리스트
        """
        return sorted(self.checkpoints, key=lambda x: x[0], reverse=True)


def load_checkpoint(filepath, model, optimizer=None):
    """
    저장된 모델 체크포인트 로드
    
    학습된 모델 가중치를 로드하여 추론 또는 학습 재개에 사용합니다.
    
    Args:
        filepath (str): 체크포인트 파일 경로
        model (nn.Module): 가중치를 로드할 모델
        optimizer (Optimizer, optional): 상태를 로드할 옵티마이저
                                        학습 재개 시에만 필요
        
    Returns:
        dict: 체크포인트 정보 (에포크, 최고 정확도 등)
    
    Raises:
        FileNotFoundError: 체크포인트 파일이 존재하지 않을 경우
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'체크포인트를 찾을 수 없습니다: {filepath}')
    
    # CPU에서 로드 후 필요 시 GPU로 이동 (메모리 효율)
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드 (학습 재개 시)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f'체크포인트 로드됨: {filepath}')
    return checkpoint


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='혼동 행렬'):
    """
    혼동 행렬(Confusion Matrix) 시각화
    
    모델의 분류 성능을 시각적으로 분석합니다.
    대각선 요소는 정확한 예측, 비대각선 요소는 오분류를 나타냅니다.
    
    Args:
        y_true (array-like): 실제 레이블
        y_pred (array-like): 예측 레이블
        class_names (list): 클래스 이름 리스트 (예: ['Covid', 'Normal', 'Viral Pneumonia'])
        save_path (str, optional): 그래프를 저장할 파일 경로
        title (str): 그래프 제목
    
    Returns:
        numpy.ndarray: 혼동 행렬 (n_classes x n_classes)
    """
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 히트맵 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '샘플 수'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('실제 레이블', fontsize=12)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.tight_layout()
    
    # 파일 저장 (경로가 지정된 경우)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'혼동 행렬 저장됨: {save_path}')
    
    plt.show()
    return cm


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    학습 및 검증 손실/정확도 곡선 시각화
    
    에포크별 학습 진행 상황을 시각적으로 분석합니다.
    과적합 여부를 판단하는 데 유용합니다.
    
    Args:
        train_losses (list): 에포크별 학습 손실 값
        val_losses (list): 에포크별 검증 손실 값
        train_accs (list): 에포크별 학습 정확도
        val_accs (list): 에포크별 검증 정확도
        save_path (str, optional): 그래프를 저장할 파일 경로
    
    Note:
        - 학습 손실 < 검증 손실: 정상
        - 학습 손실 << 검증 손실: 과적합 의심
        - 두 손실 모두 높음: 과소적합 의심
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 손실 그래프
    ax1.plot(epochs, train_losses, 'b-', label='학습 손실', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='검증 손실', linewidth=2)
    ax1.set_title('학습 및 검증 손실', fontsize=14, fontweight='bold')
    ax1.set_xlabel('에포크', fontsize=12)
    ax1.set_ylabel('손실', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 정확도 그래프
    ax2.plot(epochs, train_accs, 'b-', label='학습 정확도', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='검증 정확도', linewidth=2)
    ax2.set_title('학습 및 검증 정확도', fontsize=14, fontweight='bold')
    ax2.set_xlabel('에포크', fontsize=12)
    ax2.set_ylabel('정확도 (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 파일 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'학습 곡선 저장됨: {save_path}')
    
    plt.show()


def calculate_metrics(y_true, y_pred, class_names):
    """
    포괄적인 분류 메트릭 계산
    
    정밀도, 재현율, F1 점수 등 다양한 분류 성능 지표를 계산합니다.
    
    Args:
        y_true (array-like): 실제 레이블
        y_pred (array-like): 예측 레이블
        class_names (list): 클래스 이름 리스트
        
    Returns:
        dict: 분류 메트릭 딕셔너리
            - classification_report: sklearn 분류 보고서
            - confusion_matrix: 혼동 행렬
            - overall_accuracy: 전체 정확도
            - per_class_metrics: 클래스별 메트릭
    
    Note:
        - Precision (정밀도): 양성 예측 중 실제 양성 비율
        - Recall (재현율): 실제 양성 중 양성으로 예측된 비율
        - F1-score: 정밀도와 재현율의 조화 평균
    """
    # sklearn의 분류 보고서 생성
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # 메트릭 딕셔너리 구성
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'overall_accuracy': report['accuracy'],
        'per_class_metrics': {}
    }
    
    # 클래스별 메트릭 추출
    for i, class_name in enumerate(class_names):
        metrics['per_class_metrics'][class_name] = {
            'precision': report[class_name]['precision'],   # 정밀도
            'recall': report[class_name]['recall'],         # 재현율
            'f1-score': report[class_name]['f1-score'],     # F1 점수
            'support': report[class_name]['support']        # 샘플 수
        }
    
    return metrics


def save_metrics(metrics, filepath):
    """
    분류 메트릭을 JSON 파일로 저장
    
    Args:
        metrics (dict): 메트릭 딕셔너리 (calculate_metrics 함수의 반환값)
        filepath (str): 저장할 파일 경로 (.json)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f'메트릭 저장됨: {filepath}')


def plot_roc_curves(y_true, y_probs, class_names, save_path=None):
    """
    다중 클래스 분류를 위한 ROC 곡선 시각화
    
    ROC(Receiver Operating Characteristic) 곡선은 분류기의 성능을 시각화합니다.
    AUC(Area Under Curve)가 1에 가까울수록 좋은 성능을 의미합니다.
    
    Args:
        y_true (array-like): 실제 레이블
        y_probs (array-like): 예측 확률 (n_samples, n_classes)
                             softmax 출력값
        class_names (list): 클래스 이름 리스트
        save_path (str, optional): 그래프를 저장할 파일 경로
    
    Note:
        - AUC = 0.5: 랜덤 분류기 수준
        - AUC > 0.8: 좋은 성능
        - AUC > 0.9: 우수한 성능
    """
    n_classes = len(class_names)
    
    # 레이블을 이진화 (One-Hot 인코딩)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # 각 클래스별 색상 정의
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 클래스별 ROC 곡선 그리기
    for i, (class_name, color) in enumerate(zip(class_names, colors[:n_classes])):
        # FPR (False Positive Rate), TPR (True Positive Rate) 계산
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)  # AUC 계산
        
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # 랜덤 분류기 기준선 (대각선)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='랜덤 분류기')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('위양성률 (False Positive Rate)', fontsize=12)
    plt.ylabel('진양성률 (True Positive Rate)', fontsize=12)
    plt.title('ROC 곡선 - 다중 클래스 분류', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 파일 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'ROC 곡선 저장됨: {save_path}')
    
    plt.show()


def count_parameters(model):
    """
    모델의 파라미터 수 계산 및 출력
    
    전체 파라미터 수와 학습 가능한 파라미터 수를 계산합니다.
    모델 복잡도를 파악하는 데 유용합니다.
    
    Args:
        model (nn.Module): PyTorch 모델
        
    Returns:
        tuple: (전체 파라미터 수, 학습 가능한 파라미터 수)
    
    Note:
        - 전체 파라미터: 모델의 모든 가중치 수
        - 학습 가능 파라미터: requires_grad=True인 가중치 수
        - 고정 파라미터: 백본 동결 시 학습되지 않는 가중치 수
    """
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    # 학습 가능한 파라미터 수 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'전체 파라미터: {total_params:,}')
    print(f'학습 가능 파라미터: {trainable_params:,}')
    print(f'고정 파라미터: {total_params - trainable_params:,}')
    
    return total_params, trainable_params
