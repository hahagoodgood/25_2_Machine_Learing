"""
COVID-19 이미지 분류 프로젝트 설정 파일
"""
import os
import torch

# ========================
# 경로 설정
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# 출력 경로 설정
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')

# 디렉토리 생성 (디렉토리가 없을 경우)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# ========================
# 모델 설정
# ========================
NUM_CLASSES = 3  # 분류할 클래스 수
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']  # 클래스 이름

# 사용 가능한 모델
MODELS = {
    'vgg16': 'VGG16',
    'resnet50': 'ResNet50',
    'densenet121': 'DenseNet121'
}

# ========================
# 학습 설정
# ========================
# 하이퍼파라미터
BATCH_SIZE = 32  # 배치 크기
LEARNING_RATE = 0.001  # 학습률
NUM_EPOCHS = 50  # 총 에포크 수
RANDOM_SEED = 42  # 재현성을 위한 랜덤 시드

# 조기 종료 설정
EARLY_STOPPING_PATIENCE = 10  # 성능 개선 없이 기다리는 에포크 수

# 학습률 스케줄러 설정
LR_SCHEDULER_FACTOR = 0.5  # 학습률 감소율
LR_SCHEDULER_PATIENCE = 5  # 학습률 감소 전에 기다리는 에포크 수
LR_SCHEDULER_MIN_LR = 1e-7  # 최소 학습률

# 검증 데이터 분할
VALIDATION_SPLIT = 0.2  # 훈련 데이터의 20%를 검증에 사용

# ========================
# 데이터 증강
# ========================
IMG_SIZE = 224  # 표준 입력 이미지 크기

# ImageNet 정규화 값 (사전학습 모델과 동일한 분포 맞춤)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB 채널별 평균
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB 채널별 표준편차

# 증강 파라미터
ROTATION_DEGREES = 15  # 회전 각도 (-15도 ~ +15도)
COLOR_JITTER_BRIGHTNESS = 0.2  # 밝기 조정 범위
COLOR_JITTER_CONTRAST = 0.2  # 대비 조정 범위
COLOR_JITTER_SATURATION = 0.1  # 채도 조정 범위
COLOR_JITTER_HUE = 0.05  # 색조 조정 범위

# ========================
# 장치 설정
# ========================
# GPU 설정
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 1번 GPU만 사용

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 16  # 데이터 로딩에 사용할 워커 수

# ========================
# 로깅 설정
# ========================
LOG_INTERVAL = 10  # N개 배치마다 학습 통계 출력
SAVE_BEST_ONLY = True  # 검증 정확도 기준 최고 모델만 저장

# ========================
# 모델별 설정
# ========================
# 드롭아웃 비율 (과적합 방지)
VGG16_DROPOUT = 0.5  # VGG16 드롭아웃 비율
RESNET50_DROPOUT = 0.3  # ResNet50 드롭아웃 비율
DENSENET121_DROPOUT = 0.3  # DenseNet121 드롭아웃 비율

# 백본 동결 설정
FREEZE_BACKBONE = False  # True: 분류기 헤드만 학습, False: 전체 네트워크 학습
