"""
Configuration file for COVID-19 Image Classification Project
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

# output 경로 설정
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TENSORBOARD_DIR = os.path.join(BASE_DIR, 'runs')

# 디렉토리 생성(디랙터리가 없을 경우)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# ========================
# 모델 설정
# ========================
NUM_CLASSES = 3
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']

# 사용할 모델
MODELS = {
    'vgg16': 'VGG16',
    'resnet50': 'ResNet50',
    'densenet121': 'DenseNet121'
}

# ========================
# 학습 설정
# ========================
# 하이퍼파라미터
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
RANDOM_SEED = 42

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-7

# Validation split
VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# ========================
# 데이터 증강
# ========================
IMG_SIZE = 224  # 표준 input size

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation parameters
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.1
COLOR_JITTER_HUE = 0.05

# ========================
# Device Configuration
# ========================
# GPU 설정: 0번 GPU만 사용
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0번 GPU만 사용

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # Number of workers for data loading

# ========================
# Logging Configuration
# ========================
LOG_INTERVAL = 10  # Print training stats every N batches
SAVE_BEST_ONLY = True  # Only save the best model based on validation accuracy

# ========================
# Model-specific settings
# ========================
# Dropout rates
VGG16_DROPOUT = 0.5
RESNET50_DROPOUT = 0.3
DENSENET121_DROPOUT = 0.3

# Feature extraction vs fine-tuning
FREEZE_BACKBONE = False  # Set to True to only train the classifier head
