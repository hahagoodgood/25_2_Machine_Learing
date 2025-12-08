# COVID-19 Image Classification - Multi-Model Comparison

이 프로젝트는 PyTorch를 사용하여 COVID-19 흉부 X-ray 이미지를 분류하는 딥러닝 모델을 구축하고, VGG16, ResNet50, DenseNet121 세 가지 모델의 성능을 비교합니다.

## 📊 Dataset

**출처**: [COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data?select=Covid19-dataset)

**구조**:
```
dataset/
├── train/
│   ├── Covid/ (111 images)
│   ├── Normal/ (70 images)
│   └── Viral Pneumonia/ (70 images)
└── test/
    ├── Covid/ (26 images)
    ├── Normal/ (20 images)
    └── Viral Pneumonia/ (20 images)
```

**클래스**: Covid, Normal, Viral Pneumonia (총 3개)  
**총 이미지 수**: 훈련 251장, 테스트 66장

## 🚀 Features

- ✅ **3가지 사전학습 모델**: VGG16, ResNet50, DenseNet121 (ImageNet weights)
- ✅ **의료 특화 평가 지표**: **Recall(재현율)을 주 지표로 사용** (실제 환자를 놓치지 않기 위함), Precision, F1-Score, AUC-ROC
- ✅ **데이터 증강**: 작은 데이터셋을 위한 적극적인 augmentation 전략
- ✅ **클래스 불균형 처리**: 가중치 기반 손실 함수 & 레이블 스무딩
- ✅ **Early Stopping**: 검증 Recall 기준 과적합 방지
- ✅ **학습률 스케줄링**: ReduceLROnPlateau, CosineAnnealingLR 등 지원
- ✅ **TensorBoard 로깅**: 실시간 학습 모니터링
- ✅ **모델 체크포인팅**: 최고 성능(Recall) 모델 자동 저장 (세션별 관리)
- ✅ **Jupyter Notebooks**: 대화형 평가 및 비교 분석

## 📁 Project Structure

```
25_2_Machine_Learing/
├── dataset/                    # 데이터셋 디렉토리
│   ├── train/
│   └── test/
├── models/                     # 모델 아키텍처 파일
│   ├── __init__.py
│   ├── vgg16_model.py
│   ├── resnet50_model.py
│   └── densenet121_model.py
├── checkpoints/                # 학습된 모델 체크포인트 (세션별 자동 생성)
├── results/                    # 평가 결과 및 시각화
├── runs/                       # TensorBoard 로그
├── config.py                   # 설정 파일
├── dataset.py                  # 데이터 로더 및 증강
├── utils.py                    # 유틸리티 함수
├── train.py                    # 학습 스크립트
├── evaluate.ipynb              # 평가 노트북
├── compare_models.ipynb        # 모델 비교 노트북
├── requirements.txt            # 의존성 패키지
└── README.md                   # 이 파일
```

## 🛠️ Installation

### 1. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

### 주요 의존성:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, matplotlib, seaborn
- scikit-learn
- tensorboard
- jupyter

## 📖 Usage

### 1. 모델 학습

각 모델을 개별적으로 학습시킬 수 있습니다:

```bash
# VGG16 학습
python train.py --model vgg16 --epochs 50 --batch_size 32

# ResNet50 학습
python train.py --model resnet50 --epochs 50 --batch_size 32

# DenseNet121 학습
python train.py --model densenet121 --epochs 50 --batch_size 32
```

**학습 옵션**:
- `--model`: 모델 선택 (vgg16, resnet50, densenet121)
- `--epochs`: 학습 에포크 수 (기본값: 50)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--lr`: 학습률 (기본값: 0.001)

**학습 과정**:
- 훈련 데이터의 20%를 검증 세트로 자동 분할
- Early stopping으로 과적합 방지 (Recall 기준)
- 최고 검증 **Recall** 모델을 `checkpoints/session_{timestamp}/` 디렉토리에 저장
- TensorBoard 로그는 `runs/` 디렉토리에 저장

### 2. TensorBoard 모니터링

학습 중 실시간으로 loss와 accuracy를 모니터링:

```bash
tensorboard --logdir=runs
```

브라우저에서 `http://localhost:6006` 으로 접속

### 3. 모델 평가

`evaluate.ipynb` 노트북을 사용하여 학습된 모델을 평가:

```bash
jupyter notebook evaluate.ipynb
```

**평가 내용**:
- 테스트 세트에서 정확도 측정
- 클래스별 precision, recall, F1-score
- Confusion matrix 시각화
- ROC curves 및 AUC 점수
- 결과를 JSON 및 CSV로 저장

### 4. 모델 비교

`compare_models.ipynb` 노트북을 사용하여 세 모델의 성능을 비교:

```bash
jupyter notebook compare_models.ipynb
```

**비교 내용**:
- 학습/검증 loss 및 accuracy 곡선
- 테스트 정확도 막대 그래프
- 클래스별 성능 비교
- F1-score 히트맵
- 학습 효율성 분석

## ⚙️ Configuration

`config.py` 파일에서 주요 설정을 수정할 수 있습니다:

```python
# 하이퍼파라미터
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Early stopping
EARLY_STOPPING_PATIENCE = 15

# 데이터 증강
IMG_SIZE = 224
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.2
```

## 🎯 Data Augmentation

작은 데이터셋의 한계를 극복하기 위해 다음과 같은 증강 전략을 사용:

**훈련 시**:
- RandomResizedCrop (224x224)
- RandomHorizontalFlip
- RandomRotation (±15도)
- RandomAffine (이동 및 스케일)
- ColorJitter (밝기, 대비, 채도, 색조 조정)
- ImageNet 정규화

**검증/테스트 시**:
- Resize (224x224)
- ImageNet 정규화만 적용

## 📈 Expected Results

훈련 후 다음과 같은 파일들이 생성됩니다:

```
checkpoints/
└── session_202403XX_XXXXXX/
    ├── vgg16_202403XX_XXXXXX_epoch012_recall95.50.pth
    └── vgg16_202403XX_XXXXXX_final_recall96.00.pth

results/
├── model_summary.csv
├── final_comparison.csv
├── *_metrics.json
├── confusion_matrices.png
├── *_roc_curves.png
├── per_class_metrics.png
├── training_curves_comparison.png
├── test_accuracy_comparison.png
└── f1_score_heatmap.png

runs/
├── vgg16/
├── resnet50/
└── densenet121/
```

## 🔬 Model Architecture

### VGG16
- 사전학습된 VGG16 backbone
- 커스텀 classifier head (4096 → 4096 → 3)
- Dropout 적용

### ResNet50
- 사전학습된 ResNet50 backbone
- 커스텀 classifier (2048 → 512 → 3)
- Dropout 적용

### DenseNet121
- 사전학습된 DenseNet121 backbone
- 커스텀 classifier (1024 → 512 → 3)
- Dropout 적용
- 작은 데이터셋에 효율적

## 🎓 Training Tips

1. **GPU 사용 권장**: CUDA 지원 GPU가 있으면 자동으로 사용됩니다
2. **배치 크기 조정**: GPU 메모리에 따라 batch_size 조정
3. **Primary Metric**: 의료 영상 진단 특성상 위음성(False Negative)을 줄이는 것이 중요하므로, **Recall(재현율)** 변화에 주목하세요.
4. **Early stopping**: 검증 Recall이 개선되지 않으면 학습이 조기 종료될 수 있습니다.
5. **재현성**: 모든 실험은 random seed=42로 고정

## 📝 Notes

- **클래스 불균형**: 가중치 기반 CrossEntropyLoss로 처리
- **레이블 스무딩(Label Smoothing)**: 0.01 적용으로 과적합 방지
- **소규모 데이터셋**: 적극적인 데이터 증강과 transfer learning 활용
- **전이 학습**: ImageNet 사전학습 가중치 사용
- **Fine-tuning**: 전체 네트워크 학습 (FREEZE_BACKBONE=False)

## 🤝 Contributing

이 프로젝트는 교육 목적으로 만들어졌습니다. 개선 사항이나 버그가 있다면 이슈를 등록해주세요.

## 📄 License

This project is for educational purposes.

## 👤 Author

Machine Learning Course Project - 2025

---

**Happy Training! 🚀**
