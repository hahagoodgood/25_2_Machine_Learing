"""
발표용 시각화 자료 생성 스크립트
PRESENTATION_GUIDE.md에 필요한 시각화 자료를 생성합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 저장 디렉토리
SAVE_DIR = os.path.join(config.BASE_DIR, 'presentation_figures')
os.makedirs(SAVE_DIR, exist_ok=True)

# 색상 팔레트
COLORS = {
    'train': '#2ecc71',      # 초록색
    'val': '#3498db',        # 파란색  
    'test': '#e74c3c',       # 빨간색
    'covid': '#e74c3c',      # 빨간색
    'normal': '#2ecc71',     # 초록색
    'viral': '#f39c12',      # 주황색
}


def plot_class_distribution():
    """
    클래스별 샘플 수 막대 그래프 생성
    PRESENTATION_GUIDE.md 2.2 데이터 구성 섹션용
    """
    # 실제 데이터셋 샘플 수 (PRESENTATION_GUIDE.md에서 확인)
    classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
    
    train_counts = [3616, 10192, 1345]
    val_counts = [600, 1696, 224]
    test_counts = [1345, 3780, 500]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 막대 그래프
    bars1 = ax.bar(x - width, train_counts, width, label='Train', color=COLORS['train'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, val_counts, width, label='Validation', color=COLORS['val'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, test_counts, width, label='Test', color=COLORS['test'], edgecolor='white', linewidth=1)
    
    # 막대 위에 값 표시
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # 스타일링
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('COVID-19 Chest X-ray Dataset Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Y축 포맷팅
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # 총 샘플 수 텍스트
    total_train = sum(train_counts)
    total_val = sum(val_counts)
    total_test = sum(test_counts)
    total_all = total_train + total_val + total_test
    
    summary_text = f'Total: {total_all:,} samples\n(Train: {total_train:,} | Val: {total_val:,} | Test: {total_test:,})'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(SAVE_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✅ 클래스별 분포 그래프 저장: {save_path}')
    return save_path


def plot_class_pie_chart():
    """
    클래스별 분포 파이 차트 생성
    PRESENTATION_GUIDE.md 2.5 클래스 불균형 처리 섹션용
    """
    # 전체 데이터 기준
    classes = ['COVID-19', 'Normal', 'Viral Pneumonia']
    totals = [3616 + 600 + 1345, 10192 + 1696 + 3780, 1345 + 224 + 500]  # 5561, 15668, 2069
    colors = [COLORS['covid'], COLORS['normal'], COLORS['viral']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 파이 차트
    ax1 = axes[0]
    explode = (0.02, 0.02, 0.02)
    wedges, texts, autotexts = ax1.pie(
        totals, 
        labels=classes,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 12}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    ax1.set_title('Class Distribution (Total Dataset)', fontsize=14, fontweight='bold', pad=15)
    
    # 오른쪽: 불균형 비율 막대
    ax2 = axes[1]
    y_pos = np.arange(len(classes))
    bars = ax2.barh(y_pos, totals, color=colors, edgecolor='white', linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes, fontsize=12)
    ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Class Imbalance Visualization', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 막대에 값 표시
    for bar, count in zip(bars, totals):
        width = bar.get_width()
        ax2.annotate(f'{count:,}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=11, fontweight='bold')
    
    # 비율 표시
    max_count = max(totals)
    for i, (bar, count) in enumerate(zip(bars, totals)):
        ratio = max_count / count
        ax2.annotate(f'×{ratio:.1f}',
                    xy=(max_count * 0.95, bar.get_y() + bar.get_height()/2),
                    ha='right', va='center',
                    fontsize=10, color='gray', style='italic')
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(SAVE_DIR, 'class_imbalance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✅ 클래스 불균형 차트 저장: {save_path}')
    return save_path


def plot_model_comparison():
    """
    모델별 파라미터 수 비교 막대 그래프
    PRESENTATION_GUIDE.md 3.2 사용 모델 비교 섹션용
    """
    models = ['VGG16', 'ResNet50', 'DenseNet121']
    params = [138, 25.6, 8]  # 단위: Million
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, params, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    # 막대 위에 값 표시
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.annotate(f'{param}M',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Model Architecture', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parameters (Millions)', fontsize=14, fontweight='bold')
    ax.set_title('Model Parameter Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 추가 정보
    info_text = 'VGG16: Deep & Simple\nResNet50: Skip Connection\nDenseNet121: Dense Block'
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(SAVE_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'✅ 모델 비교 그래프 저장: {save_path}')
    return save_path


def main():
    """모든 시각화 자료 생성"""
    print('=' * 60)
    print('발표용 시각화 자료 생성 중...')
    print('=' * 60)
    print(f'저장 경로: {SAVE_DIR}\n')
    
    # 1. 클래스별 분포
    plot_class_distribution()
    
    # 2. 클래스 불균형 차트
    plot_class_pie_chart()
    
    # 3. 모델 비교
    plot_model_comparison()
    
    print('\n' + '=' * 60)
    print('✅ 모든 시각화 자료 생성 완료!')
    print(f'저장 위치: {SAVE_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()
