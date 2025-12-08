"""
시각화 모듈 (Visualization Module)

학습 결과를 시각화하는 플롯 함수들을 제공합니다.

주요 기능:
- Loss/Accuracy 곡선 플롯
- 배치별 Loss/Recall 곡선 플롯
- AUC-ROC 곡선 플롯
- Metrics Grid (2x2 서브플롯)
- 혼동 행렬 (Confusion Matrix)
- 클래스별 성능 바 차트
- 종합 대시보드

사용법:
    from visualization import generate_all_plots
    generate_all_plots(history, test_metrics, cm, ...)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# ========================================
# 공통 설정
# ========================================

# 색상 팔레트 정의
COLORS = {
    'train': '#3498db',      # 파랑
    'val': '#e74c3c',        # 빨강
    'best': '#f39c12',       # 주황
    'recall': '#9b59b6',     # 보라
    'precision': '#1abc9c',  # 청록
    'f1': '#34495e',         # 진회색
    'test': '#27ae60',       # 녹색
}


def setup_plot_style():
    """플롯 스타일 설정"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10


def get_epoch_step(num_epochs):
    """에포크 수에 따라 적절한 눈금 간격 계산"""
    if num_epochs <= 20:
        return 1
    elif num_epochs <= 50:
        return 2 # 20~50: 2단위
    elif num_epochs <= 100:
        return 5 # 50~100: 5단위
    elif num_epochs <= 200:
        return 10 # 100~200: 10단위
    elif num_epochs <= 500:
        return 20 # 200~500: 20단위
    else:
        return 50 # 500~: 50단위


def set_epoch_ticks(ax, num_epochs):
    """x축 눈금을 정수로 설정하고 간격 조정"""
    step = get_epoch_step(num_epochs)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))



# ========================================
# 개별 플롯 함수들
# ========================================

def plot_loss_curve(history, best_epoch, model_name, save_path):
    """Loss 곡선 (Train: 실선, Val: 점선 + 마커)"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, history['train_loss'], '-', color=COLORS['train'], 
             label='Train Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], '--', color=COLORS['val'], 
             label='Val Loss', linewidth=2, marker='o', markersize=4)
    plt.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', 
                label=f'Best Epoch ({best_epoch})', linewidth=2)
    plt.scatter([best_epoch], [history['val_loss'][best_epoch-1]], 
                color=COLORS['best'], s=100, zorder=5, marker='o')
    plt.fill_between(epochs_range, history['train_loss'], history['val_loss'], 
                     alpha=0.1, color='gray')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # x축 눈금 간격 조정
    set_epoch_ticks(plt.gca(), len(epochs_range))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curve(history, best_epoch, model_name, save_path):
    """Accuracy 곡선 (Train: 실선, Val: 점선 + 마커)"""
    epochs_range = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, history['train_acc'], '-', color=COLORS['train'], 
             label='Train Acc', linewidth=2)
    plt.plot(epochs_range, history['val_acc'], '--', color=COLORS['val'], 
             label='Val Acc', linewidth=2, marker='o', markersize=4)
    plt.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', 
                label=f'Best Epoch ({best_epoch})', linewidth=2)
    plt.scatter([best_epoch], [history['val_acc'][best_epoch-1]], 
                color=COLORS['best'], s=100, zorder=5, marker='o')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{model_name} - Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # x축 눈금 간격 조정
    set_epoch_ticks(plt.gca(), len(epochs_range))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_batch_loss(history, batches_per_epoch, model_name, save_path):
    """배치별 Loss 곡선 (Train: Batch, Val: Epoch)"""
    if not history.get('train_batch'):
        return
    
    batch_loss = [v for eb in history['train_batch'] for v in eb['loss']]
    batch_steps = range(1, len(batch_loss) + 1)
    num_epochs = len(history['train_loss'])
    epoch_ticks = [e * batches_per_epoch for e in range(1, num_epochs + 1)]
    
    plt.figure(figsize=(14, 6))
    plt.plot(batch_steps, batch_loss, '-', color=COLORS['train'], 
             alpha=0.5, linewidth=1, label='Train (Batch)')
    
    # Val Loss 에포크별 표시
    val_losses = history['val_loss']
    plt.scatter(epoch_ticks, val_losses, color=COLORS['val'], s=50, zorder=5, marker='o')
    plt.plot(epoch_ticks, val_losses, '--', color=COLORS['val'], 
             linewidth=2, label='Val (Epoch)', marker='o', markersize=4)
    
    # 에포크 구분선 및 X축 눈금
    step = get_epoch_step(num_epochs)
    
    shown_epoch_ticks = [e * batches_per_epoch for e in range(step, num_epochs + 1, step)]
    shown_labels = [str(e) for e in range(step, num_epochs + 1, step)]
    
    for e_tick in shown_epoch_ticks:
        plt.axvline(x=e_tick, color='gray', linestyle=':', alpha=0.3)
        
    plt.xticks(shown_epoch_ticks, shown_labels)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Loss (Train: Batch, Val: Epoch)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_batch_recall(history, test_metrics, best_epoch, batches_per_epoch, model_name, save_path):
    """배치별 Recall 곡선 (Train: Batch, Val: Epoch, Test: 수평선)"""
    if not history.get('train_batch'):
        return
    
    batch_recall = [v for eb in history['train_batch'] for v in eb['recall']]
    batch_steps = range(1, len(batch_recall) + 1)
    num_epochs = len(history['train_loss'])
    epoch_ticks = [e * batches_per_epoch for e in range(1, num_epochs + 1)]
    
    plt.figure(figsize=(14, 6))
    plt.plot(batch_steps, batch_recall, '-', color=COLORS['train'], 
             alpha=0.5, linewidth=1, label='Train (Batch)')
    
    # Val Recall 에포크별 표시
    val_recalls = history.get('val_recall', [])
    if val_recalls:
        plt.scatter(epoch_ticks, val_recalls, color=COLORS['val'], s=50, zorder=5, marker='o')
        plt.plot(epoch_ticks, val_recalls, '--', color=COLORS['val'], 
                 linewidth=2, label='Val (Epoch)', marker='o', markersize=4)
    
    # Test Recall 수평선
    plt.axhline(y=test_metrics['recall'], color=COLORS['test'], linestyle='-.', 
               linewidth=2, label=f"Test(Ep{best_epoch}): {test_metrics['recall']:.1f}%")
    
    # 에포크 구분선 및 X축 눈금
    step = get_epoch_step(num_epochs)
    
    shown_epoch_ticks = [e * batches_per_epoch for e in range(step, num_epochs + 1, step)]
    shown_labels = [str(e) for e in range(step, num_epochs + 1, step)]
    
    for e_tick in shown_epoch_ticks:
        plt.axvline(x=e_tick, color='gray', linestyle=':', alpha=0.3)
        
    plt.xticks(shown_epoch_ticks, shown_labels)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall (%)', fontsize=12)
    plt.title(f'{model_name} - Training Recall (Train: Batch, Val: Epoch)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_auc_curve(history, best_epoch, model_name, save_path):
    """AUC-ROC 변화 곡선"""
    if not history.get('val_auc'):
        return
    
    epochs_range = range(1, len(history['val_auc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['val_auc'], '-', color=COLORS['val'], 
             linewidth=2, marker='o', markersize=4, label='Val AUC-ROC')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)', alpha=0.7)
    plt.axhline(y=0.8, color='green', linestyle=':', label='Good (0.8)', alpha=0.5)
    plt.axhline(y=0.9, color='blue', linestyle=':', label='Excellent (0.9)', alpha=0.5)
    plt.fill_between(epochs_range, 0.5, history['val_auc'], alpha=0.2, color=COLORS['val'])
    plt.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.ylim(0.4, 1.0)
    plt.title(f'{model_name} - AUC-ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # x축 눈금 간격 조정
    set_epoch_ticks(plt.gca(), len(epochs_range))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics_grid(history, test_metrics, best_epoch, model_name, save_path):
    """Metrics Grid (2x2 서브플롯: Recall, Precision, F1, All)"""
    if not history.get('val_recall'):
        return
    
    epochs_range = range(1, len(history['val_recall']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 공통 그리드 설정 함수
    def setup_subplot(ax, title, train_key, val_key, test_key):
        if history.get(train_key):
            ax.plot(epochs_range, history[train_key], '-', 
                   color=COLORS['train'], linewidth=2, label='Train')
        ax.plot(epochs_range, history[val_key], '--', 
               color=COLORS['val'], linewidth=2, label='Val', marker='o', markersize=3)
        ax.axhline(y=test_metrics[test_key], color=COLORS['test'], linestyle='-.', 
                  linewidth=1.5, label=f"Test(Ep{best_epoch}): {test_metrics[test_key]:.1f}%")
        ax.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # x축 눈금 간격 조정
        set_epoch_ticks(ax, len(epochs_range))
    
    # 6-1. Recall
    setup_subplot(axes[0, 0], 'Recall (%)', 'train_recall', 'val_recall', 'recall')
    
    # 6-2. Precision
    setup_subplot(axes[0, 1], 'Precision (%)', 'train_precision', 'val_precision', 'precision')
    
    # 6-3. F1-Score
    setup_subplot(axes[1, 0], 'F1-Score (%)', 'train_f1', 'val_f1', 'f1_score')
    
    # 6-4. 모든 Val 지표 비교 (Test 수평선 없음)
    axes[1, 1].plot(epochs_range, history['val_recall'], '--', 
                   color=COLORS['recall'], linewidth=2, label='Recall', marker='o', markersize=3)
    axes[1, 1].plot(epochs_range, history['val_precision'], '--', 
                   color=COLORS['precision'], linewidth=2, label='Precision', marker='s', markersize=3)
    axes[1, 1].plot(epochs_range, history['val_f1'], '--', 
                   color=COLORS['f1'], linewidth=2, label='F1-Score', marker='^', markersize=3)
    axes[1, 1].axvline(x=best_epoch, color=COLORS['best'], linestyle=':', alpha=0.7)
    axes[1, 1].set_title('All Validation Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    set_epoch_ticks(axes[1, 1], len(epochs_range))
    
    plt.suptitle(f'{model_name} - Metrics Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """혼동 행렬 (개선: 비율 + 카운트)"""
    plt.figure(figsize=(10, 8))
    
    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 어노테이션 생성 (개수 + 비율)
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'{model_name} - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class(per_class_report, class_names, model_name, save_path):
    """클래스별 성능 바 차트"""
    recall_vals = [per_class_report[c]['recall'] * 100 for c in class_names]
    precision_vals = [per_class_report[c]['precision'] * 100 for c in class_names]
    f1_vals = [per_class_report[c]['f1-score'] * 100 for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width, recall_vals, width, label='Recall', color=COLORS['recall'])
    bars2 = plt.bar(x, precision_vals, width, label='Precision', color=COLORS['precision'])
    bars3 = plt.bar(x + width, f1_vals, width, label='F1-Score', color=COLORS['f1'])
    
    # 값 표시
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title(f'{model_name} - Per-Class Performance (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(x, class_names)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_dashboard(history, test_metrics, cm, per_class_report, class_names,
                   best_epoch, stop_epoch, training_time, best_val_recall,
                   model_name, save_path):
    """종합 대시보드 (3x2 레이아웃)"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 9-1. Loss (좌상단)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(epochs_range, history['train_loss'], '-', color=COLORS['train'], linewidth=2, label='Train')
    ax1.plot(epochs_range, history['val_loss'], '--', color=COLORS['val'], linewidth=2, 
             label='Val', marker='o', markersize=3)
    ax1.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', alpha=0.7)
    ax1.set_title('Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    set_epoch_ticks(ax1, len(epochs_range))
    
    # 9-2. Accuracy (중앙상단)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs_range, history['train_acc'], '-', color=COLORS['train'], linewidth=2, label='Train')
    ax2.plot(epochs_range, history['val_acc'], '--', color=COLORS['val'], linewidth=2, 
             label='Val', marker='o', markersize=3)
    ax2.axhline(y=test_metrics['accuracy'], color=COLORS['test'], linestyle='-.', 
               linewidth=2, label=f"Test(Ep{best_epoch}): {test_metrics['accuracy']:.1f}%")
    ax2.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', alpha=0.7, label=f'Best: Ep{best_epoch}')
    ax2.set_title('Accuracy (%)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3)
    set_epoch_ticks(ax2, len(epochs_range))
    
    # 9-3. Val Metrics (우상단)
    ax3 = fig.add_subplot(2, 3, 3)
    if history.get('val_recall'):
        ax3.plot(epochs_range, history['val_recall'], '--', color=COLORS['recall'], 
                linewidth=2, label='Val Recall', marker='o', markersize=3)
        ax3.plot(epochs_range, history['val_precision'], '--', color=COLORS['precision'], 
                linewidth=2, label='Val Precision', marker='s', markersize=3)
        ax3.plot(epochs_range, history['val_f1'], '--', color=COLORS['f1'], 
                linewidth=2, label='Val F1', marker='^', markersize=3)
    ax3.axhline(y=test_metrics['recall'], color=COLORS['recall'], linestyle='-.', 
               linewidth=1.5, alpha=0.7, label=f"Test R: {test_metrics['recall']:.1f}%")
    ax3.axhline(y=test_metrics['precision'], color=COLORS['precision'], linestyle='-.', 
               linewidth=1.5, alpha=0.7, label=f"Test P: {test_metrics['precision']:.1f}%")
    ax3.axhline(y=test_metrics['f1_score'], color=COLORS['f1'], linestyle='-.', 
               linewidth=1.5, alpha=0.7, label=f"Test F1: {test_metrics['f1_score']:.1f}%")
    ax3.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', alpha=0.7)
    ax3.set_title(f'Val Metrics (Best: Ep{best_epoch})', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.legend(fontsize=6, loc='lower right', ncol=2)
    ax3.grid(True, alpha=0.3)
    set_epoch_ticks(ax3, len(epochs_range))
    
    # 9-4. Confusion Matrix (좌하단)
    ax4 = fig.add_subplot(2, 3, 4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    ax4.set_title('Confusion Matrix', fontweight='bold')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    
    # 9-5. Per-Class Bar (중앙하단)
    ax5 = fig.add_subplot(2, 3, 5)
    recall_vals = [per_class_report[c]['recall'] * 100 for c in class_names]
    precision_vals = [per_class_report[c]['precision'] * 100 for c in class_names]
    f1_vals = [per_class_report[c]['f1-score'] * 100 for c in class_names]
    x = np.arange(len(class_names))
    ax5.bar(x - 0.2, recall_vals, 0.2, label='Recall', color=COLORS['recall'])
    ax5.bar(x, precision_vals, 0.2, label='Precision', color=COLORS['precision'])
    ax5.bar(x + 0.2, f1_vals, 0.2, label='F1', color=COLORS['f1'])
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names, fontsize=8)
    ax5.set_title('Per-Class Performance', fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 9-6. Summary Text (우하단)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
═══════════════════════════
     TEST RESULTS
═══════════════════════════
  Accuracy:   {test_metrics['accuracy']:.2f}%
  Recall:     {test_metrics['recall']:.2f}%
  Precision:  {test_metrics['precision']:.2f}%
  F1-Score:   {test_metrics['f1_score']:.2f}%
  AUC-ROC:    {test_metrics['auc_roc']:.4f}

═══════════════════════════
     TRAINING INFO
═══════════════════════════
  Best Epoch:      {best_epoch}
  Stop Epoch:      {stop_epoch}
  Training Time:   {training_time/60:.2f} min
  Best Val Recall: {best_val_recall:.2f}%
  
  Inference: {test_metrics.get('inference_time_per_sample_ms', 0):.1f} ms/sample
"""
    ax6.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Training Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========================================
# 통합 함수
# ========================================

def generate_all_plots(history, test_metrics, cm, per_class_report, class_names,
                       model_name, best_epoch, stop_epoch, training_time,
                       best_val_recall, plot_dir, timestamp, batches_per_epoch):
    """
    모든 시각화 플롯 생성
    
    Args:
        history: 학습 히스토리 딕셔너리
        test_metrics: 테스트 결과 딕셔너리
        cm: 혼동 행렬 (numpy array)
        per_class_report: sklearn classification_report 결과
        class_names: 클래스 이름 리스트
        model_name: 모델 이름
        best_epoch: 최고 성능 에포크
        stop_epoch: 종료 에포크
        training_time: 학습 시간 (초)
        best_val_recall: 최고 검증 Recall
        plot_dir: 플롯 저장 디렉토리
        timestamp: 타임스탬프 문자열
        batches_per_epoch: 에포크당 배치 수
    """
    setup_plot_style()
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Loss 곡선
    plot_loss_curve(history, best_epoch, model_name,
                   os.path.join(plot_dir, f'{model_name}_{timestamp}_loss_curve.png'))
    
    # 2. Accuracy 곡선
    plot_accuracy_curve(history, best_epoch, model_name,
                       os.path.join(plot_dir, f'{model_name}_{timestamp}_accuracy_curve.png'))
    
    # 3. 배치별 Loss 곡선
    plot_batch_loss(history, batches_per_epoch, model_name,
                   os.path.join(plot_dir, f'{model_name}_{timestamp}_batch_loss.png'))
    
    # 4. 배치별 Recall 곡선
    plot_batch_recall(history, test_metrics, best_epoch, batches_per_epoch, model_name,
                     os.path.join(plot_dir, f'{model_name}_{timestamp}_batch_recall.png'))
    
    # 5. AUC-ROC 곡선
    plot_auc_curve(history, best_epoch, model_name,
                  os.path.join(plot_dir, f'{model_name}_{timestamp}_auc_curve.png'))
    
    # 6. Metrics Grid
    plot_metrics_grid(history, test_metrics, best_epoch, model_name,
                     os.path.join(plot_dir, f'{model_name}_{timestamp}_metrics_grid.png'))
    
    # 7. 혼동 행렬
    plot_confusion_matrix(cm, class_names, model_name,
                         os.path.join(plot_dir, f'{model_name}_{timestamp}_confusion_matrix.png'))
    
    # 8. 클래스별 성능
    plot_per_class(per_class_report, class_names, model_name,
                  os.path.join(plot_dir, f'{model_name}_{timestamp}_per_class.png'))
    
    # 9. 종합 대시보드
    plot_dashboard(history, test_metrics, cm, per_class_report, class_names,
                  best_epoch, stop_epoch, training_time, best_val_recall,
                  model_name, os.path.join(plot_dir, f'{model_name}_{timestamp}_dashboard.png'))
    
    print(f'시각화 플롯 저장됨: {plot_dir}')
