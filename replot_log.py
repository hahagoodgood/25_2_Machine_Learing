
import os
import json
import argparse
import torch
import visualization
import config

def replot_from_log(log_path):
    """
    기존 학습 로그(JSON)를 읽어 시각화 플롯을 다시 생성합니다.
    새로운 visualization.py 스타일이 적용됩니다.
    """
    if not os.path.exists(log_path):
        print(f"Error: File not found - {log_path}")
        return

    print(f"Loading log from {log_path}...")
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # 로그 데이터 파싱
    history = log_data['history']
    test_metrics = log_data.get('test_metrics', {})
    
    # Confusion Matrix 복원 (리스트 -> numpy)
    cm = log_data.get('confusion_matrix')
    if cm:
        cm = __import__('numpy').array(cm)
    else:
        cm = __import__('numpy').zeros((3, 3)) # Dummy if missing
        
    per_class_report = log_data.get('per_class_report', {})
    
    # 메타데이터
    model_name = log_data.get('model_name', 'model')
    timestamp = log_data.get('timestamp', 'replot')
    best_epoch = log_data.get('best_epoch', 0)
    stop_epoch = log_data.get('stop_epoch', len(history['train_loss']))
    training_time = log_data.get('training_time', 0)
    best_val_recall = log_data.get('best_val_recall', 0.0)
    
    # 디렉토리 설정
    base_dir = os.path.dirname(log_path)
    plot_dir = os.path.join(base_dir, 'plots_replot') # 덮어쓰기 방지 위해 별도 폴더
    
    print(f"Generating plots to {plot_dir}...")
    
    # 배치 수 추정 (로그에 없다면 config 사용)
    batches_per_epoch = len(history.get('train_batch', [[]])[0].get('loss', [])) if history.get('train_batch') else len(config.TRAIN_LOADER) if hasattr(config, 'TRAIN_LOADER') else 100
    # config에 TRAIN_LOADER가 없으므로 대략적인 값이나, train_batch 길이를 이용
    if history.get('train_batch'):
         batches_per_epoch = len(history['train_batch'][0]['loss'])
    else:
         batches_per_epoch = 1 # Fallback
         
    visualization.generate_all_plots(
        history=history,
        test_metrics=test_metrics,
        cm=cm,
        per_class_report=per_class_report,
        class_names=config.CLASS_NAMES,
        model_name=model_name,
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
        training_time=training_time,
        best_val_recall=best_val_recall,
        plot_dir=plot_dir,
        timestamp=timestamp,
        batches_per_epoch=batches_per_epoch
    )
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-plot detailed visualizations from training log.')
    parser.add_argument('log_path', type=str, help='Path to training_log.json file')
    args = parser.parse_args()
    
    replot_from_log(args.log_path)
