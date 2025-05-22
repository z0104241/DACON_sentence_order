"""통합 평가 및 유틸리티 모듈"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from inference import Predictor
import config

def calculate_accuracy(true_orders, predicted_orders):
    """정확도 계산"""
    total = len(true_orders)
    exact_matches = sum(1 for t, p in zip(true_orders, predicted_orders) if t == p)
    
    position_matches = [0, 0, 0, 0]
    for true, pred in zip(true_orders, predicted_orders):
        for i in range(4):
            if true[i] == pred[i]:
                position_matches[i] += 1
    
    return {
        'exact_accuracy': exact_matches / total,
        'position_accuracies': [m / total for m in position_matches],
        'avg_position_accuracy': np.mean(position_matches) / total,
        'total_samples': total
    }

def evaluate_model(model_path=config.OUTPUT_DIR, num_samples=100):
    """모델 평가"""
    # 검증 데이터 준비
    df = pd.read_csv(config.TRAIN_FILE)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df = val_df.head(num_samples).reset_index(drop=True)
    
    # 예측 실행
    predictor = Predictor(model_path)
    predictor.load_model()
    
    true_orders = []
    predicted_orders = []
    
    print(f"평가 중... ({len(val_df)}개 샘플)")
    for _, row in val_df.iterrows():
        sentences = [row[f'sentence_{i}'] for i in range(4)]
        true_order = [row[f'answer_{i}'] for i in range(4)]
        
        pred_order = predictor.predict_single(sentences)
        
        true_orders.append(true_order)
        predicted_orders.append(pred_order)
    
    # 메트릭 계산
    metrics = calculate_accuracy(true_orders, predicted_orders)
    
    print(f"""
평가 결과:
- 정확도: {metrics['exact_accuracy']:.3f}
- 평균 위치 정확도: {metrics['avg_position_accuracy']:.3f}
- 위치별 정확도: {[f'{acc:.3f}' for acc in metrics['position_accuracies']]}
- 총 샘플: {metrics['total_samples']}
""")
    
    return metrics

def compare_modes(num_samples=50):
    """일반 모드 vs 사고 모드 비교"""
    df = pd.read_csv(config.TRAIN_FILE)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df = val_df.head(num_samples).reset_index(drop=True)
    
    predictor = Predictor()
    predictor.load_model()
    
    normal_preds = []
    thinking_preds = []
    
    print("모드 비교 중...")
    for _, row in val_df.iterrows():
        sentences = [row[f'sentence_{i}'] for i in range(4)]
        
        normal_pred = predictor.predict_single(sentences, use_thinking=False)
        thinking_pred = predictor.predict_single(sentences, use_thinking=True)
        
        normal_preds.append(normal_pred)
        thinking_preds.append(thinking_pred)
    
    # 일치율 계산
    agreement = sum(1 for n, t in zip(normal_preds, thinking_preds) if n == t)
    agreement_rate = agreement / len(normal_preds)
    
    print(f"두 모드 일치율: {agreement_rate:.3f} ({agreement}/{len(normal_preds)})")
    
    return normal_preds, thinking_preds, agreement_rate

def analyze_errors(true_orders, predicted_orders, sentences_list):
    """오류 분석"""
    errors = []
    for i, (true, pred, sentences) in enumerate(zip(true_orders, predicted_orders, sentences_list)):
        if true != pred:
            errors.append({
                'index': i,
                'true_order': true,
                'predicted_order': pred,
                'sentences': sentences
            })
    
    print(f"총 오류: {len(errors)} / {len(true_orders)} ({len(errors)/len(true_orders)*100:.1f}%)")
    
    # 가장 빈번한 틀린 패턴
    error_patterns = Counter([tuple(err['predicted_order']) for err in errors])
    print(f"가장 빈번한 오류 패턴: {error_patterns.most_common(3)}")
    
    return errors

def plot_accuracy(metrics, save_path="accuracy_plot.png"):
    """정확도 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 전체 정확도
    ax1.bar(['Exact Match', 'Avg Position'], 
           [metrics['exact_accuracy'], metrics['avg_position_accuracy']])
    ax1.set_title('Overall Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # 위치별 정확도
    ax2.bar(range(4), metrics['position_accuracies'])
    ax2.set_title('Position-wise Accuracy')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(range(4))
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"차트 저장: {save_path}")

def quick_test():
    """빠른 테스트"""
    # 샘플 문장
    test_sentences = [
        "블록체인 기술은 투표 과정의 투명성을 크게 향상시킬 수 있다.",
        "이러한 특성은 유권자들에게 신뢰를 제공하며, 민주적 참여를 촉진하는 데 기여할 수 있다.",
        "결과적으로 블록체인 기반의 투표 시스템은 공정하고 신뢰할 수 있는 선거 환경을 조성할 잠재력을 지닌다.",
        "각 투표는 변경 불가능한 기록으로 저장되어 조작의 가능성을 원천적으로 차단한다."
    ]
    
    predictor = Predictor()
    predictor.load_model()
    
    print("테스트 문장:")
    for i, sent in enumerate(test_sentences):
        print(f"  {i}: {sent}")
    
    normal_pred = predictor.predict_single(test_sentences, use_thinking=False)
    thinking_pred = predictor.predict_single(test_sentences, use_thinking=True)
    
    print(f"\n일반 모드 예측: {normal_pred}")
    print(f"사고 모드 예측: {thinking_pred}")

def main():
    """메인 함수"""
    print("1. 빠른 테스트")
    quick_test()
    
    print("\n2. 모델 평가")
    metrics = evaluate_model(num_samples=100)
    
    print("\n3. 정확도 차트 생성")
    plot_accuracy(metrics)

if __name__ == "__main__":
    main()