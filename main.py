"""간소화된 메인 실행 파일"""

import argparse
import pandas as pd

def train_mode():
    """훈련 모드"""
    print("훈련 모드")
    from train import main as train_main
    train_main()

def inference_mode(args):
    """추론 모드"""
    print("추론 모드")
    from inference import Predictor
    
    predictor = Predictor()
    if args.test_file and pd.io.common.file_exists(args.test_file):
        predictor.predict_csv(args.test_file, args.output)
    else:
        # 데모
        test_sentences = [
            "블록체인 기술은 투표 과정의 투명성을 크게 향상시킬 수 있다.",
            "이러한 특성은 유권자들에게 신뢰를 제공하며, 민주적 참여를 촉진하는 데 기여할 수 있다.",
            "결과적으로 블록체인 기반의 투표 시스템은 공정하고 신뢰할 수 있는 선거 환경을 조성할 잠재력을 지닌다.",
            "각 투표는 변경 불가능한 기록으로 저장되어 조작의 가능성을 원천적으로 차단한다."
        ]
        
        predictor.load_model()
        normal_pred = predictor.predict_single(test_sentences, use_thinking=False)
        thinking_pred = predictor.predict_single(test_sentences, use_thinking=True)
        
        print("테스트 문장:")
        for i, sent in enumerate(test_sentences):
            print(f"  {i}: {sent}")
        print(f"\n일반 모드: {normal_pred}")
        print(f"사고 모드: {thinking_pred}")

def eval_mode():
    """평가 모드"""
    print("평가 모드")
    from eval import main as eval_main
    eval_main()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    
    # 훈련
    subparsers.add_parser('train')
    
    # 추론
    inf_parser = subparsers.add_parser('inference')
    inf_parser.add_argument('--test_file', type=str, default="test.csv")
    inf_parser.add_argument('--output', type=str, default="predictions.csv")
    
    # 평가
    subparsers.add_parser('eval')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode()
    elif args.mode == 'inference':
        inference_mode(args)
    elif args.mode == 'eval':
        eval_mode()
    else:
        print("사용법:")
        print("  python main.py train")
        print("  python main.py inference [--test_file test.csv] [--output predictions.csv]")
        print("  python main.py eval")

if __name__ == "__main__":
    main()