#!/usr/bin/env python3
"""
문장 순서 예측 추론 스크립트
Qwen3-8B + PEFT 모델을 사용하여 문장 순서를 예측하고 CSV로 저장
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import argparse
import os

# 설정
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_PATH = "qwen3_model"  # 학습된 모델 경로

def load_model_and_tokenizer():
    """모델과 토크나이저 로드"""
    
    print("🔧 모델 로드 중...")
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 베이스 모델
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype='auto'
    )
    
    # 학습된 모델 로드
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    
    print("✅ 모델 로드 완료!")
    return model, tokenizer

def predict_order(sentences, model, tokenizer):
    """긴추론 방식으로 문장 순서 예측"""
    
    prompt = f"""다음 문장들을 논리적으로 연결되는 순서로 재배열하세요.

문장들:
0: {sentences[0]}
1: {sentences[1]}
2: {sentences[2]}
3: {sentences[3]}

단계별로 분석해보고 최종 순서를 숫자로 답하세요:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    
    input_length = inputs.input_ids.shape[1]
    generated = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    # 순서 추출
    order = extract_final_order(generated)
    return order

def extract_final_order(text):
    """텍스트에서 최종 순서 추출"""
    
    # 4자리 연속 숫자 우선
    four_digit = re.search(r'\b([0-3]{4})\b', text)
    if four_digit:
        return four_digit.group(1)
    
    # 마지막 라인에서 숫자들
    lines = text.split('\n')
    for line in reversed(lines):
        numbers = re.findall(r'[0-3]', line)
        if len(numbers) >= 4:
            return ''.join(numbers[:4])
    
    # 전체에서 숫자들
    all_numbers = re.findall(r'[0-3]', text)
    if len(all_numbers) >= 4:
        return ''.join(all_numbers[-4:])
    
    return "0123"

def load_test_data(file_path):
    """테스트 데이터 로드"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"테스트 파일을 찾을 수 없습니다: {file_path}")
    
    print(f"📂 테스트 데이터 로드: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   데이터 크기: {df.shape}")
    
    return df

def extract_sentences_from_row(row):
    """행에서 문장들 추출"""
    
    # 다양한 컬럼명 시도
    column_patterns = [
        ['sentence_0', 'sentence_1', 'sentence_2', 'sentence_3'],
        ['sent_0', 'sent_1', 'sent_2', 'sent_3'],
        ['text_0', 'text_1', 'text_2', 'text_3'],
        ['0', '1', '2', '3']
    ]
    
    for pattern in column_patterns:
        try:
            sentences = [row[col] for col in pattern]
            return sentences
        except KeyError:
            continue
    
    # 마지막 시도: 처음 4개 컬럼
    try:
        sentences = [row.iloc[i] for i in range(4)]
        return sentences
    except:
        raise ValueError("문장 컬럼을 찾을 수 없습니다.")

def predict_all_rows(df, model, tokenizer):
    """df의 모든 행에 대해 순서 예측"""
    
    print(f"🚀 총 {len(df)}개 행 처리 시작...")
    print("=" * 60)
    
    results = []
    error_count = 0
    
    # 진행바 포함 반복
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="순서 예측"):
        
        try:
            # 문장들 추출
            sentences = extract_sentences_from_row(row)
            
            # 순서 예측
            predicted_order = predict_order(sentences, model, tokenizer)
            
            # 예측 순서를 개별 컬럼으로 분리
            if len(predicted_order) == 4 and predicted_order.isdigit():
                answer_0 = int(predicted_order[0])
                answer_1 = int(predicted_order[1])
                answer_2 = int(predicted_order[2])
                answer_3 = int(predicted_order[3])
            else:
                # 길이가 4가 아니면 기본값
                answer_0, answer_1, answer_2, answer_3 = 0, 1, 2, 3
                error_count += 1
            
            results.append({
                'ID': f'TEST_{idx:04d}',
                'answer_0': answer_0,
                'answer_1': answer_1,
                'answer_2': answer_2,
                'answer_3': answer_3
            })
        
        except Exception as e:
            print(f"❌ 행 {idx} 처리 실패: {e}")
            error_count += 1
            # 실패시 기본값으로 추가
            results.append({
                'ID': f'TEST_{idx:04d}',
                'answer_0': 0,
                'answer_1': 1,
                'answer_2': 2,
                'answer_3': 3
            })
            continue
    
    print("=" * 60)
    print(f"✅ 완료! {len(results)}개 행 처리됨")
    if error_count > 0:
        print(f"⚠️  {error_count}개 행에서 오류 발생 (기본값 사용)")
    
    return results

def analyze_results(results):
    """결과 분석 및 통계"""
    
    # 순서 재구성해서 분석
    orders = []
    for r in results:
        order = f"{r['answer_0']}{r['answer_1']}{r['answer_2']}{r['answer_3']}"
        orders.append(order)
    
    unique_orders = set(orders)
    
    print(f"\n📊 결과 분석:")
    print(f"총 {len(unique_orders)}가지 순서 패턴")
    print("-" * 30)
    
    # 빈도순 정렬
    order_counts = {order: orders.count(order) for order in unique_orders}
    sorted_orders = sorted(order_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 10개만 출력
    for order, count in sorted_orders[:10]:
        percentage = (count / len(orders)) * 100
        print(f"{order}: {count:4d}번 ({percentage:5.1f}%)")
    
    if len(sorted_orders) > 10:
        print(f"... ({len(sorted_orders) - 10}가지 더)")
    
    # 다양성 지수
    diversity = len(unique_orders) / len(orders) * 100
    print(f"\n🎯 다양성 지수: {diversity:.1f}%")
    
    if diversity < 10:
        print("⚠️  매우 낮은 다양성 - 심각한 편향")
    elif diversity < 25:
        print("🔶 낮은 다양성 - 일부 편향 존재") 
    elif diversity < 50:
        print("🔵 보통 다양성")
    else:
        print("✅ 높은 다양성 - 좋은 성능")

def save_results(results, filename):
    """결과를 지정된 형식의 CSV로 저장"""
    
    results_df = pd.DataFrame(results)
    
    # 컬럼 순서 정리
    results_df = results_df[['ID', 'answer_0', 'answer_1', 'answer_2', 'answer_3']]
    
    # CSV 저장
    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"💾 결과가 {filename}에 저장되었습니다.")

    return results_df

def main(input_file, output_file):
    """메인 실행 함수"""
    
    print("🎯 문장 순서 예측 추론 시작!")
    print("=" * 60)
    
    # 1. 모델 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 테스트 데이터 로드
    df = load_test_data(input_file)
    
    # 3. 전체 예측
    results = predict_all_rows(df, model, tokenizer)
    
    # 4. 결과 분석
    analyze_results(results)
    
    # 5. 결과 저장
    results_df = save_results(results, output_file)
    
    print(f"\n🎉 추론 완료!")
    print(f"📁 입력 파일: {input_file}")
    print(f"📁 출력 파일: {output_file}")
    print(f"📊 처리된 샘플: {len(results)}개")
    
    return results, results_df

if __name__ == "__main__":
    
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='문장 순서 예측 추론')
    parser.add_argument('--input', '-i', default='test.csv', 
                      help='입력 CSV 파일 경로 (기본값: test.csv)')
    parser.add_argument('--output', '-o', default='prediction_results.csv',
                      help='출력 CSV 파일 경로 (기본값: prediction_results.csv)')
    
    args = parser.parse_args()
    
    try:
        # 메인 실행
        results, results_df = main(args.input, args.output)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("사용법: python inference.py --input test.csv --output results.csv")