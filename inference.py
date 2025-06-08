#!/usr/bin/env python3
"""
순수 AI 추론 스크립트 (후처리 제외)
Few-shot 프롬프트 + 배치 처리 + 메모리 관리
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
from tqdm import tqdm
import argparse
import os
from typing import List, Tuple, Dict, Any, Union
import time
from config import (
    MODEL_NAME, 
    HUGGINGFACE_REPO, 
    ADAPTER_SUBFOLDER, 
    CACHE_DIR,
    USE_4BIT,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    MAX_NEW_TOKENS,
    INFERENCE_BATCH_SIZE,
    PROMPT_TEMPLATE,
    OUTPUT_DIR  
)


### 허깅페이스 업로드용
# def load_model_and_tokenizer() -> Tuple[PeftModel, AutoTokenizer]: 
#     """모델과 토크나이저 로드"""
#     print("🔧 모델 로드 중...")
    
#     # 토크나이저
#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_NAME,  
#         trust_remote_code=True,
#         cache_dir=CACHE_DIR  
#     )
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"
    
#     # 양자화 설정
#     compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=USE_4BIT,
#         bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
#         bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
#         bnb_4bit_compute_dtype=compute_dtype,
#     )
    
#     # 베이스 모델
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         cache_dir=CACHE_DIR
#     )
    
#     # LoRA 어댑터 로드
#     model = PeftModel.from_pretrained(
#         base_model, 
#         HUGGINGFACE_REPO,
#         subfolder=ADAPTER_SUBFOLDER,
#         cache_dir=CACHE_DIR
#     )
#     model.eval()
    
#     print("✅ 모델 로드 완료!")
#     return model, tokenizer



def load_model_and_tokenizer() -> Tuple[PeftModel, AutoTokenizer]:
    """모델과 토크나이저 로드 (로컬 어댑터 경로 사용)"""
    print("🔧 모델 로드 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,  
        trust_remote_code=True,
        cache_dir=CACHE_DIR  
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )

    # === 로컬 어댑터 로드 ===
    
    model = PeftModel.from_pretrained(
        base_model, 
        OUTPUT_DIR    
    )
    model.eval()

    print("✅ 모델 로드 완료!")
    return model, tokenizer

def create_fewshot_prompt(sentences):
    """config.py의 템플릿으로 프롬프트 생성"""
    return PROMPT_TEMPLATE.format(
        sentence_0=sentences[0],
        sentence_1=sentences[1],
        sentence_2=sentences[2],
        sentence_3=sentences[3]
    )


def predict_batch(sentences_batch: List[List[str]], model: PeftModel, tokenizer: AutoTokenizer) -> List[str]:
    """배치 단위로 문장 순서 예측"""
    
    # 프롬프트 생성 (Few-shot 적용)
    prompts = [create_fewshot_prompt(sentences) for sentences in sentences_batch]
    
    # 메시지 형태로 변환
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
             for messages in messages_batch]
    
    # 토크나이징
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to(model.device)
    
    # 배치 추론 (greedy decoding)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decoding
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    
    # 결과 디코딩
    results = []
    for i, output in enumerate(outputs):
        input_length = inputs.input_ids[i].shape[0]
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        results.append(extract_order_enhanced(generated))
    
    return results

def extract_order_enhanced(text: str) -> str:
    """강화된 파싱 함수"""
    
    # 1순위: 정확한 패턴들
    exact_patterns = [
        r'답[:：]\s*([0-3]),\s*([0-3]),\s*([0-3]),\s*([0-3])',
        r'답[:：]\s*([0-3])\s*,\s*([0-3])\s*,\s*([0-3])\s*,\s*([0-3])',
        r'순서[:：]\s*([0-3]),\s*([0-3]),\s*([0-3]),\s*([0-3])',
        r'([0-3]),\s*([0-3]),\s*([0-3]),\s*([0-3])',
        r'([0-3])\s*→\s*([0-3])\s*→\s*([0-3])\s*→\s*([0-3])',
        r'([0-3])\s+([0-3])\s+([0-3])\s+([0-3])',
        r'([0-3])-([0-3])-([0-3])-([0-3])',
        r'([0-3])\.([0-3])\.([0-3])\.([0-3])',
    ]
    
    for pattern in exact_patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 4:
                result = [int(g) for g in match.groups()]
            else:
                result = [int(d) for d in match.group(1) if d.isdigit()]
                
            if len(result) == 4 and set(result) == {0,1,2,3}:
                return ''.join(map(str, result))
    
    # 2순위: 4자리 연속 숫자
    four_digit = re.search(r'\b([0-3]{4})\b', text)
    if four_digit:
        return four_digit.group(1)
    
    # 3순위: 순서/답 키워드 뒤 숫자들
    patterns = [r'순서[:：]\s*([0-3\s,]+)', r'답[:：]\s*([0-3\s,]+)', r'결과[:：]\s*([0-3\s,]+)']
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            numbers = re.findall(r'[0-3]', match.group(1))
            if len(numbers) >= 4:
                return ''.join(numbers[:4])
    
    # 4순위: 마지막 줄에서 숫자들
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        numbers = re.findall(r'[0-3]', lines[-1])
        if len(numbers) >= 4:
            return ''.join(numbers[:4])
    
    # 5순위: 전체에서 숫자들
    all_numbers = re.findall(r'[0-3]', text)
    if len(all_numbers) >= 4:
        return ''.join(all_numbers[-4:])
    
    # 파싱 실패시 원본 텍스트 반환
    return text

def extract_sentences(row: pd.Series) -> List[str]:
    """행에서 문장들 추출"""
    patterns = [
        ['sentence_0', 'sentence_1', 'sentence_2', 'sentence_3'],
        ['sent_0', 'sent_1', 'sent_2', 'sent_3'],
        ['text_0', 'text_1', 'text_2', 'text_3'],
        ['0', '1', '2', '3']
    ]
    
    for pattern in patterns:
        try:
            return [str(row[col]) for col in pattern]
        except KeyError:
            continue
    
    # 처음 4개 컬럼 사용
    return [str(row.iloc[i]) for i in range(min(4, len(row)))]

def process_result(predicted_order: str) -> Dict[str, Union[int, str]]:
    """예측 결과를 처리 (후처리 없음)"""
    if len(predicted_order) == 4 and predicted_order.isdigit():
        # 파싱 성공
        return {
            'answer_0': int(predicted_order[0]),
            'answer_1': int(predicted_order[1]),
            'answer_2': int(predicted_order[2]),
            'answer_3': int(predicted_order[3]),
            'raw_output': '',
            'parsing_status': 'SUCCESS'
        }
    else:
        # 파싱 실패
        return {
            'answer_0': '',
            'answer_1': '',
            'answer_2': '',
            'answer_3': '',
            'raw_output': predicted_order,
            'parsing_status': 'FAILED'
        }

def main(input_file: str, output_file: str) -> pd.DataFrame:
    """메인 실행 함수"""
    
    start_time = time.time()
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 데이터 로드
    df = pd.read_csv(input_file)#.head(5)
    print(f"📂 데이터 로드: {len(df)}개 행")
    
    # 배치별 처리
    results = []
    total_batches = (len(df) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    
    print(f"🚀 배치 크기: {INFERENCE_BATCH_SIZE}, 총 배치: {total_batches}")
    
    for batch_idx in tqdm(range(total_batches), desc="배치 처리"):
        # 배치 데이터 준비
        start_idx = batch_idx * INFERENCE_BATCH_SIZE
        end_idx = min(start_idx + INFERENCE_BATCH_SIZE, len(df))
        batch_rows = df.iloc[start_idx:end_idx]
        
        try:
            # 배치 내 문장들 추출
            sentences_batch = [extract_sentences(row) for _, row in batch_rows.iterrows()]
            
            # 배치 예측
            predicted_orders = predict_batch(sentences_batch, model, tokenizer)
            
            # 결과 저장 (순수 AI 결과만)
            for i, (row_idx, _) in enumerate(batch_rows.iterrows()):
                result = process_result(predicted_orders[i])
                results.append({
                    'ID': f'TEST_{row_idx:04d}',
                    'answer_0': result['answer_0'],
                    'answer_1': result['answer_1'],
                    'answer_2': result['answer_2'],
                    'answer_3': result['answer_3'],
                    'raw_output': result['raw_output'],
                    'parsing_status': result['parsing_status'],
                    # 후처리용 원본 문장들 저장
                    'sentence_0': sentences_batch[i][0],
                    'sentence_1': sentences_batch[i][1],
                    'sentence_2': sentences_batch[i][2],
                    'sentence_3': sentences_batch[i][3]
                })
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ 배치 {batch_idx} 오류: {e}")
            # 배치 전체를 에러로 처리
            for i, (row_idx, _) in enumerate(batch_rows.iterrows()):
                results.append({
                    'ID': f'TEST_{row_idx:04d}',
                    'answer_0': '',
                    'answer_1': '',
                    'answer_2': '',
                    'answer_3': '',
                    'raw_output': f'BATCH_ERROR: {e}',
                    'parsing_status': 'ERROR',
                    'sentence_0': '',
                    'sentence_1': '',
                    'sentence_2': '',
                    'sentence_3': ''
                })
    
    # 최종 결과 저장
    results_df = pd.DataFrame(results)
    results_df.iloc[:,:5].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 파싱 성공률 계산
    success_count = len(results_df[results_df['parsing_status'] == 'SUCCESS'])
    failed_count = len(results_df[results_df['parsing_status'] == 'FAILED'])
    error_count = len(results_df[results_df['parsing_status'] == 'ERROR'])
    total_count = len(results_df)
    
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    # 처리 시간 계산
    elapsed_time = time.time() - start_time
    samples_per_sec = len(df) / elapsed_time
    
    print(f"💾 순수 AI 추론 결과 저장: {output_file}")
    print(f"📊 파싱 성공률: {success_rate:.1f}% ({success_count}/{total_count})")
    print(f"   - 성공: {success_count}")
    print(f"   - 실패: {failed_count}")
    print(f"   - 에러: {error_count}")
    print(f"⏱️  처리 시간: {elapsed_time:.1f}초")
    print(f"🚀 처리 속도: {samples_per_sec:.1f} 샘플/초")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='순수 AI 추론 (후처리 제외)')
    parser.add_argument('--input', '-i', default='test.csv', help='입력 CSV 파일')
    parser.add_argument('--output', '-o', default='predictions_0527.csv', help='출력 CSV 파일')
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output)
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류: {e}")