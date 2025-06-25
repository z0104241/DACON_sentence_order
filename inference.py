#!/usr/bin/env python3
"""
Gemma-3-12B-it 특화 추론 스크립트 (marker prompt + 4bit)
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

def load_model_and_tokenizer() -> Tuple[PeftModel, AutoTokenizer]:
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
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()
    print("✅ 모델 로드 완료!")
    return model, tokenizer

def create_gemma_prompt(sentences):
    """Gemma marker 프롬프트 생성"""
    return PROMPT_TEMPLATE.format(
        sentence_0=sentences[0],
        sentence_1=sentences[1],
        sentence_2=sentences[2],
        sentence_3=sentences[3]
    )

def predict_batch_return_raw(sentences_batch: List[List[str]], model: PeftModel, tokenizer: AutoTokenizer) -> Tuple[List[str], List[str]]:
    prompts = [create_gemma_prompt(sentences) for sentences in sentences_batch]
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    results = []
    raw_outputs = []
    for i, output in enumerate(outputs):
        input_length = inputs.input_ids[i].shape[0]
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        results.append(extract_order_enhanced(generated))
        raw_outputs.append(generated)
    return results, raw_outputs

def extract_order_enhanced(text: str) -> str:
    """Gemma marker 기반 파싱 강화"""
    # 모델 출력에서 <end_of_turn> 앞/뒤 클린업
    text = text.split("<end_of_turn>")[0].strip()
    # marker 다음 숫자 시퀀스 추출
    patterns = [
        r'<start_of_turn>model\s*([0-3][,0-3 ]{5,})',
        r'([0-3]),\s*([0-3]),\s*([0-3]),\s*([0-3])',
        r'([0-3])\s+([0-3])\s+([0-3])\s+([0-3])',
        r'([0-3])-([0-3])-([0-3])-([0-3])',
        r'([0-3])\.([0-3])\.([0-3])\.([0-3])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            numbers = re.findall(r'[0-3]', match.group(0))
            if len(numbers) == 4 and set(numbers) == {0,1,2,3}:
                return ''.join(numbers)
    # 4자리 연속 숫자
    four_digit = re.search(r'\b([0-3]{4})\b', text)
    if four_digit:
        return four_digit.group(1)
    # 전체에서 숫자 4개 추출
    all_numbers = re.findall(r'[0-3]', text)
    if len(all_numbers) >= 4:
        return ''.join(all_numbers[:4])
    return text

def extract_sentences(row: pd.Series) -> List[str]:
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
    return [str(row.iloc[i]) for i in range(min(4, len(row)))]

def process_result(predicted_order: str) -> Dict[str, Union[int, str]]:
    if len(predicted_order) == 4 and predicted_order.isdigit():
        return {
            'answer_0': int(predicted_order[0]),
            'answer_1': int(predicted_order[1]),
            'answer_2': int(predicted_order[2]),
            'answer_3': int(predicted_order[3]),
            'raw_output': '',
            'parsing_status': 'SUCCESS'
        }
    else:
        return {
            'answer_0': '',
            'answer_1': '',
            'answer_2': '',
            'answer_3': '',
            'raw_output': predicted_order,
            'parsing_status': 'FAILED'
        }

def main(input_file: str, output_file: str) -> pd.DataFrame:
    import time
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer()
    df = pd.read_csv(input_file)
    print(f"📂 데이터 로드: {len(df)}개 행")
    results = []
    total_batches = (len(df) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    print(f"🚀 배치 크기: {INFERENCE_BATCH_SIZE}, 총 배치: {total_batches}")
    for batch_idx in tqdm(range(total_batches), desc="배치 처리"):
        start_idx = batch_idx * INFERENCE_BATCH_SIZE
        end_idx = min(start_idx + INFERENCE_BATCH_SIZE, len(df))
        batch_rows = df.iloc[start_idx:end_idx]
        try:
            sentences_batch = [extract_sentences(row) for _, row in batch_rows.iterrows()]
            predicted_orders, raw_outputs = predict_batch_return_raw(sentences_batch, model, tokenizer)
            for i, (row_idx, _) in enumerate(batch_rows.iterrows()):
                result = process_result(predicted_orders[i])
                results.append({
                    'ID': f'TEST_{row_idx:04d}',
                    'answer_0': result['answer_0'],
                    'answer_1': result['answer_1'],
                    'answer_2': result['answer_2'],
                    'answer_3': result['answer_3'],
                    'context': raw_outputs[i],
                    'raw_output': result['raw_output'],
                    'parsing_status': result['parsing_status'],
                    'sentence_0': sentences_batch[i][0],
                    'sentence_1': sentences_batch[i][1],
                    'sentence_2': sentences_batch[i][2],
                    'sentence_3': sentences_batch[i][3]
                })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            for i, (row_idx, _) in enumerate(batch_rows.iterrows()):
                results.append({
                    'ID': f'TEST_{row_idx:04d}',
                    'answer_0': '',
                    'answer_1': '',
                    'answer_2': '',
                    'answer_3': '',
                    'context': '',
                    'raw_output': f'BATCH_ERROR: {e}',
                    'parsing_status': 'ERROR',
                    'sentence_0': '',
                    'sentence_1': '',
                    'sentence_2': '',
                    'sentence_3': ''
                })
    results_df = pd.DataFrame(results)
    # 결측/파싱실패 재추론 생략 (동일 로직 가능)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    success_count = (results_df['parsing_status'] == 'SUCCESS').sum()
    failed_count = (results_df['parsing_status'] == 'FAILED').sum()
    error_count = (results_df['parsing_status'] == 'ERROR').sum()
    total_count = len(results_df)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    elapsed_time = time.time() - start_time
    samples_per_sec = len(df) / elapsed_time
    print(f"💾 순수 AI 추론 결과 저장: {output_file}")
    print(f"📊 파싱 성공률: {success_rate:.1f}% ({success_count}/{total_count})")
    print(f"   - 성공: {success_count}")
    print(f"   - 실패: {failed_count}")
    print(f"   - 에러: {error_count}")
    print(f"⏱️  처리 시간: {elapsed_time:.1f}초")
    print(f"🚀 처리 속도: {samples_per_sec:.1f} 샘플/초")
    return results_df.iloc[:,:5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gemma-3-12B-it marker 기반 추론')
    parser.add_argument('--input', '-i', default='test.csv', help='입력 CSV 파일')
    parser.add_argument('--output', '-o', default='predictions_gemma.csv', help='출력 CSV 파일')
    args = parser.parse_args()
    try:
        main(args.input, args.output)
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류: {e}")
