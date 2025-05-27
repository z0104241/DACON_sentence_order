"""통합 훈련 모듈 - 데이터 전처리 + 모델 + 훈련"""
import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



def load_and_prepare_data():
    """데이터 로드 및 준비"""
    df = pd.read_csv(config.TRAIN_FILE)
    
    # 훈련/검증 분할
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    
    # 데이터 포맷팅
    def format_data(df):
        formatted = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = config.PROMPT_TEMPLATE.format(
                sentence_0=row['sentence_0'],
                sentence_1=row['sentence_1'],
                sentence_2=row['sentence_2'],
                sentence_3=row['sentence_3'],
            ) + f" {row['answer_0']},{row['answer_1']},{row['answer_2']},{row['answer_3']}<|im_end|>"
            formatted.append({"text": text})
        return formatted

    train_data = format_data(train_df)
    val_data = format_data(val_df)
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"훈련: {len(train_dataset)}, 검증: {len(val_dataset)}")
    return train_dataset, val_dataset

def setup_model():
    """모델 및 토크나이저 설정"""
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 양자화 설정
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True
    # )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # LoRA 설정
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGETS,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(train_dataset, val_dataset, model, tokenizer):
    """모델 훈련"""
    # 데이터 콜레이터
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # 훈련 인자
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        max_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        save_steps=config.SAVE_STEPS,
        # evaluation_strategy="steps",  # 계속 주석 처리 또는 삭제된 상태로 둡니다.
        do_eval=True,                   # 이 줄을 추가하여 평가 수행을 명시합니다.
        eval_steps=config.SAVE_STEPS,    # 평가 간격을 지정합니다 (save_steps와 동일하게).
        logging_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,  # 이 옵션을 사용하려면 평가 전략과 저장 전략이 일치해야 합니다.
    )
    
    # 트레이너
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #max_seq_length=config.MAX_SEQ_LENGTH,
        #tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        #packing=False,
    )
    
    # 훈련 시작
    print("훈련 시작...")
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"모델 저장 완료: {config.OUTPUT_DIR}")
    
    return trainer

def main():
    """메인 함수"""
    print("데이터 준비...")
    train_dataset, val_dataset = load_and_prepare_data()
    
    print("모델 설정...")
    model, tokenizer = setup_model()
    
    print("훈련 시작...")
    trainer = train_model(train_dataset, val_dataset, model, tokenizer)
    
    print("훈련 완료!")

if __name__ == "__main__":
    main()