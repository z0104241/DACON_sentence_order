import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
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
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    
    def format_data(df):
        formatted = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = config.PROMPT_TEMPLATE.format(
                sentence_0=row['sentence_0'],
                sentence_1=row['sentence_1'],
                sentence_2=row['sentence_2'],
                sentence_3=row['sentence_3'],
            ) + f" {row['answer_0']},{row['answer_1']},{row['answer_2']},{row['answer_3']}<|im_end|>"
            formatted.append({"text": text, "original_id": row.get("ID", idx)})
        return formatted

    train_data = format_data(train_df)
    val_data = format_data(val_df)
    return train_data, val_data

def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
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

def train_model_phase(model, tokenizer, train_list, val_list, steps, phase_name="Phase"):
    print(f"\n--- {phase_name} 학습({steps} steps) ---")
    train_dataset = Dataset.from_list(train_list)
    val_dataset = Dataset.from_list(val_list) if val_list else None

    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, phase_name),
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        max_steps=steps,
        warmup_steps=max(1, steps // 10),
        save_steps=max(1, steps // 2),
        do_eval=bool(val_dataset),
        eval_steps=max(1, steps // 2),
        logging_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(config.OUTPUT_DIR, f"{phase_name}_model"))
    tokenizer.save_pretrained(os.path.join(config.OUTPUT_DIR, f"{phase_name}_model"))
    return trainer

def get_losses_on_dataset(model, tokenizer, data_list, batch_size=1):
    device = model.device
    model.eval()
    
    # === [수정] 토크나이즈 진행 ===
    tokenized_list = []
    for item in tqdm(data_list, desc="Tokenizing for HNM"):
        encoding = tokenizer(
            item["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt"
        )
        # flatten (배치 차원 제거)
        d = {k: v.squeeze(0) for k, v in encoding.items()}
        d["original_id"] = item["original_id"]
        tokenized_list.append(d)
    
    temp_dataset = Dataset.from_list(tokenized_list)
    
    def collate_fn(batch):
        keys = ['input_ids', 'attention_mask']
        return {k: torch.stack([torch.tensor(item[k]) for item in batch]) for k in keys}


    dataloader = DataLoader(temp_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    all_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="HNM: Loss 계산 중"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            batch_loss = outputs.loss.item()
            for _ in range(batch["input_ids"].shape[0]):
                all_losses.append(batch_loss)
    return all_losses


def main():
    print("데이터 준비...")
    train_data, val_data = load_and_prepare_data()
    print("모델 설정...")
    model, tokenizer = setup_model()
    print("HNM 단계적 파인튜닝 시작!")

    # --- 초기 파인튜닝 ---
    if getattr(config, "DO_HNM", True):
        initial_steps = int(config.MAX_STEPS * getattr(config, "HNM_INITIAL_TRAIN_RATIO", 0.5))
        trainer = train_model_phase(model, tokenizer, train_data, val_data, initial_steps, "initial")

        # --- Hard-negative mining ---
        print("\nHard Negative Mining...")
        losses = get_losses_on_dataset(trainer.model, tokenizer, train_data, batch_size=getattr(config, "HNM_BATCH_SIZE_FOR_LOSS_CALC", 2))
        # 상위 N% hard sample 추출
        n_hard = int(len(losses) * getattr(config, "HNM_HARD_SAMPLE_RATIO", 0.3))
        hard_indices = np.argsort(losses)[-n_hard:]  # loss 높은 샘플
        hard_train_data = [train_data[i] for i in hard_indices]

        print(f"Hard sample {len(hard_train_data)}개로 2차 집중 파인튜닝")
        finetune_steps = config.MAX_STEPS - initial_steps
        train_model_phase(trainer.model, tokenizer, hard_train_data, [], finetune_steps, "focused_hnm")

        print("HNM 파인튜닝 완료!")
    else:
        # HNM 미사용시 전체 데이터로 일반 파인튜닝
        train_model_phase(model, tokenizer, train_data, val_data, config.MAX_STEPS, "full")

if __name__ == "__main__":
    main()
