"""GPU 전용 설정"""

import os

# 모델 설정
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048

# LoRA 설정
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 훈련 설정
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUMULATION = 16
MAX_STEPS = 1000
WARMUP_STEPS = 100
SAVE_STEPS = 200

# 추론 설정
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_NEW_TOKENS = 100

# 파일 경로
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_DIR = "qwen3_model"
PREDICTIONS_FILE = "predictions.csv"

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)