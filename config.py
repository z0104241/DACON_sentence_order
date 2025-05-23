"""GPU 전용 설정"""

import os

# 모델 설정
MODEL_NAME = "Qwen/Qwen3-8B"  # 베이스 모델
HUGGINGFACE_REPO = "z0104241/DACON_sentence_order"  # 허깅페이스 리포지토리
ADAPTER_SUBFOLDER = "qwen3_model"  # LoRA 어댑터가 있는 서브폴더
MAX_SEQ_LENGTH = 2048

# 양자화 설정
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = False

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
TEMPERATURE = 0.1
TOP_P = 0.95
# TOP_K = 20
MAX_NEW_TOKENS = 120
INFERENCE_BATCH_SIZE = 6  # 배치 처리용
CHECKPOINT_INTERVAL = 1000  # 체크포인트 저장 간격

# 파일 경로
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
OUTPUT_DIR = "qwen3_model"
PREDICTIONS_FILE = "predictions.csv"

# 로컬 캐시 디렉토리
CACHE_DIR = "./model_cache"

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)