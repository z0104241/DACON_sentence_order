import os
# 모델 설정
MODEL_NAME = "Qwen/Qwen3-8B"  # 베이스 모델
HUGGINGFACE_REPO = "z0104241/DACON_sentence_order"
ADAPTER_SUBFOLDER = "qwen3_model"
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

# === 기존 파라미터 (주석 처리) ===
# LEARNING_RATE = 1e-4
# BATCH_SIZE = 1
# GRAD_ACCUMULATION = 16
# MAX_STEPS = 1000
# WARMUP_STEPS = 100
# SAVE_STEPS = 200

# === A5000(24GB), 증강데이터 기준 파라미터 ===
LEARNING_RATE = 2e-4    # 더 큰 데이터, 배치크기 증가에 맞춰 살짝 상향
BATCH_SIZE = 8         # 단일 GPU에서 4bit, LoRA시 24GB 충분
GRAD_ACCUMULATION = 8   # Effective batch size 32 (4x8)
MAX_STEPS = 800        # (데이터 28k / 32 = 약 900step/epoch) → 2~3epoch 정도 커버
WARMUP_STEPS = 200
SAVE_STEPS = 10000        # 더 자주 저장

# 추론 설정 등 기타 동일
#TRAIN_FILE = "train.csv"   
TRAIN_FILE = "train_augmented.csv"  
TEST_FILE = "test.csv"
OUTPUT_DIR = "qwen3_model"
PREDICTIONS_FILE = "predictions.csv"
CACHE_DIR = "./model_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
TEMPERATURE = 0.1
TOP_P = 1.0
# TOP_K = 20
MAX_NEW_TOKENS = 150
INFERENCE_BATCH_SIZE = 8  # 배치 처리용
CHECKPOINT_INTERVAL = 10000  # 체크포인트 저장 간격

# 프롬프트 설정
FEWSHOT_EXAMPLE = """예시:
문장들:
0: 119에 신고했다.
1: 아파트에서 화재가 발생했다.
2: 소방차가 현장에 도착했다.
3: 불이 완전히 진화되었다.
답: 1,0,2,3
"""

PROMPT_TEMPLATE = (
    "다음은 문장 순서 배열의 예시입니다. 문맥을 파악하여 가장 자연스러운 순서를 찾으세요.\n\n"
    + FEWSHOT_EXAMPLE +
    "이제 다음 문장들을 배열하세요:\n\n"
    "문장들:\n"
    "0: {sentence_0}\n"
    "1: {sentence_1}\n"
    "2: {sentence_2}\n"
    "3: {sentence_3}\n"
    "답: <|im_start|>assistant"
)
