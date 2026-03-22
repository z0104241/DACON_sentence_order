import os

# 모델 설정
MODEL_NAME = "google/gemma-3-12b-it"
HUGGINGFACE_REPO = "z0104241/DACON_sentence_order"
ADAPTER_SUBFOLDER = "gemma3_model"
MAX_SEQ_LENGTH = 2048

# 양자화 설정 (QLoRA)
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # A5000은 bfloat16 지원
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = False

# LoRA 설정
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 훈련 파라미터 (A5000 24GB, 증강 데이터 기준)
LEARNING_RATE = 2e-4
BATCH_SIZE = 3
GRAD_ACCUMULATION = 8
MAX_STEPS = 1800
WARMUP_STEPS = 100
SAVE_STEPS = 300  # 1800 steps 동안 6회 저장

# 파일 경로
TRAIN_FILE = "train.csv"
TRAIN_AUG_FILE = "train_augmented_gemma.csv"
TEST_FILE = "test.csv"
OUTPUT_DIR = "gemma3_model"
PREDICTIONS_FILE = "predictions.csv"
CACHE_DIR = "./model_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 추론 설정
TEMPERATURE = 0.1
TOP_P = 1.0
MAX_NEW_TOKENS = 150
INFERENCE_BATCH_SIZE = 8

# Gemma-3 전용 프롬프트
FEWSHOT_EXAMPLE = """
<start_of_turn>user
[예시]
문장:
0: 119에 신고했다.
1: 아파트에서 화재가 발생했다.
2: 소방차가 현장에 도착했다.
3: 불이 완전히 진화되었다.
정답: 1,0,2,3
<end_of_turn>
<start_of_turn>model
1,0,2,3
<end_of_turn>
"""

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "당신은 문장들의 자연스러운 순서를 판단하는 언어 전문가입니다.\n"
    "아래 문장들을 가장 자연스러운 흐름으로 배열하세요.\n"
    "문장:\n"
    "0: {sentence_0}\n"
    "1: {sentence_1}\n"
    "2: {sentence_2}\n"
    "3: {sentence_3}\n"
    "정답:\n"
    "<end_of_turn>\n"
    "<start_of_turn>model"
)
