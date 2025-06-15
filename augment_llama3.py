import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from tqdm import tqdm
import torch
import random
import numpy as np
import itertools

# --- 시드 고정 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map={"": "cuda:0"}
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=1.1,
    do_sample=True,
    top_p=0.92,
    repetition_penalty=1.15
)

BATCH_SIZE = 3  # VRAM에 따라 조정

def make_prompt(sentence):
    return (
        "llm 학습용 데이터를 증강할거야. 아래 문장을 의미를 바꾸지 않고 다양한 표현으로 페러프레이징 해줘. **절대로 다른 언어는 사용하지말고 한국어만 사용해.**\n"
        "예시:\n"
        "문장: 인공지능은 미래 사회를 변화시킬 중요한 기술이다.\n"
        "출력: 미래 사회를 바꿀 핵심 기술 중 하나가 바로 인공지능이다.\n"
        "문장: 데이터 분석은 기업 의사결정에 큰 영향을 준다.\n"
        "출력: 기업이 의사결정을 내릴 때 데이터 분석이 중요한 역할을 한다.\n"
        f"문장: {sentence}\n"
        "출력:"
    )

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def cleanup():
    print("모델, 캐시, VRAM 비우는 중...")

    global model, tokenizer
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    gc.collect()

    hf_cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch/transformers"),
        os.getenv("TRANSFORMERS_CACHE", ""),
        os.getenv("HF_HOME", ""),
    ]
    for d in hf_cache_dirs:
        if d and os.path.exists(d):
            print(f"캐시 폴더 삭제: {d}")
            shutil.rmtree(d, ignore_errors=True)

    print("메모리, 캐시 정리 완료")


def main():
    # === train.csv 읽기 ===
    df = pd.read_csv('train.csv').head(2)

    # sentence 컬럼명 추출
    cols = [c for c in df.columns if c.startswith("sentence_")]
    ans_cols = [c for c in df.columns if c.startswith("answer_")]

    # ----------------------
    # 1. 파라프레이즈 데이터 생성 (증강행)
    # ----------------------
    to_paraphrase = []
    for idx, row in df.iterrows():
        for col in cols:
            to_paraphrase.append((idx, col, row[col]))

    prompts = [make_prompt(x[2]) for x in to_paraphrase]
    paraphrased_results = []
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), total=total_batches):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        outputs = generator(batch_prompts)
        for j, out in enumerate(outputs):
            text = out[0]["generated_text"]
            text = text.replace(batch_prompts[j], '').strip()
            paraphrased = text.split('\n')[0]
            if "출력:" in paraphrased:
                paraphrased = paraphrased.split("출력:")[-1].strip()
            paraphrased = paraphrased.lstrip('-').strip()
            paraphrased_results.append(paraphrased)

    # ----------------------
    # 2. 파라프레이즈된 증강 행 만들기 (단, 나중에 순서만 섞어서 쓸 것)
    # ----------------------
    result_idx = 0
    paraphrased_aug_rows = []
    for idx, row in df.iterrows():
        new_row = row.copy()
        new_row["ID"] = str(row["ID"]) + "_aug"
        for col in cols:
            new_row[col] = paraphrased_results[result_idx]
            result_idx += 1
        paraphrased_aug_rows.append(new_row)
    aug_df = pd.DataFrame(paraphrased_aug_rows)

    # ----------------------
    # 3. 순서 섞기 (파라프 행만)
    # ----------------------
    max_aug = 1  # 한 행 당 몇 개 생성할지
    reordered_rows = []
    for _, row in aug_df.iterrows():
        orig_perm = tuple([0, 1, 2, 3])
        all_perms = list(itertools.permutations(range(4)))
        hard_perms = [p for p in all_perms if 1 <= hamming_distance(orig_perm, p) <= 2]
        if not hard_perms:
            continue
        selected = random.sample(hard_perms, min(max_aug, len(hard_perms)))
        sentences = [row[f'sentence_{i}'] for i in range(4)]
        for idx2, perm in enumerate(selected, 1):
            aug_sentences = [sentences[i] for i in perm]
            aug_row = row.copy()
            for i in range(4):
                aug_row[f'sentence_{i}'] = aug_sentences[i]
                aug_row[f'answer_{i}'] = perm[i]
            aug_row['ID'] = f"{row['ID']}_{idx2}"
            reordered_rows.append(aug_row.copy())
    reordered_df = pd.DataFrame(reordered_rows)

    # ----------------------
    # 4. 원본 + (파라프+순서섞기)만 합치기
    # ----------------------
    reordered_df.to_csv("train_augmented_kor.csv", index=False)
    print("완료! → train_augmented_kor.csv")

if __name__ == "__main__":
    main()
    #cleanup()
