import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm
import random, gc
import numpy as np
import re, os, sys, itertools, contextlib
os.environ["LLAMA_LOG_LEVEL"] = "WARN"
os.environ["LLAMA_DEBUG_OFF"] = "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

class DevNull:
    def write(self, _): pass
    def flush(self): pass

@contextlib.contextmanager
def suppress_llama_log():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = DevNull()
    sys.stderr = DevNull()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


llm = Llama(
    model_path="./gemma-3-27b-it-q4_0.gguf",  # 실제 파일 경로/이름 확인!
    n_ctx=2048,
    n_gpu_layers=-1,
    seed=SEED
)

def make_doc_prompt(sentences):
    system_prompt = "당신은 친절하고 유능한 한국어 문서 페러프레이저입니다."
    user_prompt = (
        "아래 4개의 문장은 한 문서의 올바른(논리적) 순서입니다. "
        "각 문장을 의미를 바꾸지 않고 다양한 한국어 표현으로 바꿔서 다시 써 주세요. "
        "각 문장은 번호와 함께 출력해 주세요.\n" +
        "문장1: " + sentences[0] + "\n" +
        "문장2: " + sentences[1] + "\n" +
        "문장3: " + sentences[2] + "\n" +
        "문장4: " + sentences[3]
    )
    full_prompt = (
        "<|start_of_turn|>system\n" + system_prompt + "<|end_of_turn|>\n"
        "<|start_of_turn|>user\n" + user_prompt + "<|end_of_turn|>\n"
        "<|start_of_turn|>model\n"
    )
    return full_prompt

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def extract_4_sentences(text):
    pattern = r"문장1[:：]\s*(.*?)\n문장2[:：]\s*(.*?)\n문장3[:：]\s*(.*?)\n문장4[:：]\s*(.*)"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return [m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()]
    # 백업: 번호 없는 줄 단위 분리
    return [t.strip() for t in text.strip().split('\n') if t.strip()][:4]

def cleanup():
    print("모델, 캐시, VRAM 비우는 중...")
    global llm
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    print("메모리, 캐시 정리 완료")

def main():
    df = pd.read_csv('train.csv')#.head(3)
    cols = [c for c in df.columns if c.startswith("sentence_")]
    ans_cols = [c for c in df.columns if c.startswith("answer_")]

    paraphrased_aug_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Doc-paraphrasing-sorted"):
        # 논리적(올바른) 순서로 문장 리스트 만들기
        # answer_i는 sentence 컬럼 중 "몇 번째가 논리적 순서인가"를 나타냄
        ordered_sentences = [None]*4
        for i in range(4):
            answer_idx = int(row[f"answer_{i}"])
            ordered_sentences[i] = row[f"sentence_{answer_idx}"]
        prompt = make_doc_prompt(ordered_sentences)
        output = llm(prompt, max_tokens=220, temperature=1.0, top_p=0.92, repeat_penalty=1.15)
        text = output["choices"][0]["text"].strip()
        if "<|end_of_turn|>" in text:
            text = text.split("<|end_of_turn|>")[-1].strip()
        new_sentences = extract_4_sentences(text)
        if len(new_sentences) != 4:
            new_sentences = ordered_sentences
        # 순서를 랜덤으로 섞음(기존 알고리즘 활용, hamming distance>=1)
        orig_perm = tuple([0, 1, 2, 3])
        all_perms = list(itertools.permutations(range(4)))
        hard_perms = [p for p in all_perms if 1 <= hamming_distance(orig_perm, p) <= 2]
        if not hard_perms:
            perm = orig_perm
        else:
            perm = random.choice(hard_perms)
        shuffled_sentences = [new_sentences[i] for i in perm]
        # shuffled_sentences를 sentence_0~3에 저장, answer_0~3은 실제 논리적 순서 인덱스
        new_row = row.copy()
        new_row["ID"] = str(row["ID"]) + "_aug"
        for i, sent in enumerate(shuffled_sentences):
            new_row[f"sentence_{i}"] = sent
            new_row[f"answer_{i}"] = perm[i]  # 실제 논리적 순서 인덱스
        paraphrased_aug_rows.append(new_row)
    aug_df = pd.DataFrame(paraphrased_aug_rows)
    aug_df.to_csv("train_augmented_gemma.csv", index=False)
    print("완료! → train_augmented_gemma.csv")

if __name__ == "__main__":
    main()
    cleanup()
