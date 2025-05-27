#!/usr/bin/env python3
"""
train.csv 데이터에 대해
- 각 샘플별로 원본 순서와 Hamming distance가 가장 큰 3가지 순열을 골라
- 해당 순열대로 sentence와 label을 재구성하여 데이터 증강
- 최종적으로 train_augmented.csv 저장
"""

import pandas as pd
import itertools
from tqdm import tqdm

def hamming_distance(a, b):
    """두 permutation의 각 자리별 차이 개수(Hamming Distance)"""
    return sum([x != y for x, y in zip(a, b)])

def main():
    # 파일명은 필요에 맞게 수정 가능
    INPUT_CSV = "train.csv"
    OUTPUT_CSV = "train_augmented.csv"

    df = pd.read_csv(INPUT_CSV)
    augmented_rows = []

    # 4개 문장 순열(0~3)
    perms = list(itertools.permutations([0, 1, 2, 3]))
    orig_perm = (0, 1, 2, 3)

    # 원본 순서와 가장 먼 순열(=Hamming distance 최댓값) 3개 선택
    distances = [(p, hamming_distance(orig_perm, p)) for p in perms if p != orig_perm]
    distances_sorted = sorted(distances, key=lambda x: -x[1])
    top3_perms = [p for p, d in distances_sorted[:3]]

    print("선택된 최상위 3개 순열:", top3_perms)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_sentences = [row[f"sentence_{i}"] for i in range(4)]
        orig_order = [row[f"answer_{i}"] for i in range(4)]
        for perm_num, perm in enumerate(top3_perms):
            permuted_sentences = [orig_sentences[i] for i in perm]
            # label: permuted_sentences[i]의 원래 index는?
            new_label = [orig_order[perm[i]] for i in range(4)]
            new_row = {
                "ID": f"AUG_{row['ID']}_{perm_num}",
                "sentence_0": permuted_sentences[0],
                "sentence_1": permuted_sentences[1],
                "sentence_2": permuted_sentences[2],
                "sentence_3": permuted_sentences[3],
                "answer_0": new_label[0],
                "answer_1": new_label[1],
                "answer_2": new_label[2],
                "answer_3": new_label[3]
            }
            augmented_rows.append(new_row)

    # 증강 데이터와 원본을 합쳐서 저장
    aug_df = pd.DataFrame(augmented_rows)
    augmented = pd.concat([df, aug_df], ignore_index=True)
    augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"증강 데이터 저장 완료: {OUTPUT_CSV} (총 {len(augmented)}개 샘플)")

if __name__ == "__main__":
    main()
