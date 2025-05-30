import pandas as pd
import itertools
from tqdm import tqdm

def get_natural_order(permuted, original):
    """
    permuted: 섞인 문장 리스트
    original: 원본 문장 리스트
    return: permuted를 original의 자연스러운 순서로 정렬하려면, permuted의 각 인덱스를 어떤 순서로 재배열해야 하는가
    예: original = [A,B,C,D], permuted = [B,A,C,D] -> 정답은 [1,0,2,3]
    """
    # 각 원본 문장이 permuted에서 어디에 있는지 찾는다
    order = [permuted.index(sent) for sent in original]
    return order

def main():
    INPUT_CSV = "train.csv"
    OUTPUT_CSV = "train_augmented.csv"

    df = pd.read_csv(INPUT_CSV)
    augmented_rows = []

    perms = list(itertools.permutations([0, 1, 2, 3]))
    orig_perm = (0, 1, 2, 3)

    # 가장 다른 3개 permutation (원본 제외)
    distances = [(p, sum([x != y for x, y in zip(orig_perm, p)])) for p in perms if p != orig_perm]
    distances_sorted = sorted(distances, key=lambda x: -x[1])
    top3_perms = [p for p, d in distances_sorted[:3]]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_sentences = [row[f"sentence_{i}"] for i in range(4)]
        for perm_num, perm in enumerate(top3_perms):
            permuted_sentences = [orig_sentences[i] for i in perm]
            # label: permuted_sentences를 원본 순서로 정렬하는 인덱스
            label = get_natural_order(permuted_sentences, orig_sentences)
            new_row = {
                "ID": f"AUG_{row['ID']}_{perm_num}",
                "sentence_0": permuted_sentences[0],
                "sentence_1": permuted_sentences[1],
                "sentence_2": permuted_sentences[2],
                "sentence_3": permuted_sentences[3],
                "answer_0": label[0],
                "answer_1": label[1],
                "answer_2": label[2],
                "answer_3": label[3]
            }
            augmented_rows.append(new_row)

    # 원본 데이터도 함께 저장
    aug_df = pd.DataFrame(augmented_rows)
    augmented = pd.concat([df, aug_df], ignore_index=True)
    augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"증강 데이터 저장 완료: {OUTPUT_CSV} (총 {len(augmented)}개 샘플)")

if __name__ == "__main__":
    main()
