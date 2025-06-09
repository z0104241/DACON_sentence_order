import pandas as pd

def postprocess(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # 1. 필요한 컬럼만 남기기 (ID, answer_0 ~ answer_3)
    columns_needed = ['ID', 'answer_0', 'answer_1', 'answer_2', 'answer_3']
    df = df[columns_needed]

    # 2. answer_* 컬럼의 값이 0.0, 1.0 등 float형인 경우 int→str 변환
    for col in ['answer_0', 'answer_1', 'answer_2', 'answer_3']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else '')

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"💾 후처리 완료: {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='predictions.csv')
    parser.add_argument('--output', '-o', default='submission.csv')
    args = parser.parse_args()
    postprocess(args.input, args.output)
