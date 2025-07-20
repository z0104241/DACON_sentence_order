import os
import subprocess
import argparse
import sys

AUGMENT_SCRIPT = "augment_gemma.py"
TRAIN_SCRIPT = "train.py"
INFERENCE_SCRIPT = "inference.py"

REQUIRED_FILES = {
    "augment": ["train.csv"],
    "train": ["train.csv", "train_augmented_gemma.csv"],
    "inference": ["test.csv", os.path.join("gemma3_model", "adapter_config.json")] # 학습된 모델이 있는지 확인
}

def check_files(file_list):
    """필요한 파일들이 존재하는지 확인하는 함수"""
    missing_files = [f for f in file_list if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 오류: 다음 필수 파일이 없습니다: {', '.join(missing_files)}", file=sys.stderr)
        print("💡 이전 단계를 실행했는지 또는 파일 경로가 올바른지 확인하세요.", file=sys.stderr)
        return False
    return True

def run_script(script_name, required_files):
    """스크립트를 실행하고 예외를 처리하는 함수"""
    print(f"\n▶️  '{script_name}' 스크립트 실행을 시작합니다...")

    # 1. 필수 파일 확인
    if not check_files(required_files):
        return

    # 2. 스크립트 실행
    try:
        # subprocess.run을 사용하여 스크립트를 별도 프로세스로 실행합니다.
        # check=True는 스크립트 실행 중 오류 발생 시 예외를 발생시킵니다.
        process = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            encoding='utf-8'
        )
        print(f"✅ '{script_name}' 스크립트 실행이 성공적으로 완료되었습니다.")

    except FileNotFoundError:
        print(f"❌ 오류: '{script_name}' 파일을 찾을 수 없습니다.", file=sys.stderr)
        print("💡 파일 이름이 올바른지, 파일이 현재 폴더에 있는지 확인하세요.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류: '{script_name}' 실행 중 오류가 발생했습니다.", file=sys.stderr)
        print(f"   - 종료 코드: {e.returncode}", file=sys.stderr)
        print("💡 스크립트의 오류 메시지를 확인하여 문제를 해결하세요.", file=sys.stderr)
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}", file=sys.stderr)
        print("💡 스크립트 코드나 환경 설정을 확인하세요.", file=sys.stderr)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="DACON 문장 순서 맞추기 프로젝트 실행기",
        formatter_class=argparse.RawTextHelpFormatter # 도움말 줄바꿈 지원
    )

    # 실행할 작업을 선택하는 인자 추가
    parser.add_argument(
        "action",
        choices=["augment", "train", "inference", "all"],
        help="""실행할 작업을 선택하세요:
  - augment: 데이터 증강 (train.csv -> train_augmented_gemma.csv)
  - train: 모델 훈련 (train.csv, train_augmented_gemma.csv 사용)
  - inference: 추론 (test.csv와 훈련된 모델 사용)
  - all: 위 세 단계를 순서대로 모두 실행
"""
    )

    args = parser.parse_args()

    # 선택된 작업에 따라 해당 함수 실행
    if args.action == "augment" or args.action == "all":
        run_script(AUGMENT_SCRIPT, REQUIRED_FILES["augment"])

    if args.action == "train" or args.action == "all":
        run_script(TRAIN_SCRIPT, REQUIRED_FILES["train"])

    if args.action == "inference" or args.action == "all":
        run_script(INFERENCE_SCRIPT, REQUIRED_FILES["inference"])

if __name__ == "__main__":
    main()