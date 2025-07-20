# 📝 DACON 문장 순서 배열 AI 경진대회 솔루션 (최종 7위)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Libs-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

이 저장소는 DACON에서 주최한 '[문장 순서 배열 AI 경진대회](https://dacon.io/competitions/official/236219/overview/description)'에서 **최종 7위**를 달성한 솔루션 코드를 담고 있습니다. **Gemma-3-12B-it** 모델을 기반으로, 주어진 문장들의 논리적 순서를 예측하는 모델의 개발 전 과정을 기록합니다.

<br>

## 💡 핵심 전략 및 특징 (Key Features)

-   **효율적 미세조정(PEFT)**: **QLoRA** 기법을 적용하여, A5000 (24GB) 단일 GPU 환경에서도 12B급 거대 언어 모델을 안정적으로 파인튜닝했습니다.
-   **데이터 증강(Data Augmentation)**: **Gemma-3-27B** 모델을 활용한 **패러프레이징(Paraphrasing)** 기법으로 학습 데이터의 양과 질을 높여 모델의 일반화 성능을 극대화했습니다.
-   **자동화 파이프라인**: 데이터 증강부터 훈련, 추론까지 이어지는 전 과정을 `main.py` 스크립트 하나로 제어할 수 있도록 설계하여 실험의 재현성과 효율성을 확보했습니다.

<br>

## 🚀 성능 개선 과정 (Experiments & Results)

초기 베이스라인 모델(`Qwen-8B`)의 정확도 **0.86**에서 시작하여, 다양한 가설을 검증하는 실험을 통해 최종 **0.874**의 성능을 달성했습니다.

| 단계 | 실험 내용 | 모델 / 방법 | 정확도 | 비고 |
| :--: | :--- | :--- | :---: | :--- |
| **1** | **베이스라인** | `Qwen-8B` + 기본 파인튜닝 | **0.86** | 성능 개선의 기준점 |
| **2** | 순열 증강 | + Permutation Augmentation | +0.05%p | 데이터의 양과 다양성 확보 |
| **3** | 모델 스케일링 | `Gemma-32B`로 변경 | 변화 없음 | 8B로 충분하다고 판단 |
| **4** | 패러프레이징 (1) | + Korean Llama 8B | **+0.1%p** | LLM을 이용한 증강의 효과 확인 |
| **5** | **패러프레이징 (2)** | **+ Gemma-3-27B** | **+0.1%p** | **가장 높은 성능 향상, 최종 채택** |
| **6** | 하이퍼파라미터 튜닝 | LoRA `r` 값 등 조정 | +0.04%p | 모델 세부 최적화 |
| **7** | 최종 모델 선정 | `Gemma-3-12B-it`로 변경 | 변화 없음 | 안정성 및 효율성 고려 |

> **최종 결론**: 이 과업에서는 모델의 크기를 무작정 키우는 것보다, **양질의 데이터를 증강하여 학습시키는 것**이 성능 향상에 가장 결정적인 요소임을 실험적으로 증명했습니다.

<br>

## ⚙️ 프로젝트 구조 및 파이프라인

프로젝트는 3단계의 파이프라인으로 구성되며, `main.py`를 통해 각 단계를 제어할 수 있습니다.

```
                  ┌────────────────────────┐
                  │       train.csv        │
                  └───────────┬──────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ 1. augment_gemma.py    │  (LLM Paraphrasing)
                  └───────────┬──────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ train_augmented_gemma.csv │
                  └───────────┬──────────┘
                              │
                              ▼
┌─────────────────┐   ┌─────────────────┐
│    train.csv    │ + │ train_aug.csv   ├─► 2. train.py (QLoRA) ──► gemma3_model
└─────────────────┘   └─────────────────┘
                                                                        │
                                                                        ▼
┌─────────────────┐   ┌─────────────────┐
│     test.csv    │ + │  gemma3_model   ├─► 3. inference.py ─────► predictions.csv
└─────────────────┘   └─────────────────┘
```

<br>

## 🛠️ 기술 스택 (Tech Stack)

-   **Base Model**: `google/gemma-3-12b-it`
-   **Augmentation Model**: `google/gemma-3-27b-it` (GGUF)
-   **Core Libraries**: `PyTorch`, `transformers`, `PEFT`, `TRL`, `bitsandbytes`
-   **Data Handling**: `pandas`, `datasets`, `scikit-learn`
-   **Augmentation Engine**: `llama-cpp-python`

<br>

## 🎬 시작하기 (Getting Started)

### 1. 환경 설정

```bash
# 1. 저장소 복제
git clone [https://github.com/z0104241/DACON_sentence_order.git](https://github.com/z0104241/DACON_sentence_order.git)
cd DACON_sentence_order

# 2. 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. 스크립트 실행

`main.py`를 사용하여 원하는 단계를 실행할 수 있습니다.

```bash
# 데이터 증강만 실행
python main.py augment

# 모델 훈련만 실행
python main.py train

# 추론만 실행
python main.py inference

# 모든 과정을 순서대로 실행
python main.py all
