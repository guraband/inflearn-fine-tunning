#!/usr/bin/env python3
"""
BERT Fine-tuning for IMDB Sentiment Analysis
Converted from 12_bert.ipynb

This script demonstrates how to fine-tune a DistilBERT model for sentiment analysis
using the IMDB dataset with a small sample of 50 examples.
"""

import time
from typing import Dict, Any

from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_imdb_dataset() -> DatasetDict:
    """IMDB 데이터셋을 로드하고 8:2로 분할합니다."""
    print("데이터셋 로딩 시작...")
    start_time = time.time()

    print("📥 IMDB 데이터셋을 로딩하고 있습니다...")
    print("   (처음 실행시 인터넷에서 다운로드하므로 시간이 걸릴 수 있습니다)")
    print("   (다음 실행부터는 로컬 캐시에서 빠르게 로딩됩니다)")

    # tqdm을 사용한 더 자세한 프로그레스바
    with tqdm(total=100, desc="다운로드 진행률", unit="%",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

        # 데이터셋 로딩 (50개 샘플) - 로컬 캐시 사용
        dataset = load_dataset(
            "imdb",
            split="train[:50]",
            cache_dir="./data_cache"  # 로컬 캐시 디렉토리 지정
        ).train_test_split(test_size=0.2)

        # 프로그레스바 완료
        pbar.n = 100
        pbar.refresh()

    end_time = time.time()
    print(f"✅ 데이터셋 로딩 완료! 소요시간: {end_time - start_time:.2f}초")
    print(f"📁 캐시 위치: ./data_cache")
    print(f"📊 전체 데이터셋: {len(dataset['train']) + len(dataset['test'])}개")
    print(f"📊 훈련 데이터: {len(dataset['train'])
                       }개, 테스트 데이터: {len(dataset['test'])}개")

    # 데이터 샘플 확인
    sample = dataset["train"][5]
    print(f"리뷰 내용: {sample['text']}")
    print(f"레이블 (0:부정, 1:긍정): {sample['label']}")
    print(f"데이터셋 크기: {len(dataset)}")

    return dataset


def load_model_and_tokenizer():
    """모델과 토크나이저를 로드합니다."""
    print("\n=== 모델 및 토크나이저 로딩 ===")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")

    print("✅ 모델 및 토크나이저 로딩 완료!")
    return tokenizer, model


def preprocess_data(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """데이터를 전처리합니다."""
    print("\n=== 데이터 전처리 ===")

    def tokenize(batch):
        """텍스트를 토큰화하고 패딩을 적용합니다."""
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    print("✅ 데이터 전처리 완료!")
    return dataset


def setup_trainer(model, dataset: DatasetDict) -> Trainer:
    """훈련 설정 및 Trainer를 구성합니다."""
    print("\n=== 훈련 설정 구성 ===")

    def compute_metrics(eval_pred):
        """평가 예측 결과의 정확도를 계산합니다."""
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir="test",
        per_device_train_batch_size=8,
        num_train_epochs=15,
        report_to="none",  # 외부 로깅툴 비활성화
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # 정확도 계산 함수 추가
    )

    print("✅ 훈련 설정 완료!")
    return trainer


def train_model(trainer: Trainer) -> Dict[str, Any]:
    """모델을 훈련합니다."""
    print("\n=== 모델 훈련 ===")
    print("훈련 시작...")

    train_result = trainer.train()

    print("훈련 완료!")
    print(f"훈련 결과: {train_result}")

    return train_result


def test_predictions(tokenizer, model):
    """학습된 모델로 예측을 수행합니다."""
    print("\n=== 예측 테스트 ===")

    # 테스트 문장들
    test_texts = [
        "I would put this at the top of the list of films in the category of unwatchable trash.",
        "I can watch this all day.",
        "This movie is absolutely fantastic!",
        "Terrible acting and boring plot."
    ]

    for text in test_texts:
        print(f"\n텍스트: {text}")

        # "pt": pytorch 형식으로 변환
        # 입력된 문장을 토큰화하여 mps에 전달
        inputs = tokenizer(text, return_tensors="pt").to("mps")

        outputs = model(**inputs)
        # 가장 높은 점수의 인덱스를 예측값으로 사용
        predictions = outputs.logits.argmax(dim=-1)

        print(f"예측 레이블: {predictions[0]}")
        print(f"예측 확률: {outputs.logits[0][predictions[0]].item()}")
        print("긍정" if predictions[0] == 1 else "부정")

        # 로짓 값 출력
        print(f"로짓 값: {outputs.logits[0]}")


def main():
    """메인 실행 함수"""
    print("🚀 BERT Fine-tuning for IMDB Sentiment Analysis 시작!")
    print("=" * 60)

    # 1. 데이터 로드
    dataset = load_imdb_dataset()

    # 2. 모델 및 토크나이저 로드
    tokenizer, model = load_model_and_tokenizer()

    # 3. 데이터 전처리
    dataset = preprocess_data(dataset, tokenizer)

    # 4. 훈련 설정
    trainer = setup_trainer(model, dataset)

    # 5. 모델 훈련
    train_result = train_model(trainer)

    # 6. 예측 테스트
    test_predictions(tokenizer, model)

    print("\n" + "=" * 60)
    print("🎉 스크립트 실행 완료!")


if __name__ == "__main__":
    main()
