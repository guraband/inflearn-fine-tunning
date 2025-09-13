#!/usr/bin/env python3
"""
BERT Fine-tuning for IMDB Sentiment Analysis
Converted from 12_bert.ipynb

This script demonstrates how to fine-tune a DistilBERT model for sentiment analysis
using the IMDB dataset with a small sample of 50 examples.
"""

import os
import time
from typing import Dict, Any, Optional

from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# MPS 유틸리티 모듈 import
from mps_utils import warm_up_mps, check_mps_availability, get_optimal_device, print_device_info


def check_label_distribution(dataset, dataset_name="샘플"):
    """데이터셋의 레이블 분포를 확인합니다."""
    print(f"\n📊 {dataset_name} 레이블 분포 확인:")
    print("=" * 50)

    # 훈련 데이터 레이블 분포
    train_labels = [dataset["train"][i]["label"]
                    for i in range(len(dataset["train"]))]
    train_positive = sum(1 for label in train_labels if label == 1)
    train_negative = sum(1 for label in train_labels if label == 0)

    # 테스트 데이터 레이블 분포
    test_labels = [dataset["test"][i]["label"]
                   for i in range(len(dataset["test"]))]
    test_positive = sum(1 for label in test_labels if label == 1)
    test_negative = sum(1 for label in test_labels if label == 0)

    print(f"훈련 데이터: 긍정 {train_positive}개, 부정 {train_negative}개")
    print(f"테스트 데이터: 긍정 {test_positive}개, 부정 {test_negative}개")
    print(f"전체: 긍정 {train_positive +
          test_positive}개, 부정 {train_negative + test_negative}개")

    # 레이블 1인 데이터 찾기
    print(f"\n🔍 {dataset_name}에서 레이블 1(긍정) 데이터 확인:")
    positive_samples = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["label"] == 1:
            positive_samples.append(i)
            if len(positive_samples) >= 5:  # 처음 5개만 수집
                break

    if positive_samples:
        print(f"   {dataset_name}에서 레이블 1인 샘플 {len(positive_samples)}개 발견!")
        for idx, sample_idx in enumerate(positive_samples):
            sample = dataset["train"][sample_idx]
            print(f"   [{idx+1}] 샘플 {sample_idx}: {sample['text'][:80]}...")
    else:
        print(f"   ⚠️  {dataset_name}에서 레이블 1인 샘플을 찾을 수 없습니다!")

    return {
        'train_positive': train_positive,
        'train_negative': train_negative,
        'test_positive': test_positive,
        'test_negative': test_negative,
        'positive_samples': positive_samples
    }


def check_full_dataset_label_distribution(full_dataset, total_train, total_test, total_size):
    """전체 데이터셋의 레이블 분포를 확인합니다."""
    print("\n📊 전체 IMDB 데이터셋 레이블 분포:")
    print("=" * 60)

    # 전체 훈련 데이터 레이블 분포 (배치 처리로 메모리 효율성 향상)
    print("전체 훈련 데이터 레이블 분포 확인 중...")
    full_train_positive = 0
    full_train_negative = 0

    # 배치 단위로 처리 (1000개씩)
    batch_size = 1000
    for i in range(0, total_train, batch_size):
        end_idx = min(i + batch_size, total_train)
        batch_labels = [full_dataset["train"][j]["label"]
                        for j in range(i, end_idx)]
        full_train_positive += sum(1 for label in batch_labels if label == 1)
        full_train_negative += sum(1 for label in batch_labels if label == 0)

        if i % 5000 == 0:  # 5000개마다 진행상황 출력
            print(f"   진행률: {i}/{total_train} ({i/total_train*100:.1f}%)")

    # 전체 테스트 데이터 레이블 분포 (배치 처리)
    print("전체 테스트 데이터 레이블 분포 확인 중...")
    full_test_positive = 0
    full_test_negative = 0

    for i in range(0, total_test, batch_size):
        end_idx = min(i + batch_size, total_test)
        batch_labels = [full_dataset["test"][j]["label"]
                        for j in range(i, end_idx)]
        full_test_positive += sum(1 for label in batch_labels if label == 1)
        full_test_negative += sum(1 for label in batch_labels if label == 0)

        if i % 5000 == 0:  # 5000개마다 진행상황 출력
            print(f"   진행률: {i}/{total_test} ({i/total_test*100:.1f}%)")

    print(f"전체 훈련 데이터: 긍정 {full_train_positive:,}개 ({full_train_positive/total_train *
          100:.1f}%), 부정 {full_train_negative:,}개 ({full_train_negative/total_train*100:.1f}%)")
    print(f"전체 테스트 데이터: 긍정 {full_test_positive:,}개 ({full_test_positive/total_test *
          100:.1f}%), 부정 {full_test_negative:,}개 ({full_test_negative/total_test*100:.1f}%)")
    print(f"전체 데이터셋: 긍정 {full_train_positive + full_test_positive:,}개 ({(full_train_positive + full_test_positive)/total_size *
          100:.1f}%), 부정 {full_train_negative + full_test_negative:,}개 ({(full_train_negative + full_test_negative)/total_size*100:.1f}%)")

    # 레이블 1이 처음 등장하는 위치 찾기
    print(f"\n🔍 레이블 1(긍정)이 처음 등장하는 위치 확인:")

    # 훈련 데이터에서 레이블 1 찾기 (전체 범위에서 검색)
    first_positive_train = None
    print("   훈련 데이터에서 레이블 1 검색 중...")
    for i in range(total_train):  # 전체 훈련 데이터 검색
        if full_dataset["train"][i]["label"] == 1:
            first_positive_train = i
            break
        if i % 5000 == 0:  # 5000개마다 진행상황 출력
            print(f"     진행률: {i}/{total_train} ({i/total_train*100:.1f}%)")

    # 테스트 데이터에서 레이블 1 찾기 (전체 범위에서 검색)
    first_positive_test = None
    print("   테스트 데이터에서 레이블 1 검색 중...")
    for i in range(total_test):  # 전체 테스트 데이터 검색
        if full_dataset["test"][i]["label"] == 1:
            first_positive_test = i
            break
        if i % 5000 == 0:  # 5000개마다 진행상황 출력
            print(f"     진행률: {i}/{total_test} ({i/total_test*100:.1f}%)")

    # 결과 출력
    if first_positive_train is not None:
        sample = full_dataset["train"][first_positive_train]
        print(f"   ✅ 훈련 데이터에서 레이블 1이 처음 등장: {first_positive_train}번째")
        print(f"      내용: {sample['text'][:100]}...")
    else:
        print("   ❌ 훈련 데이터 전체에서 레이블 1을 찾을 수 없습니다!")

    if first_positive_test is not None:
        sample = full_dataset["test"][first_positive_test]
        print(f"   ✅ 테스트 데이터에서 레이블 1이 처음 등장: {first_positive_test}번째")
        print(f"      내용: {sample['text'][:100]}...")
    else:
        print("   ❌ 테스트 데이터 전체에서 레이블 1을 찾을 수 없습니다!")

    # 전체 데이터에서 레이블 1인 샘플 확인 (더 넓은 범위에서 검색)
    print(f"\n🔍 전체 데이터에서 레이블 1(긍정) 샘플 확인:")
    full_positive_samples = []

    # 훈련 데이터에서 검색 (처음 5000개)
    print("   훈련 데이터에서 긍정 샘플 검색 중...")
    for i in range(min(5000, total_train)):
        if full_dataset["train"][i]["label"] == 1:
            full_positive_samples.append(("train", i))
            if len(full_positive_samples) >= 3:
                break

    # 테스트 데이터에서도 검색 (처음 5000개)
    if len(full_positive_samples) < 3:
        print("   테스트 데이터에서 긍정 샘플 검색 중...")
        for i in range(min(5000, total_test)):
            if full_dataset["test"][i]["label"] == 1:
                full_positive_samples.append(("test", i))
                if len(full_positive_samples) >= 5:
                    break

    if full_positive_samples:
        print(f"   전체 데이터에서 레이블 1인 샘플 {len(full_positive_samples)}개 발견!")
        for idx, (split, sample_idx) in enumerate(full_positive_samples):
            sample = full_dataset[split][sample_idx]
            print(
                f"   [{idx+1}] {split} 데이터 샘플 {sample_idx}: {sample['text'][:80]}...")
    else:
        print("   ⚠️  전체 데이터에서 레이블 1인 샘플을 찾을 수 없습니다!")
        print("   이는 데이터셋에 문제가 있을 수 있습니다.")

    return {
        'train_positive': full_train_positive,
        'train_negative': full_train_negative,
        'test_positive': full_test_positive,
        'test_negative': full_test_negative,
        'positive_samples': full_positive_samples,
        'first_positive_train': first_positive_train,
        'first_positive_test': first_positive_test
    }


def get_sample_size(total_size):
    """사용자로부터 샘플 크기를 입력받습니다."""
    print(f"\n📊 전체 IMDB 데이터셋: {total_size:,}개")
    print("=" * 50)
    print("🎯 학습에 사용할 샘플 크기를 선택하세요:")
    print("   - 엔터: 기본값 1000개 사용")
    print("   - 0: 전체 데이터 사용")
    print("   - 숫자: 해당 개수만큼 샘플 사용")
    print("=" * 50)

    while True:
        try:
            user_input = input("샘플 크기 입력 (엔터=1000, 0=전체): ").strip()

            if user_input == "":
                return 1000  # 기본값
            elif user_input == "0":
                return total_size  # 전체 데이터
            else:
                sample_size = int(user_input)
                if sample_size < 0:
                    print("❌ 음수는 입력할 수 없습니다. 다시 입력하세요.")
                    continue
                elif sample_size > total_size:
                    print(
                        f"❌ 전체 데이터({total_size:,}개)보다 큰 수는 입력할 수 없습니다. 다시 입력하세요.")
                    continue
                return sample_size
        except ValueError:
            print("❌ 숫자를 입력하거나 엔터를 눌러주세요.")
            continue


def get_checkpoint_info():
    """체크포인트 관련 정보를 사용자로부터 입력받습니다."""
    print("\n💾 체크포인트 설정:")
    print("=" * 40)
    print("🎯 체크포인트 저장 방식을 선택하세요:")
    print("   - 엔터: 자동 저장 (에포크마다)")
    print("   - 0: 저장 안함")
    print("   - 숫자: N 에포크마다 저장")
    print("=" * 40)

    while True:
        try:
            user_input = input(
                "체크포인트 저장 주기 (엔터=자동, 0=저장안함, 숫자=N에포크마다): ").strip()

            if user_input == "":
                return "auto"  # 자동 저장
            elif user_input == "0":
                return "no"  # 저장 안함
            else:
                save_every = int(user_input)
                if save_every < 1:
                    print("❌ 1 이상의 숫자를 입력하세요.")
                    continue
                return save_every
        except ValueError:
            print("❌ 숫자를 입력하거나 엔터를 눌러주세요.")
            continue


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """가장 최근 체크포인트를 찾습니다."""
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            checkpoints.append(item)

    if not checkpoints:
        return None

    # 체크포인트 번호로 정렬
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoints[-1])

    print(f"🔍 발견된 체크포인트: {len(checkpoints)}개")
    print(f"   최신 체크포인트: {latest_checkpoint}")

    return latest_checkpoint


def ask_resume_training() -> bool:
    """이어서 학습할지 사용자에게 묻습니다."""
    print("\n🔄 이어서 학습하기:")
    print("=" * 30)
    print("기존 체크포인트가 발견되었습니다.")
    print("   - 엔터: 이어서 학습")
    print("   - n: 새로 시작")
    print("=" * 30)

    while True:
        user_input = input("이어서 학습하시겠습니까? (엔터=예, n=아니오): ").strip().lower()

        if user_input == "" or user_input == "y" or user_input == "yes":
            return True
        elif user_input == "n" or user_input == "no":
            return False
        else:
            print("❌ 엔터 또는 'n'을 입력하세요.")
            continue


def load_imdb_dataset(full_dataset, total_size, sample_size=None) -> DatasetDict:
    """IMDB 데이터셋을 지정된 크기로 샘플링하여 8:2로 분할합니다."""
    print("데이터셋 샘플링 시작...")
    start_time = time.time()

    # 사용자 입력으로 샘플 크기 결정
    if sample_size is None:
        sample_size = get_sample_size(total_size)

    print(f"\n📊 선택된 샘플 크기: {sample_size:,}개")
    if sample_size == total_size:
        print("   - 전체 데이터 사용")
    else:
        print(f"   - 전체의 {sample_size/total_size*100:.2f}% 사용")

    # tqdm을 사용한 더 자세한 프로그레스바
    with tqdm(total=100, desc="샘플 데이터 처리", unit="%",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

        if sample_size == total_size:
            # 전체 데이터 사용
            dataset = full_dataset["train"].train_test_split(test_size=0.2)
        else:
            # 지정된 크기만큼 샘플링
            dataset = full_dataset["train"].select(
                range(sample_size)).train_test_split(test_size=0.2)

        # 프로그레스바 완료
        pbar.n = 100
        pbar.refresh()

    end_time = time.time()
    print(f"✅ 샘플 데이터 처리 완료! 소요시간: {end_time - start_time:.2f}초")
    print(f"📁 캐시 위치: ./data_cache")
    print(f"📊 사용된 샘플: {len(dataset['train']) + len(dataset['test'])}개")
    print(f"📊 훈련 샘플: {len(dataset['train'])}개, 테스트 샘플: {
          len(dataset['test'])}개")

    # 샘플 데이터 레이블 분포 확인
    sample_distribution = check_label_distribution(dataset, "샘플")

    # 데이터 샘플 확인 (처음 10개)
    print("\n📋 데이터 샘플 확인 (처음 10개):")
    print("=" * 80)
    for i in range(min(10, len(dataset["train"]))):
        sample = dataset["train"][i]
        print(f"\n[{i+1}] 리뷰 내용: {sample['text'][:100]}...")  # 처음 100자만 표시
        print(f"    레이블 (0:부정, 1:긍정): {sample['label']}")
        print(f"    텍스트 길이: {len(sample['text'])}자")

    print(f"\n데이터셋 분할: {list(dataset.keys())}")  # ['train', 'test']

    return dataset


def load_full_dataset_info():
    """전체 IMDB 데이터셋 정보를 로드합니다."""
    print("🔍 전체 IMDB 데이터셋 크기 확인 중...")
    full_dataset = load_dataset("imdb", cache_dir="./data_cache")
    total_train = len(full_dataset["train"])
    total_test = len(full_dataset["test"])
    total_size = total_train + total_test

    print(f"📊 전체 IMDB 데이터셋: {total_size:,}개")
    print(f"   - 훈련 데이터: {total_train:,}개")
    print(f"   - 테스트 데이터: {total_test:,}개")

    return full_dataset, total_train, total_test, total_size


def load_model_and_tokenizer():
    """모델과 토크나이저를 로드합니다."""
    print("\n=== 모델 및 토크나이저 로딩 ===")

    # MPS warm-up 먼저 수행
    warm_up_mps()

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


def setup_trainer(model, dataset: DatasetDict, checkpoint_config="auto", resume_from_checkpoint=None) -> Trainer:
    """훈련 설정 및 Trainer를 구성합니다."""
    print("\n=== 훈련 설정 구성 ===")

    def compute_metrics(eval_pred):
        """평가 예측 결과의 정확도를 계산합니다."""
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc}

    # 데이터셋 크기에 따른 동적 설정
    train_size = len(dataset["train"])
    total_samples = train_size + len(dataset["test"])

    print(f"📊 데이터셋 크기: {total_samples:,}개 (훈련: {train_size:,}개)")

    # 샘플 크기에 따른 설정 조정
    if total_samples <= 1000:
        # 소규모 데이터셋 (1000개 이하)
        batch_size = 16
        num_epochs = 15
        logging_steps = 5
        print("   - 소규모 데이터셋 설정 적용")
    elif total_samples <= 10000:
        # 중간 규모 데이터셋 (1000-10000개)
        batch_size = 32
        num_epochs = 10
        logging_steps = 10
        print("   - 중간 규모 데이터셋 설정 적용")
    else:
        # 대규모 데이터셋 (10000개 이상)
        batch_size = 64
        num_epochs = 5
        logging_steps = 50
        print("   - 대규모 데이터셋 설정 적용")

    # 체크포인트 설정
    output_dir = "checkpoints"
    save_strategy = "no"
    save_steps = None

    if checkpoint_config == "auto":
        save_strategy = "epoch"
        print("   - 체크포인트: 에포크마다 자동 저장")
    elif checkpoint_config == "no":
        print("   - 체크포인트: 저장 안함")
    elif isinstance(checkpoint_config, int):
        save_strategy = "steps"
        save_steps = checkpoint_config * \
            (train_size // batch_size)  # 에포크당 스텝 수 계산
        print(f"   - 체크포인트: {checkpoint_config} 에포크마다 저장 (매 {save_steps} 스텝)")

    # M4 Pro에 최적화된 설정
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        report_to="none",  # 외부 로깅툴 비활성화
        logging_steps=logging_steps,
        dataloader_pin_memory=False,  # MPS에서는 pin_memory 비활성화
        dataloader_num_workers=0,     # MPS에서는 멀티프로세싱 비활성화
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy="no",           # 평가 비활성화로 속도 향상
        learning_rate=2e-5,           # 학습률 명시적 설정
        weight_decay=0.01,            # 가중치 감쇠 추가
        load_best_model_at_end=False,  # 최고 모델 로드 비활성화
        save_total_limit=3,           # 최대 3개 체크포인트만 유지
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # 정확도 계산 함수 추가
    )

    # 체크포인트에서 이어서 학습
    if resume_from_checkpoint:
        print(f"🔄 체크포인트에서 이어서 학습: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return trainer

    print("✅ 훈련 설정 완료!")
    print(f"🚀 M4 Pro 최적화 설정 적용됨! (배치크기: {batch_size}, 에포크: {num_epochs})")
    return trainer


def train_model(trainer: Trainer) -> Dict[str, Any]:
    """모델을 훈련합니다."""
    print("\n=== 모델 훈련 ===")
    print("훈련 시작...")

    train_result = trainer.train()

    print("훈련 완료!")
    print(f"훈련 결과: {train_result}")

    return train_result


def save_final_model(trainer: Trainer, model_name: str = "fine_tuned_bert"):
    """파인튜닝된 모델을 로컬에 저장합니다."""
    print(f"\n💾 파인튜닝된 모델 저장 중...")

    # 모델 저장 디렉토리 생성
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # 모델과 토크나이저 저장
    trainer.save_model(model_dir)
    trainer.tokenizer.save_pretrained(model_dir)

    print(f"✅ 모델 저장 완료!")
    print(f"   📁 저장 위치: {model_dir}/")
    print(f"   📄 파일들:")
    print(f"      - config.json (모델 설정)")
    print(f"      - model.safetensors (모델 가중치)")
    print(f"      - tokenizer.json (토크나이저)")
    print(f"      - tokenizer_config.json (토크나이저 설정)")
    print(f"      - vocab.txt (어휘 사전)")

    return model_dir


def load_saved_model(model_path: str):
    """저장된 모델을 로드합니다."""
    print(f"📂 저장된 모델 로드 중: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print("✅ 모델 로드 완료!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None


def test_predictions(tokenizer, model):
    """학습된 모델로 예측을 수행합니다."""
    print("\n=== 예측 테스트 ===")

    # 최적 디바이스 확인
    device = get_optimal_device()
    print(f"사용 디바이스: {device}")

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
        # 입력된 문장을 토큰화하여 적절한 디바이스에 전달
        inputs = tokenizer(text, return_tensors="pt").to(device)

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

    # 0. 디바이스 정보 출력
    print_device_info()

    # 1. 전체 데이터셋 정보 로드
    full_dataset, total_train, total_test, total_size = load_full_dataset_info()

    # 2. 전체 데이터셋 레이블 분포 확인
    # full_distribution = check_full_dataset_label_distribution(
    #     full_dataset, total_train, total_test, total_size)

    # 3. 사용자 입력으로 샘플 데이터 로드
    dataset = load_imdb_dataset(full_dataset, total_size)

    # 4. 체크포인트 설정
    checkpoint_config = get_checkpoint_info()

    # 5. 기존 체크포인트 확인
    output_dir = "checkpoints"
    latest_checkpoint = find_latest_checkpoint(output_dir)
    resume_from_checkpoint = None

    if latest_checkpoint:
        if ask_resume_training():
            resume_from_checkpoint = latest_checkpoint
        else:
            print("🔄 새로 시작합니다. 기존 체크포인트는 무시됩니다.")

    # 6. 모델 및 토크나이저 로드
    tokenizer, model = load_model_and_tokenizer()

    # 7. 데이터 전처리
    dataset = preprocess_data(dataset, tokenizer)

    # 8. 훈련 설정 (체크포인트 포함)
    trainer = setup_trainer(
        model, dataset, checkpoint_config, resume_from_checkpoint)

    # 9. 모델 훈련 (이미 체크포인트에서 이어서 학습했다면 건너뜀)
    if not resume_from_checkpoint:
        train_result = train_model(trainer)
    else:
        print("✅ 체크포인트에서 이어서 학습이 완료되었습니다!")

    # 10. 파인튜닝된 모델 저장
    model_name = f"bert_imdb_{
        len(dataset['train']) + len(dataset['test'])}samples"
    saved_model_dir = save_final_model(trainer, model_name)

    # 11. 예측 테스트
    test_predictions(tokenizer, model)

    print("\n" + "=" * 60)
    print("🎉 스크립트 실행 완료!")
    if checkpoint_config != "no":
        print(f"💾 체크포인트 저장 위치: {output_dir}/")
        print("   다음 실행 시 이어서 학습할 수 있습니다.")
    print(f"💾 최종 모델 저장 위치: {saved_model_dir}/")
    print("   이 모델을 다른 프로젝트에서 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
