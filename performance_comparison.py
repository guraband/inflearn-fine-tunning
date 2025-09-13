#!/usr/bin/env python3
"""
성능 비교 스크립트
원본 버전과 M4 Pro 최적화 버전의 성능을 비교합니다.
"""

import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from mps_utils import warm_up_mps, get_optimal_device


def benchmark_model_loading():
    """모델 로딩 성능 비교"""
    print("🔍 모델 로딩 성능 비교")
    print("=" * 50)

    # MPS warm-up
    warm_up_mps()
    device = get_optimal_device()

    # 원본 버전 (기본 설정)
    print("\n📊 원본 버전 (기본 설정):")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")
    model = model.to(device)

    end_time = time.time()
    print(f"   로딩 시간: {end_time - start_time:.2f}초")
    print(f"   모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   모델 크기: {sum(p.numel() * p.element_size()
          for p in model.parameters()) / 1024**2:.2f}MB")

    return model, tokenizer


def benchmark_inference_speed(model, tokenizer, device):
    """추론 속도 벤치마크"""
    print("\n🚀 추론 속도 벤치마크")
    print("=" * 50)

    # 테스트 문장들
    test_texts = [
        "This movie is absolutely fantastic!",
        "I would put this at the top of the list of films in the category of unwatchable trash.",
        "I can watch this all day.",
        "Terrible acting and boring plot.",
        "The best movie I have ever seen in my entire life.",
        "Waste of time and money.",
        "Outstanding performance by all actors.",
        "Boring and predictable storyline.",
        "A masterpiece of cinema.",
        "Complete garbage, don't waste your time."
    ] * 10  # 100개 문장으로 확장

    print(f"테스트 문장 수: {len(test_texts)}개")

    # 단일 문장 추론 시간 측정
    single_times = []
    for i, text in enumerate(test_texts[:10]):  # 처음 10개만 측정
        start_time = time.time()

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        end_time = time.time()
        single_times.append(end_time - start_time)

        if i < 3:  # 처음 3개만 출력
            print(f"   문장 {i+1}: {end_time - start_time:.4f}초")

    avg_single_time = sum(single_times) / len(single_times)
    print(f"   평균 단일 문장 추론 시간: {avg_single_time:.4f}초")

    # 배치 추론 시간 측정
    print(f"\n📊 배치 추론 성능 테스트:")
    batch_sizes = [1, 4, 8, 16, 32]

    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue

        batch_texts = test_texts[:batch_size]

        # 배치 토큰화
        start_time = time.time()
        inputs = tokenizer(batch_texts, return_tensors="pt",
                           padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        end_time = time.time()
        total_time = end_time - start_time
        per_sample_time = total_time / batch_size

        print(f"   배치 크기 {batch_size:2d}: {
              total_time:.4f}초 (문장당 {per_sample_time:.4f}초)")


def benchmark_memory_usage():
    """메모리 사용량 벤치마크"""
    print("\n💾 메모리 사용량 벤치마크")
    print("=" * 50)

    device = get_optimal_device()

    if device == "mps":
        # MPS 메모리 정보
        print("MPS 메모리 정보:")
        print(f"   디바이스: {device}")

        # 메모리 사용량 측정을 위한 더미 텐서 생성
        memory_usage = []
        tensor_sizes = [1000, 2000, 4000, 8000, 16000]

        for size in tensor_sizes:
            try:
                # 더미 텐서 생성
                dummy_tensor = torch.rand(size, size).to(device)
                memory_usage.append(size * size * 4)  # 4바이트 per float32
                print(f"   {size}x{size} 텐서: {size*size*4/1024**2:.2f}MB")
            except RuntimeError as e:
                print(f"   {size}x{size} 텐서: 메모리 부족 - {e}")
                break
    else:
        print("CPU 모드에서는 메모리 측정을 건너뜁니다.")


def benchmark_data_loading():
    """데이터 로딩 성능 벤치마크"""
    print("\n📊 데이터 로딩 성능 벤치마크")
    print("=" * 50)

    # 작은 샘플로 테스트
    sample_sizes = [100, 500, 1000, 5000]

    for sample_size in sample_sizes:
        print(f"\n샘플 크기: {sample_size}개")

        start_time = time.time()

        # 데이터셋 로드
        full_dataset = load_dataset("imdb", cache_dir="./data_cache")
        dataset = full_dataset["train"].select(range(sample_size))

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # 토큰화
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=512)

        dataset = dataset.map(tokenize, batched=True)

        end_time = time.time()
        print(f"   로딩 + 토큰화 시간: {end_time - start_time:.2f}초")
        print(f"   문장당 평균 시간: {(end_time - start_time)/sample_size:.4f}초")


def main():
    """메인 벤치마크 함수"""
    print("🚀 M4 Pro 성능 벤치마크 시작!")
    print("=" * 60)

    # 1. 모델 로딩 성능
    model, tokenizer = benchmark_model_loading()

    # 2. 추론 속도 벤치마크
    device = get_optimal_device()
    benchmark_inference_speed(model, tokenizer, device)

    # 3. 메모리 사용량 벤치마크
    benchmark_memory_usage()

    # 4. 데이터 로딩 성능 벤치마크
    benchmark_data_loading()

    print("\n" + "=" * 60)
    print("🎉 벤치마크 완료!")
    print("\n💡 성능 최적화 팁:")
    print("   1. 배치 크기를 늘려서 GPU 활용도 향상")
    print("   2. Mixed Precision Training (fp16) 사용")
    print("   3. DataLoader의 num_workers 증가")
    print("   4. Gradient Accumulation으로 메모리 효율성 향상")
    print("   5. M4 Pro의 Neural Engine 활용")


if __name__ == "__main__":
    main()
