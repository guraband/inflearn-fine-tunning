#!/usr/bin/env python3
"""
원래 모델과 파인튜닝된 모델 비교 스크립트

이 스크립트는 원본 DistilBERT 모델과 파인튜닝된 모델의 성능을
다양한 테스트 문장으로 비교합니다.
"""

import os
import time
import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mps_utils import get_optimal_device, print_device_info


def load_models():
    """원본 모델과 파인튜닝된 모델을 로드합니다."""
    print("📂 모델 로딩 중...")

    # 디바이스 확인
    device = get_optimal_device()
    print(f"   - 사용 디바이스: {device}")

    # 원본 모델 로드
    print("   - 원본 DistilBERT 모델 로딩...")
    original_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased")
    original_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")
    original_model = original_model.to(device)  # 디바이스로 이동

    # 파인튜닝된 모델 찾기
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ models 디렉토리가 없습니다.")
        print("   먼저 12_bert.py를 실행하여 모델을 학습하세요.")
        return None, None, None, None

    model_dirs = [d for d in os.listdir(
        models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        print("❌ 저장된 모델이 없습니다.")
        print("   먼저 12_bert.py를 실행하여 모델을 학습하세요.")
        return None, None, None, None

    # 가장 최근 모델 선택 (파일 수정 시간 기준)
    model_dirs.sort(key=lambda x: os.path.getmtime(
        os.path.join(models_dir, x)), reverse=True)
    selected_model = model_dirs[0]

    print(f"   - 파인튜닝된 모델 로딩: {selected_model}")
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(models_dir, selected_model))
    fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(models_dir, selected_model))
    fine_tuned_model = fine_tuned_model.to(device)  # 디바이스로 이동

    print("✅ 모델 로딩 완료!")
    return original_tokenizer, original_model, fine_tuned_tokenizer, fine_tuned_model


def predict_sentiment(text: str, tokenizer, model, model_name: str) -> Tuple[str, float, float]:
    """텍스트의 감정을 예측합니다."""
    device = get_optimal_device()

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 예측 수행 (모델은 이미 디바이스에 있음)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    # 결과 반환
    sentiment = "긍정" if predictions[0] == 1 else "부정"
    confidence = probabilities[0][predictions[0]].item()
    negative_prob = probabilities[0][0].item()
    positive_prob = probabilities[0][1].item()

    return sentiment, confidence, positive_prob


def get_test_sentences() -> List[Dict[str, any]]:
    """테스트 문장들을 반환합니다."""
    return [
        {
            "text": "This movie was absolutely fantastic! I loved every minute of it.",
            "expected": "긍정",
            "category": "명확한 긍정"
        },
        {
            "text": "Terrible acting and boring plot. Complete waste of time.",
            "expected": "부정",
            "category": "명확한 부정"
        },
        {
            "text": "The story was engaging and the characters were well-developed.",
            "expected": "긍정",
            "category": "긍정적 평가"
        },
        {
            "text": "I would not recommend this film to anyone.",
            "expected": "부정",
            "category": "부정적 평가"
        },
        {
            "text": "Amazing cinematography and outstanding performances!",
            "expected": "긍정",
            "category": "기술적 칭찬"
        },
        {
            "text": "This is the worst film I have ever seen in my life.",
            "expected": "부정",
            "category": "극도 부정"
        },
        {
            "text": "The movie was okay, nothing special but not bad either.",
            "expected": "중립",
            "category": "중립적"
        },
        {
            "text": "I can watch this movie over and over again!",
            "expected": "긍정",
            "category": "강한 긍정"
        },
        {
            "text": "The acting was mediocre and the plot was predictable.",
            "expected": "부정",
            "category": "부정적 비판"
        },
        {
            "text": "This film exceeded all my expectations!",
            "expected": "긍정",
            "category": "기대 초과"
        }
    ]


def compare_models():
    """두 모델을 비교합니다."""
    print("🔍 모델 비교 시작!")
    print("=" * 60)

    # 디바이스 정보 출력
    print_device_info()

    # 모델 로드
    original_tokenizer, original_model, fine_tuned_tokenizer, fine_tuned_model = load_models()

    if original_tokenizer is None:
        return

    # 테스트 문장들
    test_sentences = get_test_sentences()

    print(f"\n📝 테스트 문장 수: {len(test_sentences)}개")
    print("=" * 60)

    # 결과 저장
    results = []
    original_correct = 0
    fine_tuned_correct = 0

    for i, test_case in enumerate(test_sentences, 1):
        text = test_case["text"]
        expected = test_case["expected"]
        category = test_case["category"]

        print(f"\n[{i}] {category}")
        print(f"텍스트: {text}")
        print(f"예상: {expected}")

        # 원본 모델 예측
        orig_sentiment, orig_confidence, orig_positive_prob = predict_sentiment(
            text, original_tokenizer, original_model, "원본"
        )

        # 파인튜닝된 모델 예측
        ft_sentiment, ft_confidence, ft_positive_prob = predict_sentiment(
            text, fine_tuned_tokenizer, fine_tuned_model, "파인튜닝"
        )

        # 결과 출력
        print(f"원본 모델:     {orig_sentiment} (신뢰도: {
              orig_confidence:.3f}, 긍정확률: {orig_positive_prob:.3f})")
        print(f"파인튜닝 모델: {ft_sentiment} (신뢰도: {
              ft_confidence:.3f}, 긍정확률: {ft_positive_prob:.3f})")

        # 정확도 계산 (중립 제외)
        if expected != "중립":
            if orig_sentiment == expected:
                original_correct += 1
                print("   ✅ 원본 모델 정답")
            else:
                print("   ❌ 원본 모델 오답")

            if ft_sentiment == expected:
                fine_tuned_correct += 1
                print("   ✅ 파인튜닝 모델 정답")
            else:
                print("   ❌ 파인튜닝 모델 오답")

        # 결과 저장
        results.append({
            "text": text,
            "expected": expected,
            "category": category,
            "original": {
                "sentiment": orig_sentiment,
                "confidence": orig_confidence,
                "positive_prob": orig_positive_prob
            },
            "fine_tuned": {
                "sentiment": ft_sentiment,
                "confidence": ft_confidence,
                "positive_prob": ft_positive_prob
            }
        })

    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("📊 비교 결과 요약")
    print("=" * 60)

    # 중립 제외한 문장 수
    non_neutral_count = sum(1 for r in results if r["expected"] != "중립")

    original_accuracy = original_correct / \
        non_neutral_count * 100 if non_neutral_count > 0 else 0
    fine_tuned_accuracy = fine_tuned_correct / \
        non_neutral_count * 100 if non_neutral_count > 0 else 0

    print(f"총 테스트 문장: {len(test_sentences)}개")
    print(f"중립 제외 문장: {non_neutral_count}개")
    print(f"원본 모델 정확도: {
          original_correct}/{non_neutral_count} ({original_accuracy:.1f}%)")
    print(f"파인튜닝 모델 정확도: {
          fine_tuned_correct}/{non_neutral_count} ({fine_tuned_accuracy:.1f}%)")

    improvement = fine_tuned_accuracy - original_accuracy
    if improvement > 0:
        print(f"🎉 파인튜닝으로 {improvement:.1f}%p 개선!")
    elif improvement < 0:
        print(f"⚠️  파인튜닝으로 {abs(improvement):.1f}%p 감소")
    else:
        print("➖ 성능 변화 없음")

    # 카테고리별 분석
    print(f"\n📈 카테고리별 분석:")
    print("-" * 40)

    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"original": 0, "fine_tuned": 0, "total": 0}

        if result["expected"] != "중립":
            categories[cat]["total"] += 1
            if result["original"]["sentiment"] == result["expected"]:
                categories[cat]["original"] += 1
            if result["fine_tuned"]["sentiment"] == result["expected"]:
                categories[cat]["fine_tuned"] += 1

    for cat, stats in categories.items():
        if stats["total"] > 0:
            orig_acc = stats["original"] / stats["total"] * 100
            ft_acc = stats["fine_tuned"] / stats["total"] * 100
            print(f"{cat:12}: 원본 {orig_acc:5.1f}% → 파인튜닝 {
                  ft_acc:5.1f}% (개선: {ft_acc-orig_acc:+5.1f}%p)")

    print("\n🎉 모델 비교 완료!")


def interactive_comparison():
    """대화형 모델 비교를 수행합니다."""
    print("💬 대화형 모델 비교")
    print("=" * 30)
    print("직접 텍스트를 입력하여 두 모델을 비교할 수 있습니다.")
    print("종료하려면 'quit' 입력")
    print("-" * 30)

    # 모델 로드
    original_tokenizer, original_model, fine_tuned_tokenizer, fine_tuned_model = load_models()

    if original_tokenizer is None:
        return

    while True:
        user_text = input("\n텍스트 입력: ").strip()

        if user_text.lower() in ['quit', 'exit', 'q']:
            break

        if not user_text:
            print("❌ 텍스트를 입력하세요.")
            continue

        print(f"\n텍스트: {user_text}")

        # 원본 모델 예측
        orig_sentiment, orig_confidence, orig_positive_prob = predict_sentiment(
            user_text, original_tokenizer, original_model, "원본"
        )

        # 파인튜닝된 모델 예측
        ft_sentiment, ft_confidence, ft_positive_prob = predict_sentiment(
            user_text, fine_tuned_tokenizer, fine_tuned_model, "파인튜닝"
        )

        print(f"원본 모델:     {orig_sentiment} (신뢰도: {
              orig_confidence:.3f}, 긍정확률: {orig_positive_prob:.3f})")
        print(f"파인튜닝 모델: {ft_sentiment} (신뢰도: {
              ft_confidence:.3f}, 긍정확률: {ft_positive_prob:.3f})")

        # 결과 비교
        if orig_sentiment == ft_sentiment:
            print("   ➖ 두 모델의 예측이 동일합니다.")
        else:
            print("   ⚡ 두 모델의 예측이 다릅니다!")

    print("\n🎉 대화형 비교 완료!")


def main():
    """메인 실행 함수"""
    print("🚀 원본 모델 vs 파인튜닝 모델 비교")
    print("=" * 50)

    while True:
        print("\n🎯 비교 모드를 선택하세요:")
        print("   1. 자동 테스트 (10개 문장)")
        print("   2. 대화형 테스트")
        print("   3. 종료")

        choice = input("\n선택 (1-3): ").strip()

        if choice == "1":
            compare_models()
        elif choice == "2":
            interactive_comparison()
        elif choice == "3":
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 1-3 중에서 선택하세요.")


if __name__ == "__main__":
    main()
