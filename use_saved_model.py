#!/usr/bin/env python3
"""
저장된 파인튜닝 모델 사용 예시

이 스크립트는 12_bert.py로 학습한 모델을 로드하여
새로운 텍스트에 대한 감정 분석을 수행합니다.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mps_utils import get_optimal_device


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


def predict_sentiment(text: str, tokenizer, model):
    """텍스트의 감정을 예측합니다."""
    device = get_optimal_device()

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    # 결과 반환
    sentiment = "긍정" if predictions[0] == 1 else "부정"
    confidence = probabilities[0][predictions[0]].item()

    return sentiment, confidence


def main():
    """메인 실행 함수"""
    print("🚀 저장된 BERT 모델로 감정 분석 시작!")
    print("=" * 50)

    # 사용 가능한 모델 찾기
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ models 디렉토리가 없습니다.")
        print("   먼저 12_bert.py를 실행하여 모델을 학습하세요.")
        return

    # 모델 목록 표시
    model_dirs = [d for d in os.listdir(
        models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not model_dirs:
        print("❌ 저장된 모델이 없습니다.")
        print("   먼저 12_bert.py를 실행하여 모델을 학습하세요.")
        return

    print("📁 사용 가능한 모델:")
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"   [{i}] {model_dir}")

    # 모델 선택
    while True:
        try:
            choice = input(f"\n모델 선택 (1-{len(model_dirs)}): ").strip()
            if choice == "":
                choice = "1"  # 기본값
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_dirs):
                selected_model = model_dirs[choice_idx]
                break
            else:
                print(f"❌ 1-{len(model_dirs)} 범위의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")

    # 모델 로드
    model_path = os.path.join(models_dir, selected_model)
    tokenizer, model = load_saved_model(model_path)

    if tokenizer is None or model is None:
        print("❌ 모델 로드에 실패했습니다.")
        return

    # 감정 분석 수행
    print(f"\n🎯 감정 분석 시작 (모델: {selected_model})")
    print("=" * 50)

    # 테스트 문장들
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible acting and boring plot. Complete waste of time.",
        "The story was engaging and the characters were well-developed.",
        "I would not recommend this film to anyone.",
        "Amazing cinematography and outstanding performances!",
        "The movie was okay, nothing special but not bad either.",
        "This is the worst film I have ever seen in my life.",
        "I can watch this movie over and over again!"
    ]

    print("📝 테스트 문장들:")
    for i, text in enumerate(test_texts, 1):
        sentiment, confidence = predict_sentiment(text, tokenizer, model)
        print(f"\n[{i}] {text}")
        print(f"    예측: {sentiment} (신뢰도: {confidence:.3f})")

    # 사용자 입력 받기
    print(f"\n💬 직접 텍스트 입력하기:")
    print("   (종료하려면 'quit' 입력)")
    print("-" * 50)

    while True:
        user_text = input("\n텍스트 입력: ").strip()

        if user_text.lower() in ['quit', 'exit', 'q']:
            break

        if not user_text:
            print("❌ 텍스트를 입력하세요.")
            continue

        sentiment, confidence = predict_sentiment(user_text, tokenizer, model)
        print(f"   예측: {sentiment} (신뢰도: {confidence:.3f})")

    print("\n🎉 감정 분석 완료!")


if __name__ == "__main__":
    import torch
    main()
