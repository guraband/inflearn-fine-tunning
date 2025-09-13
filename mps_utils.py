#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) 유틸리티 모듈

Apple Silicon Mac에서 PyTorch MPS 디바이스의 성능을 최적화하기 위한
warm-up 및 유틸리티 함수들을 제공합니다.
"""

import time
import torch


def warm_up_mps():
    """
    MPS 디바이스 warm-up을 수행합니다.

    Apple Silicon Mac의 MPS 디바이스 성능을 최적화하기 위해
    다양한 크기의 텐서 연산을 수행하여 GPU를 준비시킵니다.

    Returns:
        bool: MPS 사용 가능 여부
    """
    print("🔥 MPS 디바이스 warm-up 중...")

    # MPS 디바이스 확인
    if not torch.backends.mps.is_available():
        print("⚠️  MPS를 사용할 수 없습니다. CPU를 사용합니다.")
        return False

    device = "mps"
    start_time = time.time()

    print("   - 1단계: 기본 연산 warm-up...")
    # 1단계: 기본 연산 warm-up (500회)
    for i in range(500):
        dummy_tensor1 = torch.rand(1000, 1000).to(device)
        dummy_tensor2 = torch.rand(1000, 1000).to(device)
        torch.matmul(dummy_tensor1, dummy_tensor2)

        if i % 50 == 0:
            print(f"     진행률: {i+1}/500")

    print("   - 2단계: 대용량 연산 warm-up...")
    # 2단계: 대용량 연산 warm-up (100회)
    for i in range(100):
        dummy_tensor3 = torch.rand(3000, 3000).to(device)
        dummy_tensor4 = torch.rand(3000, 3000).to(device)
        torch.matmul(dummy_tensor3, dummy_tensor4)

        if i % 10 == 0:
            print(f"     진행률: {i+1}/100")

    print("   - 3단계: 복합 연산 warm-up...")
    # 3단계: 복합 연산 warm-up (50회)
    for i in range(50):
        # 다양한 연산 조합
        x = torch.rand(2000, 2000).to(device)
        y = torch.rand(2000, 2000).to(device)

        # 행렬 곱셈
        result1 = torch.matmul(x, y)
        # 전치 행렬
        result2 = torch.matmul(x.t(), y)
        # 요소별 곱셈
        result3 = x * y
        # 합계
        result4 = torch.sum(result1 + result2 + result3)
        # 추가 연산들
        result5 = torch.relu(result4)
        result6 = torch.softmax(result1, dim=1)
        result7 = torch.mean(result6)

        if i % 10 == 0:
            print(f"     진행률: {i+1}/50")

    print("   - 4단계: 극대용량 연산 warm-up...")
    # 4단계: 극대용량 연산 warm-up (20회)
    for i in range(20):
        # M4 Pro의 메모리 한계까지 활용
        dummy_tensor5 = torch.rand(4000, 4000).to(device)
        dummy_tensor6 = torch.rand(4000, 4000).to(device)
        torch.matmul(dummy_tensor5, dummy_tensor6)

        if i % 5 == 0:
            print(f"     진행률: {i+1}/20")

    end_time = time.time()
    print(f"✅ MPS warm-up 완료! 소요시간: {end_time - start_time:.2f}초")
    print(f"   - 총 연산: 670회 (기본 500회 + 대용량 100회 + 복합 50회 + 극대용량 20회)")
    print(f"   - 최대 텐서 크기: 4000x4000")
    print(f"   - 메모리 사용량: 최대 ~128GB (4000x4000x4바이트x2)")

    return True


def check_mps_availability():
    """
    MPS 디바이스 사용 가능 여부를 확인합니다.

    Returns:
        dict: MPS 상태 정보
            - available: MPS 사용 가능 여부
            - device: 사용 가능한 디바이스 ('mps' 또는 'cpu')
            - message: 상태 메시지
    """
    if torch.backends.mps.is_available():
        return {
            'available': True,
            'device': 'mps',
            'message': 'MPS 디바이스 사용 가능'
        }
    else:
        return {
            'available': False,
            'device': 'cpu',
            'message': 'MPS 디바이스 사용 불가 - CPU 사용'
        }


def get_optimal_device():
    """
    최적의 디바이스를 반환합니다.

    Returns:
        str: 사용할 디바이스 ('mps' 또는 'cpu')
    """
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_device_info():
    """
    현재 사용 가능한 디바이스 정보를 출력합니다.
    """
    print("🔍 디바이스 정보:")
    print("=" * 40)

    # PyTorch 버전
    print(f"PyTorch 버전: {torch.__version__}")

    # MPS 상태
    mps_status = check_mps_availability()
    print(f"MPS 상태: {mps_status['message']}")

    # CPU 정보
    print(f"CPU 코어 수: {torch.get_num_threads()}")

    # 권장 디바이스
    optimal_device = get_optimal_device()
    print(f"권장 디바이스: {optimal_device}")

    print("=" * 40)


if __name__ == "__main__":
    # 모듈을 직접 실행할 때의 테스트 코드
    print("🚀 MPS Utils 모듈 테스트")
    print_device_info()

    if torch.backends.mps.is_available():
        warm_up_mps()
    else:
        print("MPS를 사용할 수 없어 warm-up을 건너뜁니다.")
