#!/usr/bin/env python3
"""
MPS Utils 모듈 테스트 스크립트

mps_utils 모듈의 기능들을 테스트하고 사용법을 보여줍니다.
"""

from mps_utils import (
    warm_up_mps,
    check_mps_availability,
    get_optimal_device,
    print_device_info
)


def test_mps_utils():
    """MPS 유틸리티 함수들을 테스트합니다."""
    print("🧪 MPS Utils 모듈 테스트 시작!")
    print("=" * 50)

    # 1. 디바이스 정보 출력
    print("\n1️⃣ 디바이스 정보 확인:")
    print_device_info()

    # 2. MPS 사용 가능 여부 확인
    print("\n2️⃣ MPS 사용 가능 여부 확인:")
    mps_status = check_mps_availability()
    print(f"   상태: {mps_status['message']}")
    print(f"   사용 가능: {mps_status['available']}")
    print(f"   디바이스: {mps_status['device']}")

    # 3. 최적 디바이스 확인
    print("\n3️⃣ 최적 디바이스 확인:")
    optimal_device = get_optimal_device()
    print(f"   권장 디바이스: {optimal_device}")

    # 4. MPS warm-up 실행 (MPS 사용 가능한 경우에만)
    print("\n4️⃣ MPS Warm-up 실행:")
    if mps_status['available']:
        success = warm_up_mps()
        if success:
            print("   ✅ Warm-up 성공!")
        else:
            print("   ❌ Warm-up 실패!")
    else:
        print("   ⏭️  MPS를 사용할 수 없어 warm-up을 건너뜁니다.")

    print("\n" + "=" * 50)
    print("🎉 테스트 완료!")


if __name__ == "__main__":
    test_mps_utils()
