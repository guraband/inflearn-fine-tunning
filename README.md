# BERT Fine-tuning for IMDB Sentiment Analysis

이 프로젝트는 Hugging Face Transformers를 사용하여 DistilBERT 모델을 IMDB 영화 리뷰 데이터셋으로 파인튜닝하는 예제입니다.

## 📁 파일 구조

```
├── 12_bert.py              # 메인 BERT 파인튜닝 스크립트
├── 12_bert.ipynb           # 원본 Jupyter 노트북
├── 12_bert_mlx.ipynb       # MLX 버전 노트북
├── mps_utils.py            # MPS 유틸리티 모듈
├── test_mps_utils.py       # MPS 유틸리티 테스트 스크립트
├── use_saved_model.py      # 저장된 모델 사용 예시
├── main.py                 # 간단한 실행 스크립트
├── pyproject.toml          # 프로젝트 의존성
├── README.md               # 이 파일
├── data_cache/             # 데이터셋 캐시 디렉토리 (자동 생성)
├── checkpoints/            # 체크포인트 저장 디렉토리 (자동 생성)
└── models/                 # 파인튜닝된 모델 저장 디렉토리 (자동 생성)
```

## 🚀 사용법

### 1. 기본 실행

```bash
python 12_bert.py
```

### 2. MPS 유틸리티 테스트

```bash
python test_mps_utils.py
```

### 3. 저장된 모델 사용

```bash
python use_saved_model.py
```

### 4. MPS 유틸리티 모듈 사용

다른 Python 스크립트에서 MPS 유틸리티를 사용하려면:

```python
from mps_utils import warm_up_mps, get_optimal_device, print_device_info

# 디바이스 정보 출력
print_device_info()

# 최적 디바이스 확인
device = get_optimal_device()

# MPS warm-up 실행
warm_up_mps()
```

## 🔧 주요 기능

### 모델 저장 및 로드

학습 완료 후 파인튜닝된 모델이 자동으로 `models/` 디렉토리에 저장됩니다.

#### 저장되는 파일들:
- `config.json`: 모델 설정
- `model.safetensors`: 모델 가중치
- `tokenizer.json`: 토크나이저
- `tokenizer_config.json`: 토크나이저 설정
- `vocab.txt`: 어휘 사전

#### 모델 사용 예시:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 저장된 모델 로드
tokenizer = AutoTokenizer.from_pretrained("models/bert_imdb_1000samples")
model = AutoModelForSequenceClassification.from_pretrained("models/bert_imdb_1000samples")

# 감정 분석 수행
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)
sentiment = "긍정" if prediction[0] == 1 else "부정"
```

### MPS Utils 모듈 (`mps_utils.py`)

Apple Silicon Mac에서 PyTorch MPS 성능을 최적화하기 위한 유틸리티 모듈입니다.

#### 주요 함수들:

- `warm_up_mps()`: MPS 디바이스 warm-up 수행
- `check_mps_availability()`: MPS 사용 가능 여부 확인
- `get_optimal_device()`: 최적의 디바이스 반환
- `print_device_info()`: 디바이스 정보 출력

#### 사용 예시:

```python
from mps_utils import warm_up_mps, get_optimal_device

# 디바이스 확인 및 warm-up
device = get_optimal_device()
if device == 'mps':
    warm_up_mps()

# 모델을 적절한 디바이스로 이동
model = model.to(device)
```

### BERT 파인튜닝 스크립트 (`12_bert.py`)

#### 주요 기능:

1. **데이터셋 로딩**: IMDB 데이터셋 로드 및 캐싱
2. **레이블 분포 분석**: 전체 데이터셋과 샘플 데이터의 레이블 분포 확인
3. **MPS 최적화**: Apple Silicon Mac에서 최적 성능을 위한 설정
4. **모델 훈련**: DistilBERT 모델 파인튜닝
5. **예측 테스트**: 학습된 모델로 감정 분석 예측

#### 데이터셋 정보:

- **전체 데이터**: 50,000개 (훈련 25,000개, 테스트 25,000개)
- **사용 샘플**: 1,000개 (전체의 2%)
- **레이블 분포**: 긍정 50%, 부정 50%

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -e .
```

### 2. 필요한 패키지들

- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `datasets` (Hugging Face)
- `scikit-learn`
- `tqdm`

### 3. Apple Silicon Mac 최적화

MPS (Metal Performance Shaders)를 사용하여 GPU 가속을 활용합니다:

```python
# MPS 사용 가능 여부 확인
import torch
print(torch.backends.mps.is_available())
```

## 📊 성능 최적화

### MPS Warm-up

Apple Silicon Mac에서 최적 성능을 위해 MPS 디바이스를 미리 준비시킵니다:

- **1단계**: 기본 연산 warm-up (500회)
- **2단계**: 대용량 연산 warm-up (100회)
- **3단계**: 복합 연산 warm-up (50회)
- **4단계**: 극대용량 연산 warm-up (20회)

### 훈련 설정

M4 Pro에 최적화된 설정:

- **배치 크기**: 32 (훈련/평가)
- **에포크**: 10
- **학습률**: 2e-5
- **가중치 감쇠**: 0.01

## 🔍 데이터셋 분석

### 레이블 분포 확인

스크립트는 다음을 자동으로 확인합니다:

1. **전체 데이터셋 레이블 분포**
2. **샘플 데이터 레이블 분포**
3. **레이블 1(긍정)이 처음 등장하는 위치**
4. **데이터 샘플 내용 확인**

### 샘플링 문제

IMDB 데이터셋은 레이블별로 정렬되어 있어서:
- **0~12,499번째**: 부정(0) 레이블
- **12,500~24,999번째**: 긍정(1) 레이블

따라서 처음 1,000개 샘플은 모두 부정 레이블입니다.

## 📝 사용 예시

### 다른 프로젝트에서 MPS Utils 사용

```python
# your_project.py
from mps_utils import warm_up_mps, get_optimal_device
import torch

# 디바이스 설정
device = get_optimal_device()

# MPS warm-up (MPS 사용 가능한 경우에만)
if device == 'mps':
    warm_up_mps()

# 모델 로드 및 디바이스 이동
model = YourModel()
model = model.to(device)

# 훈련 또는 추론 수행
# ...
```

## 🐛 문제 해결

### MPS 관련 문제

1. **MPS 사용 불가**: PyTorch 버전 확인 (1.12+ 필요)
2. **메모리 부족**: 배치 크기 줄이기
3. **성능 저하**: warm-up 실행 확인

### 데이터셋 관련 문제

1. **레이블 불균형**: 샘플링 방식 변경 고려
2. **캐시 문제**: `data_cache` 디렉토리 삭제 후 재실행

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해 주세요.
