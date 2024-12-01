# DNN Classifier with mini-batch gradient descent

2020170812 임정섭

## 파일 구조

> TODO: tree로 출력

## 실행 방법

### 1. 환경변수 설정

### 리눅스/MacOS 환경

파이썬 3.10 이상이 설치된 리눅스 환경을 기준으로 설명합니다.

```bash
export TEST_DATA_PATH="./data/test_data.txt"
export TRAIN_DATA_PATH="./data/train_data.txt"
export DEVICE="cpu"
```

#### 2. 파이썬 패키지 설치

Poetry 활용을 권장합니다.

```bash
poetry install  # Poetry가 설치되어 있을 경우
pip install -r requirements/cpu.txt  # Poetry가 없을 경우
```

#### 2.1 CUDA 환경 구성 (선택)

```bash
export DEVICE="gpu"
```

CUDA 환경에서 실행하려면 추가 패키지를 설치해야 합니다.

```bash
# CUDA 12.X (Colab 환경)
poetry install --with gpu  # Poetry가 설치되어 있을 경우
pip install -r requirements/gpu.txt  # Poetry가 없을 경우
```

CUDA 12.X 이외 버전에는 별도의 패키지를 설치해야 합니다.

> ([CuPy 설치방법 참고](https://docs.cupy.dev/en/v12.3.0/install.html#installing-cupy))

```bash
pip install cupy-cudaxxx  # CUDA 버전에 따라 xxx를 적절히 변경
```

#### 3. 모델 학습 및 테스트

```bash
pwd  # /.../cose362-machine-learning-dnn 확인
poetry run python dnn/main.py  # Poetry가 설치되어 있을 경우
python dnn/main.py  # Poetry가 없을 경우
```

### Colab 환경

![alt text](docs/colab.png)

## 모델 개요

### 데이터셋

- input x는 D=13차원의 연속형 변수입니다.
- output y는 K=2개의 클래스를 갖는 범주형 변수입니다.
- 데이터셋에는 N=60290개의 sample이 존재합니다.

### Inductive Biases

## 모델 상세

## 모델 선택

## 성능 평가
