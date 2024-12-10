# Docker 환경 설정 가이드

## Dockerfile 패키지 설명

### 기본 시스템 패키지
- `build-essential`: C/C++ 컴파일러 및 기본 빌드 도구
- `libgl1-mesa-glx`: OpenGL 라이브러리 (일부 시각화 패키지에 필요)
- `libglib2.0-0`: GLib 라이브러리 (시스템 유틸리티)
- `libpng-dev`: PNG 이미지 처리 라이브러리
- `libjpeg-dev`: JPEG 이미지 처리 라이브러리
- `tzdata`: 타임존 데이터 (시간대 설정에 필요)

### 개발 도구
- `curl`: URL을 통한 데이터 전송 도구
- `git`: 버전 관리 시스템
- `zsh`: Z 셸 (향상된 커맨드라인 셸)

### ZSH 관련 설정
- `oh-my-zsh`: ZSH 설정 프레임워크
- `powerlevel10k`: ZSH 테마
- `zsh-autosuggestions`: 자동 완성 추천 플러그인
- `zsh-syntax-highlighting`: 명령어 구문 강조 플러그인

## 최적화된 Dockerfile

```dockerfile
FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libpng-dev \
  libjpeg-dev \
  curl \
  git \
  tzdata \
  zsh \
  && rm -rf /var/lib/apt/lists/*

# ZSH 설정
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
  && git clone https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
  && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# uv 패키지 매니저 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
  . $HOME/.profile && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv venv /app/.venv && \
  . /app/.venv/bin/activate && \
  uv pip install --upgrade pip

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV TZ=Asia/Seoul

# 타임존 설정
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ZSH 설정 파일 복사
COPY ./zsh/zshrc /root/.zshrc
COPY ./zsh/p10k.zsh /root/.p10k.zsh
COPY ./.project_root ./app/.project_root

# 작업 디렉토리 설정
WORKDIR /app
RUN mkdir -p /app/data /app/runs /app/models /app/src \
  && chmod 777 /app/data /app/runs /app/models /app/src

# 의존성 설치
COPY requirements.txt .
RUN . /app/.venv/bin/activate && \
  export PATH="$HOME/.local/bin:$PATH" && \
  uv pip install -r requirements.txt

# 기본 셸 설정
SHELL ["/bin/zsh", "-c"]
```

## 주요 구성 요소 설명

### 1. uv 패키지 매니저
- pip보다 빠른 Python 패키지 설치 도구
- 의존성 해결 및 가상 환경 관리 기능 제공

### 2. 환경 변수 설정
- `PYTHONUNBUFFERED`: Python 출력 버퍼링 비활성화
- `PYTHONDONTWRITEBYTECODE`: .pyc 파일 생성 방지
- `PYTHONPATH`: Python 모듈 검색 경로
- `TZ`: 타임존 설정

### 3. 디렉토리 구조
```
/app/
  ├── data/      # 데이터 파일
  ├── models/    # 학습된 모델
  ├── runs/      # 실행 결과
  └── src/       # 소스 코드
```

### 4. ZSH 설정
- 개선된 명령행 인터페이스 제공
- 자동 완성 및 구문 강조 기능
- powerlevel10k 테마로 시각적 개선

## 사용 팁
- 대부분의 패키지는 머신러닝/딥러닝 환경을 위한 것입니다
- 필요에 따라 패키지를 추가하거나 제거할 수 있습니다
- 환경 변수는 프로젝트 요구사항에 맞게 조정 가능합니다