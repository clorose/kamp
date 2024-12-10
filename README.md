# 프로젝트 실행방법

## 사전 준비사항

1. [도커](https://github.com/clorose/docker) 설치가 필요합니다.
2. 다음 필수 파일들이 프로젝트 루트 경로에 있어야 합니다:
   ```
   .
   ├── .project_root     # 프로젝트 루트 식별자
   └── zsh/              # zsh 설정 파일들 (기본 설정)
       ├── zshrc
       └── p10k.zsh
   ```
   필수 파일은 [여기](https://drive.google.com/drive/folders/1clLbRyTU1UMfZn4n6Cig7JIB3WGW0lAS?usp=sharing)에서 다운로드 가능합니다.

## 셸 선택
- 기본적으로 `zsh`와 추가 기능이 포함된 환경이 제공됩니다.
- `bash`를 선호하시는 경우 `/bash` 폴더의 `Dockerfile`을 사용해주세요. (자세한 적용법은 [도커 레포지토리](https://github.com/clorose/docker) 참조)

## 실행 방법

### 1. 최초 실행 시
```bash
docker compose up --build dev
```

### 2. 이후 실행 시
```bash
docker compose up dev
```

## 주의사항
- GPU 사용 시 NVIDIA 드라이버와 CUDA가 설치되어 있어야 합니다.
- 필수 파일(`.project_root`, `zsh` 설정파일)이 없으면 실행되지 않습니다.

## 문제해결
- GPU 인식 문제 발생 시 `nvidia-smi` 명령어로 상태를 확인해보세요.