# Bouldering Pose & Trajectory Analyser

볼더링 영상을 분석하여 신체 주요 부위의 궤적, 관절 각도, 속도 및 무게 중심(CoM)을 시각화해주는 도구입니다.

## 주요 기능
- **궤적 추적**: 골반, 손, 발, 팔꿈치, 무릎의 이동 경로 표시 (누적 가능)
- **실시간 데이터**: 무게 중심(CoM), 실시간 속도(n/s), 관절 각도 표시
- **커스터마이징**: 각 부위별 궤적 표시 여부 선택 및 색상 모드 설정
- **환경 독립성**: 로컬 Python 환경 및 Podman(Docker) 컨테이너 지원

## 실행 방법

### 방법 1: Podman/Docker (추천)
의존성 설치 없이 즉시 실행 가능합니다.
```bash
./run_podman.sh <입력디렉토리> <출력디렉토리>
```

### 방법 2: 로컬 Python
```bash
./process_all.sh <입력디렉토리> <출력디렉토리>
```

## 주요 옵션 (Python 직접 실행 시)
- `--traj-len`: 궤적 누적 길이 (기본값: 1000)
- `--no-start`: 시작점 마커 숨기기
- `--mono-color`: 궤적 색상을 단일색으로 통일
- `--no-l-hand`, `--no-r-knee` 등: 특정 부위별 추적 비활성화

## 기술 스택
- Python 3.12
- MediaPipe (Pose Landmarker Task API)
- OpenCV
- Podman / Docker
