# 🌿 Companion Plant Monitoring System

> 라즈베리파이 기반 **토마토 성장 감지 & 병충해 경고 + 센서 기반 반려식물 관리 시스템**

---

## 📌 프로젝트 개요

본 프로젝트는 **OpenCV + YOLOv5 + 라즈베리파이 센서**를 기반으로 식물 상태를 실시간으로 모니터링하고,  
**토마토의 생장 상태 및 병충해 여부를 감지하여 사용자에게 이메일로 알림**을 제공하는 스마트 반려식물 시스템입니다.

- 토마토 생장 단계 감지 (`green`, `half_red`, `red`)
- 병충해 여부 감지 (`tomato_blight`)
- 온습도(DHT11), 조도(BH1750), 수위(센서) 실시간 측정
- Flask 웹서버 기반 사용자 로그인 / 감지 기록 제공
- YOLOv5 커스텀 모델 2종 (`growth.pt`, `condition.pt`)
- 병충해/완숙 토마토 감지 시 이메일 전송 기능 포함

---

## 📽 데모 영상

> 아래 영상은 시스템이 실제로 동작하는 모습을 촬영한 시연 영상입니다.

[![토마토 모니터링 시스템 시연 영상](https://img.youtube.com/vi/iGr49FzID6Q/0.jpg)](https://youtube.com/shorts/iGr49FzID6Q)

👉 [시연 영상 바로 보기](https://youtube.com/shorts/iGr49FzID6Q)

- 실시간 영상 스트리밍 + bounding box 시각화
- 성장 단계 및 병충해 상태 표시
- 웹 UI에서 실시간 센서 상태 확인
- 병충해 및 완숙 감지 시 이메일 전송되는 모습 포함

---

## ⚙️ 시스템 구조도

```text
+----------------------------+
|    Raspberry Pi           |
|                            |
|  +-------------+          |
|  | Camera      |--YOLOv5--+--> 성장/병해 추론 결과
|  +-------------+          |
|                            |
|  +-------------+          |
|  | DHT11       |--온습도---+
|  | BH1750      |--조도-----+--> Flask 웹 서버
|  | Water Sensor|--수위-----+--> 실시간 JSON 제공
|  +-------------+          |
|                            |
|  +-------------+          |
|  | Flask Server |<----> User 로그인/관리
|  +-------------+          |
|                            |
|  +-------------+          |
|  | SMTP(Gmail) |<--- 알림 전송 (완숙/병해)
|  +-------------+          |
+----------------------------+
```

---

## 📂 디렉토리 구조

```
Companion_Plant_Project/
├── app_sensor.py          # Flask 웹서버 + 센서/카메라 추론 통합
├── users.db               # 사용자 인증용 SQLite DB
├── templates/
│   ├── index.html         # 메인 모니터링 화면
│   └── login.html         # 로그인 화면
├── static/                # (옵션) 스타일, JS, 이미지 등
└── yolov5/
    ├── growth.pt          # 생장 추론용 YOLO 모델
    └── condition.pt       # 병충해 추론용 YOLO 모델
```

---

## 🚀 실행 방법

### 1. 환경 설정

- Raspberry Pi OS 설치 (카메라 모듈 활성화 필요)
- Python 가상환경 생성 및 패키지 설치:

```bash
sudo apt update
sudo apt install python3-pip libatlas-base-dev
pip3 install flask opencv-python torch torchvision              adafruit-circuitpython-dht              smbus2 RPi.GPIO
```

### 2. YOLOv5 모델 저장 위치

`/home/pi/yolov5/yolov5-env/growth.pt`, `condition.pt` 경로에 모델 배치 (필요시 `app_sensor.py` 경로 수정)

### 3. 실행

```bash
python3 app_sensor.py
```

접속: 브라우저에서 `http://라즈베리파이_IP:5000`

---

## ✉️ 알림 기능

- 감지 Confidence ≥ 0.6 조건 만족 시:
  - **"토마토 병충해"** → 🚨 메일 발송
  - **"토마토 완숙(red)"** → 🍅 수확 알림 발송

> 메일 발송을 위해 `app_sensor.py` 내 `MAIL_USERNAME`, `MAIL_PASSWORD` 항목 수정 필요  
> Gmail 사용자는 [앱 비밀번호](https://myaccount.google.com/apppasswords) 필요

---

## 👤 로그인 기능

- 회원가입 시: `아이디 + 비밀번호 + 이메일`
- 인증된 사용자만 웹에 접근 가능 (로그인 필수)

---

## 📝 개발자 메모

- YOLO 추론은 `ThreadPoolExecutor`로 병렬 실행
- 센서 측정 실패 시 이전 값 유지
- 카메라 미연결 오류 자동 복구 시도 포함
- 향후 기능: 식물별 AI 추천 물주기 주기, TTS 알림 등

---
