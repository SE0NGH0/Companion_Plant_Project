# 🌿 Companion Plant Monitoring System

> 라즈베리파이 기반 **토마토 성장 감지 & 병충해 경고 + 센서 기반 반려식물 관리 시스템**

---

## 📌 프로젝트 개요

본 프로젝트는 OpenCV, YOLOv5, 라즈베리파이 기반의 센서 제어를 통해  
**토마토의 성장 단계와 병충해를 실시간 감지하고**, 센서 데이터를 수집하며,  
**이메일 알림과 웹 UI로 사용자에게 시각화된 정보를 제공**하는 통합 식물 모니터링 시스템입니다.

---

## 📽 데모 영상

[![Demo](https://img.youtube.com/vi/iGr49FzID6Q/0.jpg)](https://youtube.com/shorts/iGr49FzID6Q)  
🔗 https://youtube.com/shorts/iGr49FzID6Q

---

## ⚙️ 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 🎯 객체 인식 | YOLOv5 커스텀 모델로 토마토 성장(`green`, `half_red`, `red`) 및 병해(`tomato_blight`) 감지 |
| 🎥 실시간 스트리밍 | OpenCV + Flask로 카메라 영상 MJPEG 스트리밍 및 bounding box 시각화 |
| 🌡️ 센서 모니터링 | DHT11(온습도), BH1750(조도), 수위 센서 데이터를 실시간 수집 및 표시 |
| 🔔 알림 시스템 | 병충해 또는 완숙 상태 발생 시 사용자 이메일로 즉시 알림 전송 |
| 👤 사용자 인증 | SQLite 기반 로그인·회원가입 시스템 (이메일 필수) |
| 📊 기록 관리 | 감지 내역 및 센서 데이터를 웹에서 실시간 확인 가능 (JSON API 제공) |

---

## 🔧 구현 개요

- YOLOv5 두 모델(`growth.pt`, `condition.pt`) 병렬 추론 (ThreadPoolExecutor 사용)
- 카메라 프레임 분석 후 bounding box, confidence 표시
- 센서값은 4초 간격 측정 & 실패 시 이전 값 유지
- Flask 서버 내 `/video_feed`, `/sensor_status`, `/data` 등 RESTful API 구현
- 이메일 발송: Gmail SMTP + 앱 비밀번호 기반 알림 구현
- 프론트엔드: HTML(Jinja2), JavaScript(fetch), CSS 반응형 UI 구성

---

## 🚀 실행 방법

```bash
# 패키지 설치
sudo apt update
sudo apt install python3-pip libatlas-base-dev
pip3 install flask opencv-python torch torchvision adafruit-circuitpython-dht smbus2 RPi.GPIO

# 실행
python3 app_sensor.py
```

접속: http://[라즈베리파이_IP]:5000

---

## 📁 프로젝트 구조

```
Companion_Plant_Project/
├── app_sensor.py          # 서버 + 센서 + YOLO 통합 코드
├── users.db               # SQLite 사용자 DB
├── templates/             # index.html, login.html
└── yolov5/
    ├── growth.pt
    └── condition.pt
```

---

## ✉️ 알림 조건

- 병충해(`tomato_blight`) 감지 시: "🚨 병충해 알림"
- 완숙 상태(`red`) 감지 시: "🍅 수확 알림"
- 중복 알림 방지를 위한 플래그 사용 (`last_blight_sent`, `last_red_sent`)

---

## 📝 개발자 메모

- Thread + Flask 조합으로 실시간 데이터 처리
- MJPEG 스트리밍 방식으로 브라우저 렌더링 속도 최적화
- 사용자별 이메일 전송을 위한 DB 쿼리 최적화
- 향후: 수분 공급 자동화, 성장 히스토리 그래프, TTS 음성 알림 계획 중

---
