import os
import time
import threading
import sqlite3
import cv2
import torch
import numpy as np
import smbus2 as smbus
import RPi.GPIO as GPIO
import board
import adafruit_dht
import smtplib
from email.mime.text import MIMEText
from functools import wraps
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash,
    Response, jsonify, g
)
from werkzeug.security import generate_password_hash, check_password_hash
import concurrent.futures

# ─────────────────────────────────────────────────────────────────────────────
# Flask 앱 설정
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLite 데이터베이스 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# ─────────────────────────────────────────────────────────────────────────────
# 이메일(SMTP) 설정 (Gmail 예시)
# ─────────────────────────────────────────────────────────────────────────────
MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
MAIL_USERNAME = "ericsungho@gmail.com"       # 실제 발신 Gmail 주소
MAIL_PASSWORD = "xspm amkw fchq zgre"         # 실제 App Password
MAIL_USE_TLS = True

def send_email(to_address: str, subject: str, body: str):
    """
    간단한 SMTP 이메일 전송 함수 (예: Gmail).
    """
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = MAIL_USERNAME
    msg["To"] = to_address

    try:
        server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)
        if MAIL_USE_TLS:
            server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_USERNAME, [to_address], msg.as_string())
        server.quit()
    except Exception as e:
        print(f"[Email Error] {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 데이터베이스 연결 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    """g._database에 DB 연결을 캐시하고 반환."""
    if "_database" not in g:
        g._database = sqlite3.connect(DB_PATH)
        g._database.row_factory = sqlite3.Row
    return g._database

@app.teardown_appcontext
def close_connection(exception):
    """앱 컨텍스트 종료 시 DB 연결도 닫기."""
    db = g.pop("_database", None)
    if db is not None:
        db.close()

def init_db():
    """
    앱 실행 시 한 번만 호출되어, users 테이블을 생성합니다.
    기존 username, password_hash 외에 email 칼럼을 추가했습니다.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        );
    """)
    db.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 로그인 체크 데코레이터
# ─────────────────────────────────────────────────────────────────────────────
def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(**kwargs)
    return wrapped_view

# ─────────────────────────────────────────────────────────────────────────────
# 회원가입 라우트 (이메일 입력 추가)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        email = request.form.get("email", "").strip()

        if not username or not password or not password2 or not email:
            error = "아이디, 비밀번호, 이메일을 모두 입력해주세요."
        elif password != password2:
            error = "비밀번호가 일치하지 않습니다."
        else:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone() is not None:
                error = "이미 사용 중인 아이디 또는 이메일입니다."
            else:
                pw_hash = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                    (username, pw_hash, email)
                )
                db.commit()
                flash("회원가입이 완료되었습니다. 로그인해주세요.", "success")
                return redirect(url_for("login"))

    return render_template("register.html", error=error)

# ─────────────────────────────────────────────────────────────────────────────
# 로그인 라우트
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()

        if user is None:
            error = "존재하지 않는 아이디입니다."
        elif not check_password_hash(user["password_hash"], password):
            error = "비밀번호가 잘못되었습니다."
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))

    return render_template("login.html", error=error)

# ─────────────────────────────────────────────────────────────────────────────
# 로그아웃 라우트
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv5 모델 로드 (토마토 성장·병충해 판별)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/home/pi/yolov5/yolov5-env"  # 실제 경로로 수정

# growth.pt: 토마토 생장 단계 (green, half_red, red) 판별용
growth_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=os.path.join(MODEL_DIR, "growth.pt"),
    force_reload=False
).to(DEVICE)
growth_model.eval()

# condition.pt: 토마토 병충해(tomato_blight) 판별용
condition_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=os.path.join(MODEL_DIR, "condition.pt"),
    force_reload=False
).to(DEVICE)
condition_model.eval()

def preprocess_for_yolo(frame):
    """
    OpenCV로 읽은 BGR frame을 YOLOv5가 기대하는 RGB numpy array로 변환.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img

def _infer_growth(frame):
    results = []
    img = preprocess_for_yolo(frame)
    preds = growth_model(img, size=640)
    df = preds.pandas().xyxy[0]
    for _, row in df.iterrows():
        x1, y1 = int(row.xmin), int(row.ymin)
        x2, y2 = int(row.xmax), int(row.ymax)
        conf = float(row.confidence)
        label = row["name"]  # "green", "half_red", "red"
        results.append({
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "source": "growth"
        })
    return results

def _infer_condition(frame):
    results = []
    img = preprocess_for_yolo(frame)
    preds = condition_model(img, size=640)
    df = preds.pandas().xyxy[0]
    for _, row in df.iterrows():
        x1, y1 = int(row.xmin), int(row.ymin)
        x2, y2 = int(row.xmax), int(row.ymax)
        conf = float(row.confidence)
        label = row["name"]  # 보통 "tomato_blight"
        results.append({
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "source": "condition"
        })
    return results

def detect_with_models(frame):
    """
    두 개의 YOLOv5 custom 모델(growth_model, condition_model)을
    ThreadPoolExecutor로 병렬 실행하여 추론 결과를 합쳐서 반환합니다.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_growth = executor.submit(_infer_growth, frame)
        future_cond   = executor.submit(_infer_condition, frame)
        for future in concurrent.futures.as_completed([future_growth, future_cond]):
            results.extend(future.result())
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 센서(DHT11, BH1750, 수위) 설정 및 전역 변수 선언
# ─────────────────────────────────────────────────────────────────────────────
# DHT11
dht_device = adafruit_dht.DHT11(board.D17)

# BH1750
I2C_CH = 1
BH1750_DEV_ADDR = 0x23
CONT_H_RES_MODE = 0x10
try:
    i2c = smbus.SMBus(I2C_CH)
except FileNotFoundError:
    print("[Warning] I2C 버스를 찾을 수 없습니다. 조도 측정을 생략합니다.")
    i2c = None

# 수위 센서
WATER_SENSOR_PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(WATER_SENSOR_PIN, GPIO.IN)

# 전역 변수에 최신 센서값을 저장
# last_update는 'DHT11이 정상적으로 읽은 시각'만 기록
sensor_data = {
    "temperature": None,
    "humidity": None,
    "lux": None,
    "water_detected": None,
    "last_update": None
}

def read_dht11(max_retries=5):
    """
    DHT11을 최대 max_retries번 읽어보고, 성공하면 (temp, hum),
    실패하면 (None, None)을 반환.
    """
    for attempt in range(1, max_retries+1):
        try:
            t = dht_device.temperature
            h = dht_device.humidity
            if (t is not None) and (h is not None):
                return (t, h)
        except RuntimeError:
            if attempt < max_retries:
                time.sleep(1)
            else:
                return (None, None)

def sensor_loop():
    """
    별도 스레드에서 주기적으로 DHT11, BH1750, 수위 센서 값을 읽어서
    전역 변수(sensor_data)에 저장합니다. DHT11이 정상적으로 읽었을 때만 last_update를 갱신.
    """
    global sensor_data
    while True:
        # --- 온습도 측정 (DHT11) ---
        temperature, humidity = read_dht11(max_retries=5)

        # DHT11이 정상적으로 읽혔을 때만 sensor_data 전체를 업데이트
        if (temperature is not None) and (humidity is not None):
            # --- 조도 측정 (BH1750) ---
            try:
                if i2c is not None:
                    lux_bytes = i2c.read_i2c_block_data(BH1750_DEV_ADDR, CONT_H_RES_MODE, 2)
                    raw = int.from_bytes(lux_bytes, byteorder='big')
                    lux = raw / 1.2
                else:
                    lux = None
            except Exception:
                lux = None

            # --- 물 감지 측정 ---
            try:
                water_detected = (GPIO.input(WATER_SENSOR_PIN) == GPIO.HIGH)
            except Exception:
                water_detected = None

            # --- 오직 DHT11이 성공적으로 읽었을 때만 sensor_data에 쓰기 ---
            sensor_data["temperature"] = temperature
            sensor_data["humidity"]    = humidity
            sensor_data["lux"]         = lux
            sensor_data["water_detected"] = water_detected
            sensor_data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # DHT11이 실패했으면 sensor_data는 건드리지 않고, 값도 last_update도 그대로 유지
        time.sleep(4)  # 4초 주기로 측정 (DHT11 안정 시간을 충분히 확보)

# ─────────────────────────────────────────────────────────────────────────────
# 비디오 스트리밍 + 감지 데이터 로직
# ─────────────────────────────────────────────────────────────────────────────
CAM_INDEX = 0
detection_history = []
lock = threading.Lock()

# 이메일 중복발송 방지 플래그
last_blight_sent = False
last_red_sent = False

# 카메라 초기화
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    os.system(f"fuser -k /dev/video{CAM_INDEX} 2>/dev/null")
    time.sleep(1)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 카메라 인덱스 {CAM_INDEX}번을 열 수 없습니다.")

def gen_frames():
    """
    MJPEG 스트리밍용 제너레이터.
    매 프레임마다 YOLO 추론 → 바운딩 박스 그리기 → JPEG 인코딩 → yield
    10초마다 confidence 기준으로 추출한 결과를 detection_history에 저장하고,
    TOMATO BLIGHT 또는 RED 감지 시 이메일 발송.
    """
    global detection_history, last_blight_sent, last_red_sent

    last_record_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 병렬 추론 수행 ──
        detections = detect_with_models(frame)
        #───────────────────────

        now = time.time()
        if now - last_record_time >= 10:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))

            # 1) confidence ≥ 0.6 필터링
            valid_dets = [det for det in detections if det["conf"] >= 0.6]

            # 2) growth 모델 결과 중 가장 높은 항목 추출
            growth_dets = [det for det in valid_dets if det["source"] == "growth"]
            if growth_dets:
                best_growth = max(growth_dets, key=lambda d: d["conf"])
                growth_label = best_growth["label"]
                growth_conf  = best_growth["conf"]
            else:
                growth_label = None
                growth_conf  = None

            # 3) condition 모델 결과 중 가장 높은 항목 추출
            cond_dets = [det for det in valid_dets if det["source"] == "condition"]
            if cond_dets:
                best_cond = max(cond_dets, key=lambda d: d["conf"])
                cond_label = best_cond["label"]
                cond_conf  = best_cond["conf"]
            else:
                cond_label = None
                cond_conf  = None

            # 4) summary 구성
            summary = {
                "growth": {
                    "label": growth_label,
                    "conf": growth_conf
                },
                "condition": {
                    "label": cond_label,
                    "conf": cond_conf
                }
            }

            # 5) DB에서 모든 사용자 이메일 조회
            try:
                db_conn = sqlite3.connect(DB_PATH)
                db_conn.row_factory = sqlite3.Row
                cursor = db_conn.cursor()
                cursor.execute("SELECT email FROM users")
                users = cursor.fetchall()
                db_conn.close()
            except Exception as e:
                print(f"[DB Error] {e}")
                users = []

            # ─── 병충해 알림 ───
            if cond_label == "tomato_blight" and not last_blight_sent:
                subject = "🚨 토마토 병충해 감지 알림"
                body    = "식물이 손상되었습니다. 확인해주세요."
                for u in users:
                    send_email(u["email"], subject, body)
                last_blight_sent = True
            if cond_label != "tomato_blight":
                last_blight_sent = False

            # ─── 완숙(red) 알림 ───
            if growth_label == "red" and not last_red_sent:
                subject = "🍅 토마토 완숙 알림"
                body    = "토마토가 모두 익었습니다. 수확 준비를 해주세요."
                for u in users:
                    send_email(u["email"], subject, body)
                last_red_sent = True
            if growth_label != "red":
                last_red_sent = False

            # 6) detection_history에 저장 (최대 50개)
            with lock:
                detection_history.insert(0, (timestamp, summary))
                if len(detection_history) > 50:
                    detection_history = detection_history[:50]

            last_record_time = now

        # ── 바운딩 박스 그리기 ──
        for det in detections:
            if det["conf"] < 0.6:
                continue

            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            class_name = det["label"]
            source = det["source"]

            text = f"{class_name} {conf:.2f}"
            color = (0, 255, 0) if source == "growth" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ── MJPEG 인코딩 & 스트리밍 ──
        ret2, buffer = cv2.imencode(".jpg", frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 메인 페이지 (로그인 필요)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")

# ─────────────────────────────────────────────────────────────────────────────
# 비디오 스트림 엔드포인트 (로그인 필요)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─────────────────────────────────────────────────────────────────────────────
# 최근 감지 내역을 JSON으로 반환 (로그인 필요)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/data")
@login_required
def data():
    with lock:
        history_copy = []
        for ts, summary in detection_history:
            growth_info = summary["growth"]
            cond_info   = summary["condition"]

            if growth_info and growth_info["label"] is not None:
                growth_label = growth_info["label"]
                growth_conf = growth_info["conf"]
            else:
                growth_label = None
                growth_conf = None

            if cond_info and cond_info["label"] is not None:
                condition_label = cond_info["label"]
                condition_conf = cond_info["conf"]
            else:
                condition_label = None
                condition_conf = None

            history_copy.append({
                "timestamp": ts,
                "growth_label": growth_label,
                "growth_conf": growth_conf,
                "condition_label": condition_label,
                "condition_conf": condition_conf
            })

    return jsonify(history_copy)

# ─────────────────────────────────────────────────────────────────────────────
# 센서 상태를 JSON으로 반환 (로그인 필요)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/sensor_status")
@login_required
def sensor_status():
    """
    최근 읽은 DHT11, BH1750, 수위 센서 값을 JSON 형태로 반환.
    """
    return jsonify(sensor_data)

# ─────────────────────────────────────────────────────────────────────────────
# 앱 시작 시 데이터베이스 초기화 및 센서 스레드 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) users 테이블 생성
    with app.app_context():
        init_db()

    # 2) 센서 읽기용 스레드 시작 (데몬 스레드로, Flask 종료 시 자동 종료)
    sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
    sensor_thread.start()

    # 3) Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, threaded=True)
