import os
import sqlite3
import cv2
import time
import threading
import smtplib
import concurrent.futures
from email.mime.text import MIMEText
from functools import wraps
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash,
    Response, jsonify, g
)
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Flask 앱 설정
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLite 데이터베이스 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드 (YOLOv5 custom .pt 파일 위치 수정)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# growth.pt와 condition.pt가 위치한 경로 (실제 경로로 수정 필요)
MODEL_DIR = "/home/pi/yolov5/yolov5-env"

# growth.pt: 토마토 생장 단계 (green, half_red, red) 판별용
growth_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=os.path.join(MODEL_DIR, "growth.pt"),
    force_reload=False
).to(DEVICE)
growth_model.eval()

# condition.pt: 토마토 병충해 (tomato_blight) 판별용
condition_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=os.path.join(MODEL_DIR, "condition.pt"),
    force_reload=False
).to(DEVICE)
condition_model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 이메일(SMTP) 설정 (Gmail 예시)
# ─────────────────────────────────────────────────────────────────────────────
# 실제 배포 시에는 환경변수나 설정 파일로 관리하세요.
MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
MAIL_USERNAME = "ericsungho@gmail.com"       # <-- 발신할 Gmail 주소
MAIL_PASSWORD = "xspm amkw fchq zgre"                  # <-- Gmail 앱 비밀번호 (2단계 인증 시 App Password)
MAIL_USE_TLS = True


def send_email(to_address: str, subject: str, body: str):
    """
    간단한 SMTP 이메일 전송 함수.
    Gmail SMTP 서버 사용 예시입니다.
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
        print(f"✔ 이메일 발송 성공: to={to_address}, subject={subject}")
    except Exception as e:
        print(f"✘ 이메일 발송 실패: {e}")


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
    # 이미 로그인 된 상태라면 index로 리다이렉트
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        email = request.form.get("email", "").strip()

        # 간단한 유효성 검사
        if not username or not password or not password2 or not email:
            error = "아이디, 비밀번호, 이메일을 모두 입력해주세요."
        elif password != password2:
            error = "비밀번호가 일치하지 않습니다."
        else:
            db = get_db()
            cursor = db.cursor()
            # 같은 username 또는 email이 있는지 확인
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone() is not None:
                error = "이미 사용 중인 아이디 또는 이메일입니다."
            else:
                # 비밀번호 해시 생성
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
    # 이미 로그인 상태라면 index로
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
            # 로그인 성공
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
# YOLOv5 전처리 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_yolo(frame):
    """
    OpenCV로 읽은 BGR frame을 YOLOv5가 기대하는 RGB numpy array 형태로 변환.
    - YOLOv5 Hub API에 np.ndarray를 바로 넣으면 내부적으로 letterbox & 텐서 변환을 수행합니다.
    """
    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 추론용 헬퍼 함수들
# ─────────────────────────────────────────────────────────────────────────────
def _infer_growth(frame):
    """
    growth_model만 돌리고, 결과에 source="growth"를 붙여서 반환합니다.
    """
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
    """
    condition_model만 돌리고, 결과에 source="condition"을 붙여서 반환합니다.
    """
    results = []
    img = preprocess_for_yolo(frame)
    preds = condition_model(img, size=640)
    df = preds.pandas().xyxy[0]

    for _, row in df.iterrows():
        x1, y1 = int(row.xmin), int(row.ymin)
        x2, y2 = int(row.xmax), int(row.ymax)
        conf = float(row.confidence)
        label = row["name"]  # "tomato_blight"
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

    반환 형식: [
        {"label": "green"/"half_red"/"red", "conf": float, "bbox": (x1,y1,x2,y2), "source": "growth"},
        {"label": "tomato_blight", "conf": float, "bbox": (x1,y1,x2,y2), "source": "condition"},
        ...
    ]
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_growth = executor.submit(_infer_growth, frame)
        future_cond   = executor.submit(_infer_condition, frame)

        for future in concurrent.futures.as_completed([future_growth, future_cond]):
            results.extend(future.result())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 비디오 스트리밍 + 감지 데이터 로직
# ─────────────────────────────────────────────────────────────────────────────
CAM_INDEX = 0
detection_history = []
lock = threading.Lock()

# 병충해(blight) 알림, 완숙(red) 알림을 중복 전송하지 않기 위한 플래그
last_blight_sent = False
last_red_sent = False

# 카메라 초기화
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    # 점유 중인 프로세스 강제 종료 후 재시도
    os.system(f"fuser -k /dev/video{CAM_INDEX} 2>/dev/null")
    time.sleep(1)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 카메라 인덱스 {CAM_INDEX}번을 열 수 없습니다.")


def gen_frames():
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

            # 2) growth 모델에서 확신도 가장 높은 객체 찾기
            growth_dets = [det for det in valid_dets if det["source"] == "growth"]
            if growth_dets:
                best_growth = max(growth_dets, key=lambda d: d["conf"])
                growth_label = best_growth["label"]      # "green"/"half_red"/"red"
                growth_conf  = best_growth["conf"]
            else:
                growth_label = None
                growth_conf  = None

            # 3) condition 모델에서 확신도 가장 높은 객체 찾기
            cond_dets = [det for det in valid_dets if det["source"] == "condition"]
            if cond_dets:
                best_cond = max(cond_dets, key=lambda d: d["conf"])
                cond_label = best_cond["label"]          # 일반적으로 "tomato_blight"
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

            # 5) 이메일 발송 로직 (각각의 상황이 필요하다면 적절히 분기)
            try:
                db_conn = sqlite3.connect(DB_PATH)
                db_conn.row_factory = sqlite3.Row
                cursor = db_conn.cursor()
                cursor.execute("SELECT email FROM users")
                users = cursor.fetchall()
                db_conn.close()
            except Exception as e:
                print(f"DB 연결 에러: {e}")
                users = []

            # ─── 병충해 알림 ───
            # cond_label == "tomato_blight"인 경우에만 이메일 발송
            if cond_label == "tomato_blight" and not last_blight_sent:
                subject = "🚨 토마토 병충해 감지 알림"
                body    = "식물이 손상되었습니다. 확인해주세요."
                for u in users:
                    send_email(u["email"], subject, body)
                last_blight_sent = True
            if cond_label != "tomato_blight":
                last_blight_sent = False

            # ─── 완숙(red) 알림 ───
            # growth_label == "red"인 경우에만 이메일 발송
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

        # ── 프레임에 바운딩 박스 그리기 ──
        for det in detections:
            if det["conf"] < 0.6:
                continue

            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            class_name = det["label"]
            source = det["source"]

            text = f"{class_name} {conf:.2f}"
            if source == "growth":
                color = (0, 255, 0)
            else:  # source == "condition"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        # ── MJPEG 인코딩 & 전송 ──
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
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 최근 감지 내역을 JSON으로 반환 (로그인 필요)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/data")
@login_required
def data():
    with lock:
        history_copy = []
        for ts, summary in detection_history:
            # growth, condition 각각에서 label/​conf 확인
            growth_info = summary["growth"]      # {"label": ..., "conf": ...} or None
            cond_info   = summary["condition"]   # {"label": ..., "conf": ...} or None

            # 예시: label 또는 conf가 None이면 null 형태로 넘길 수 있도록 처리
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
                "growth_label": growth_label,       # "green"/"half_red"/"red" 또는 null
                "growth_conf": growth_conf,         # 예: 0.82 또는 null
                "condition_label": condition_label, # "tomato_blight" 또는 null
                "condition_conf": condition_conf    # 예: 0.75 또는 null
            })

    return jsonify(history_copy)


# ─────────────────────────────────────────────────────────────────────────────
# 앱 시작 시 데이터베이스 초기화
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 첫 실행 시 users 테이블 생성 (email 칼럼 포함)
    with app.app_context():
        init_db()
    app.run(host="0.0.0.0", port=5000, threaded=True)
