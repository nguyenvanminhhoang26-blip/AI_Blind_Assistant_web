"""AI Blind Assistant - bản web (Streamlit).

Camera trình duyệt -> YOLOv8n (best.pt) -> vị trí / khoảng cách / mức nguy hiểm
-> lọc qua nhiều frame -> chọn cảnh báo -> phát giọng nói tiếng Việt (mp3).
"""
import base64
import queue
import threading
import uuid
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.alerts.web_alert import WebAlertManager
from src.detection.detection import detect_objects, load_model
from src.detection.temporal_filter import TemporalFilter

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "weights" / "best.pt"
MAX_WIDTH = 640

# cv2.putText không vẽ được chữ có dấu -> dùng nhãn không dấu trên hình
POSITION_EN = {"Bên trái": "Left", "Ở giữa": "Center", "Bên phải": "Right"}
DISTANCE_EN = {"Xa": "Far", "Gần": "Near", "Rất gần": "Very near"}
THREAT_EN = {"Thấp": "Low", "Trung bình": "Medium", "Cao": "High"}
THREAT_COLOR_BGR = {
    "Thấp": (0, 200, 0),
    "Trung bình": (0, 200, 255),
    "Cao": (0, 0, 255),
}


@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH)), threading.Lock()


model, model_lock = get_model()


def run_detection(bgr):
    height, width = bgr.shape[:2]
    if width > MAX_WIDTH:
        bgr = cv2.resize(bgr, (MAX_WIDTH, int(height * MAX_WIDTH / width)))
    with model_lock:
        detections = detect_objects(model, bgr)
    return bgr, detections


def draw(bgr, detections):
    """Vẽ box lên ảnh BGR, trả về ảnh BGR."""
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        color = THREAT_COLOR_BGR.get(d["threat_level"], (0, 255, 0))
        label = (
            f'{d["class_name"]} {POSITION_EN.get(d["position"], "")} '
            f'{DISTANCE_EN.get(d["distance_level"], "")} '
            f'{THREAT_EN.get(d["threat_level"], "")} {d["confidence"]:.2f}'
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def describe(selected):
    return (
        f'{selected["class_name"]} - {selected["position"]} - '
        f'{selected["distance_level"]} - Nguy hiểm: {selected["threat_level"]}'
    )


def play_audio(slot, audio_path):
    """Phát mp3 tự động. Mỗi lần là một iframe mới nên luôn phát lại."""
    data = base64.b64encode(Path(audio_path).read_bytes()).decode()
    html = (
        f'<audio autoplay src="data:audio/mpeg;base64,{data}"></audio>'
        f"<!-- {uuid.uuid4().hex} -->"
    )
    with slot.container():
        components.html(html, height=0)


def get_rtc_configuration():
    """STUN mặc định; thêm TURN nếu có khai báo [turn] trong Secrets."""
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn = None
    try:
        if "turn" in st.secrets:
            turn = st.secrets["turn"]
    except Exception:
        turn = None
    if turn:
        urls = turn["urls"]
        if isinstance(urls, str):
            urls = [urls]
        ice_servers.append({
            "urls": list(urls),
            "username": turn["username"],
            "credential": turn["credential"],
        })
    return {"iceServers": ice_servers}, turn is not None


st.set_page_config(page_title="AI Blind Assistant", page_icon="🦯",
                   layout="wide")
st.title("🦯 AI Blind Assistant")
st.caption(
    "Nhận diện chướng ngại vật qua camera và cảnh báo bằng giọng nói tiếng "
    "Việt. Hệ thống chỉ đưa ra cảnh báo tham khảo, không thay thế gậy dẫn "
    "đường hay sự hỗ trợ của người khác."
)

tab_live, tab_photo = st.tabs(["📹 Camera trực tiếp", "📷 Chụp / tải ảnh"])

# ---------- Tab ảnh (chạy trước vì tab live có vòng lặp chờ cảnh báo) ----------
with tab_photo:
    st.write("Chụp ảnh bằng camera hoặc tải ảnh lên để thử nhận diện.")
    shot = st.camera_input("Chụp ảnh")
    upload = st.file_uploader("…hoặc tải ảnh lên", type=["jpg", "jpeg", "png"])
    source = shot or upload
    if source is not None:
        raw = np.frombuffer(source.getvalue(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Không đọc được ảnh.")
        else:
            bgr, detections = run_detection(image)
            st.image(cv2.cvtColor(draw(bgr, detections), cv2.COLOR_BGR2RGB),
                     caption="Kết quả nhận diện")
            manager = WebAlertManager(cooldown=0, confidence_threshold=0.5)
            audio_path, selected = manager.process(detections)
            if not detections:
                st.info("Không phát hiện đối tượng nào.")
            elif selected is None:
                st.info(f"Phát hiện {len(detections)} đối tượng, "
                        "không cần cảnh báo.")
            else:
                st.warning(f"⚠️ {describe(selected)}")
                if audio_path:
                    play_audio(st.empty(), audio_path)

# ---------- Tab camera trực tiếp ----------
with tab_live:
    st.write("Bấm **START**, cho phép camera và bật loa để nghe cảnh báo.")

    if "live" not in st.session_state:
        st.session_state.live = {
            "filter": TemporalFilter(history_size=5),
            "alert": WebAlertManager(cooldown=3, confidence_threshold=0.5),
            "queue": queue.Queue(),
        }
    live = st.session_state.live

    def video_frame_callback(frame):
        bgr = frame.to_ndarray(format="bgr24")
        bgr, detections = run_detection(bgr)

        live["filter"].update(detections)
        confirmed = live["filter"].filter_detections(detections)
        audio_path, selected = live["alert"].process(confirmed)
        if audio_path:
            live["queue"].put((audio_path, describe(selected)))

        return av.VideoFrame.from_ndarray(draw(bgr, detections),
                                          format="bgr24")

    rtc_configuration, has_turn = get_rtc_configuration()
    if not has_turn:
        st.info(
            "Nếu camera cứ báo \"Connection is taking longer than expected\", "
            "mạng của bạn đang chặn WebRTC. Hãy dùng tab **Chụp / tải ảnh**, "
            "hoặc cấu hình máy chủ TURN trong Secrets."
        )

    ctx = webrtc_streamer(
        key="blind-assistant",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        rtc_configuration=rtc_configuration,
        async_processing=True,
    )

    status = st.empty()
    audio_slot = st.empty()
    while ctx.state.playing:
        try:
            audio_path, text = live["queue"].get(timeout=1)
        except queue.Empty:
            continue
        status.warning(f"⚠️ {text}")
        play_audio(audio_slot, audio_path)
