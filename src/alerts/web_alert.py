"""Bản AlertManager dành cho web (không dùng pygame).

Logic chọn cảnh báo, cooldown và ngắt cảnh báo giữ nguyên như
src/alerts/alert_manager.py. Khác biệt duy nhất: trình duyệt mới là nơi phát
âm thanh, nên server không biết âm thanh còn đang phát hay không. Ta ước lượng
bằng AUDIO_SECONDS (độ dài gần đúng của mỗi file mp3 cảnh báo).
"""
import shutil
import tempfile
import time
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT_DIR / "outputs" / "audio"

# Thư mục chứa bản sao mp3 gửi cho trình duyệt (mỗi lần một tên file mới
# để trình duyệt chắc chắn tự phát lại, kể cả khi cùng một cảnh báo).
SERVE_DIR = Path(tempfile.gettempdir()) / "blind_assistant_alerts"

AUDIO_SECONDS = 2.5

ALERT_PRIORITY = {
    "Pothole": 1,
    "Obstacle": 2,
    "Road-barrier": 3,
    "Railway": 4,
    "Stairs": 5,
    "Vehicle": 6,
    "Person": 7,
    "Pole": 8,
}
POSITION_CODES = {
    "Bên trái": "Left",
    "Ở giữa": "Center",
    "Bên phải": "Right",
}
THREAT_PRIORITY = {"Cao": 1, "Trung bình": 2, "Thấp": 3}
THREAT_CODES = {"Cao": "Cao", "Trung bình": "TrungBinh", "Thấp": "Thap"}


class WebAlertManager:
    def __init__(self, cooldown=3, confidence_threshold=0.5):
        self.cooldown = cooldown
        self.confidence_threshold = confidence_threshold
        self.last_alert_time = {}
        self.last_alert_threat = {}
        self.current_audio_threat = None
        self.busy_until = 0.0

    def select_alert(self, detections):
        candidates = [
            d for d in detections
            if d["class_name"] in ALERT_PRIORITY
            and d["confidence"] >= self.confidence_threshold
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda d: (
                THREAT_PRIORITY.get(d["threat_level"], 99),
                ALERT_PRIORITY[d["class_name"]],
            )
        )
        return candidates[0]

    def process(self, detections, now=None):
        """Trả về (đường_dẫn_mp3_cho_trình_duyệt, detection_được_chọn).

        Nếu chưa cần phát cảnh báo thì trả về (None, None).
        """
        selected = self.select_alert(detections)
        if selected is None:
            return None, None

        class_name = selected["class_name"]
        position_code = POSITION_CODES.get(selected["position"])
        threat_level = selected["threat_level"]
        threat_code = THREAT_CODES.get(threat_level)
        if position_code is None or threat_code is None:
            return None, None

        now = time.monotonic() if now is None else now
        alert_key = (class_name, position_code)
        last_time = self.last_alert_time.get(alert_key, 0)

        previous_threat = self.last_alert_threat.get(alert_key)
        current_priority = THREAT_PRIORITY[threat_level]
        threat_increased = (
            previous_threat is not None
            and current_priority < THREAT_PRIORITY[previous_threat]
        )

        audio_playing = now < self.busy_until
        if not audio_playing:
            self.current_audio_threat = None
        interrupt_current = (
            self.current_audio_threat is not None
            and current_priority < THREAT_PRIORITY[self.current_audio_threat]
        )

        if not threat_increased and not interrupt_current:
            if now - last_time < self.cooldown:
                return None, None

        # Âm thanh đang phát và không đủ ưu tiên để ngắt -> bỏ qua
        if audio_playing and not interrupt_current:
            return None, None

        source = AUDIO_DIR / f"{class_name}_{position_code}_{threat_code}.mp3"
        if not source.exists():
            print(f"Khong tim thay am thanh: {source}")
            return None, None

        served = copy_for_browser(source)

        self.last_alert_time[alert_key] = now
        self.last_alert_threat[alert_key] = threat_level
        self.current_audio_threat = threat_level
        self.busy_until = now + AUDIO_SECONDS
        return served, selected


def copy_for_browser(source):
    """Sao chép mp3 sang tên file duy nhất và dọn các file cũ."""
    SERVE_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = time.time() - 120
    for old in SERVE_DIR.glob("*.mp3"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass

    target = SERVE_DIR / f"{source.stem}_{uuid.uuid4().hex[:8]}.mp3"
    shutil.copyfile(source, target)
    return str(target)
