import argparse
import re
import sys
import time
import random
import os
import threading
import subprocess

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from openai import OpenAI

# =========================
# LLM 설정
# =========================
client = OpenAI(api_key=api_key_h)
 
LLM_MODEL = "gpt-4o"
 
SYSTEM_PROMPT = (
    "당신은 차량 내 스마트 AI 비서입니다. 운전자의 위험 행동에 대해 경고 문장을 생성합니다. "
    "문장은 10~15 단어, tts 사용 시 2~5초 사이로 구성하세요. "
    "경고 메시지는 반드시 두 파트로 구성합니다: "
    "① 위험 상황 인식 (현재 상태를 짧게 언급) + ② 구체적 후속 행동 제안 (지금 당장 할 수 있는 행동). "
    "예시: '휴대폰 사용 중 사고 위험이 높습니다, 지금 거치대에 올려두세요.' "
    "주의 단계는 자연스럽고 친근하게, 위험 단계는 단호하고 간결하게 작성하세요."
)
 
# 경고 단계별 프롬프트
USER_PROMPT_WARNING = {
    1: (
        "운전자가 휴대폰을 조작하고 있습니다. "
        "부드럽게 위험을 알리고, 지금 바로 할 수 있는 구체적인 행동(예: 내려놓기, 거치대 사용, 잠시 후 확인 등)을 함께 제안하는 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
    2: (
        "운전자가 경고 후에도 계속 휴대폰을 조작하고 있습니다. "
        "더 단호한 톤으로 위험을 강조하고, 즉각 취해야 할 구체적인 행동을 명확히 지시하는 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
    3: (
        "운전자가 두 번의 경고에도 휴대폰 조작을 멈추지 않습니다. "
        "매우 긴박하고 강한 톤으로, 지금 즉시 멈춰야 한다는 것과 취해야 할 행동을 포함한 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
}
 
USER_PROMPT_POSITIVE = (
    "운전자가 경고를 듣고 휴대폰을 내려놓았습니다. "
    "안전한 선택을 칭찬하고, 앞으로도 안전 운전을 유지할 수 있도록 격려하는 "
    "따뜻하고 짧은 한국어 문장 1개를 생성하세요."
)
 
# 백업 문장 (단계별)
BACKUP_WARNING = {
    1: [
        "휴대폰 사용은 사고 위험을 높입니다, 잠깐 거치대에 올려두세요.",
        "운전 중 휴대폰 조작은 위험합니다, 지금 내려놓고 전방을 보세요.",
        "주의가 분산되고 있습니다, 휴대폰을 내려놓고 운전에 집중하세요.",
    ],
    2: [
        "계속된 휴대폰 사용은 매우 위험합니다, 즉시 내려놓고 전방을 주시하세요.",
        "경고합니다, 지금 당장 휴대폰을 내려놓고 두 손으로 운전하세요.",
        "심각한 사고 위험 상황입니다, 즉각 휴대폰을 놓고 전방을 보세요.",
    ],
    3: [
        "즉시 멈추세요! 휴대폰을 내려놓고 지금 당장 전방을 주시하세요!",
        "위험합니다! 지금 즉시 휴대폰을 좌석에 내려놓으세요!",
        "긴급 경고! 휴대폰을 놓고 두 손으로 핸들을 잡으세요!",
    ],
}
 
BACKUP_POSITIVE = [
    "잘 하셨어요, 이렇게 전방을 주시하며 안전 운전 계속해 주세요.",
    "훌륭합니다! 휴대폰은 도착 후 확인하고 지금처럼 운전에 집중해 주세요.",
    "좋아요, 안전한 선택이었습니다. 계속 이렇게 운전해 주세요.",
]
 
# 중복 방지 캐시
RECENT_CACHE = []
CACHE_MAX = 8
 
# =========================
# HitL 상태 관리
# =========================
class WarningState:
    """휴대폰 감지 및 HitL 피드백 루프 상태 머신"""
 
    # 상태 정의
    IDLE        = "IDLE"         # 정상 주행, 감지 없음
    DETECTING   = "DETECTING"    # 휴대폰 감지 시작 (2초 카운팅 중)
    WARNING     = "WARNING"      # 경고 발화 후 개선 판단 window (5초)
    IMPROVED    = "IMPROVED"     # 개선 감지 → 긍정 강화 발화
    DONE        = "DONE"         # 루프 완료 (최대 3회 or 개선)
 
    # 기준 시간 (선행 연구 기반)
    DETECT_THRESHOLD_SEC  = 2.0  # 휴대폰 지속 감지 ≥ 2초 → 경고 트리거
    IMPROVE_WINDOW_SEC    = 5.0  # 경고 후 5초 내 미감지 → 개선으로 판정
    MAX_WARNING_COUNT     = 3    # 최대 경고 반복 횟수
    POST_ACTION_COOLDOWN  = 8.0  # 개선/완료 후 재감지 쿨다운
    # 내려놓는 동작 중 오발화 방지: 연속 N프레임 미감지여야 개선으로 판정
    IMPROVE_CONFIRM_FRAMES = 8   # 약 0.3초 (30fps 기준)
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.state               = self.IDLE
        self.detect_start_time   = None
        self.warning_issued_time = None
        self.warning_count       = 0
        self.action_done_time    = None
        self.no_detect_frames    = 0   # 연속 미감지 프레임 카운터
 
    def is_in_cooldown(self, now: float) -> bool:
        """개선/완료 직후 쿨다운 중인지 확인"""
        if self.action_done_time is None:
            return False
        return (now - self.action_done_time) < self.POST_ACTION_COOLDOWN
 
    def update(self, phone_detected: bool, now: float):
        """
        매 프레임 호출.
        반환값: ("action", payload)
            action = None | "warn" | "positive" | "log_done"
            payload = warning_count(int) | None
        """
 
        # ── 쿨다운 중이면 아무것도 하지 않음 ──────────────────────────
        if self.state in (self.IMPROVED, self.DONE):
            if self.is_in_cooldown(now):
                return None, None
            else:
                self.reset()  # 쿨다운 끝 → 초기화
 
        # ── IDLE: 감지 대기 ────────────────────────────────────────────
        if self.state == self.IDLE:
            if phone_detected:
                self.detect_start_time = now
                self.state = self.DETECTING
            return None, None
 
        # ── DETECTING: 2초 지속 감지 확인 ─────────────────────────────
        if self.state == self.DETECTING:
            if not phone_detected:
                # 2초 미만에 사라짐 → 초기화
                self.reset()
                return None, None
 
            elapsed = now - self.detect_start_time
            if elapsed >= self.DETECT_THRESHOLD_SEC:
                # ★ 2초 이상 지속 → 경고 트리거
                self.warning_count += 1
                self.warning_issued_time = now
                self.state = self.WARNING
                return "warn", self.warning_count
            return None, None
 
        # ── WARNING: 개선 판단 window (5초) ───────────────────────────
        if self.state == self.WARNING:
            window_elapsed = now - self.warning_issued_time
 
            if not phone_detected:
                # 내려놓는 동작 중 순간 미감지로 오발화 방지:
                # 연속 IMPROVE_CONFIRM_FRAMES 프레임 동안 미감지여야 개선 판정
                self.no_detect_frames += 1
                if self.no_detect_frames >= self.IMPROVE_CONFIRM_FRAMES:
                    # ★ 연속 미감지 확인 → 개선 판정
                    self.state = self.IMPROVED
                    self.action_done_time = now
                    return "positive", None
                return None, None
            else:
                # 감지 중이면 카운터 리셋
                self.no_detect_frames = 0
 
            if window_elapsed >= self.IMPROVE_WINDOW_SEC:
                if self.warning_count >= self.MAX_WARNING_COUNT:
                    self.state = self.DONE
                    self.action_done_time = now
                    return "log_done", self.warning_count
                else:
                    self.warning_count += 1
                    self.warning_issued_time = now
                    self.no_detect_frames = 0
                    return "warn", self.warning_count
 
            return None, None
 
        return None, None
 
 
# =========================
# 문장 후처리 / 중복 제어
# =========================
def _dedup(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()
 
def _post_one_sentence(text: str) -> str:
    text = _dedup(text)
    parts = re.split(r"(?<=[\.!?…])\s+", text)
    first = parts[0] if parts else text
    return first[:120]
 
def _not_duplicate(text: str) -> bool:
    return _dedup(text) not in RECENT_CACHE
 
def _remember(text: str):
    t = _dedup(text)
    if t:
        RECENT_CACHE.append(t)
        if len(RECENT_CACHE) > CACHE_MAX:
            del RECENT_CACHE[0]
 
def backup_utterance(pool: list) -> str:
    for _ in range(5):
        cand = random.choice(pool)
        if _not_duplicate(cand):
            return cand
    return pool[0]
 
 
# =========================
# LLM 호출
# =========================
def generate_warning(level: int) -> str:
    """level: 1(부드러움) / 2(단호함) / 3(긴박함)"""
    prompt = USER_PROMPT_WARNING.get(level, USER_PROMPT_WARNING[3])
    backup_pool = BACKUP_WARNING.get(level, BACKUP_WARNING[3])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=60,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        text = _post_one_sentence(text)
        if not text or not _not_duplicate(text):
            text = backup_utterance(backup_pool)
        _remember(text)
        return text
    except Exception as e:
        print(f"⚠️ LLM 실패 (경고 level={level}): {e}")
        text = backup_utterance(backup_pool)
        _remember(text)
        return text
 
 
def generate_positive() -> str:
    """긍정 강화 메시지 생성"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_POSITIVE},
            ],
            max_tokens=60,
        )
        text = resp.choices[0].message.content if resp and resp.choices else ""
        text = _post_one_sentence(text)
        if not text or not _not_duplicate(text):
            text = backup_utterance(BACKUP_POSITIVE)
        _remember(text)
        return text
    except Exception as e:
        print(f"⚠️ LLM 실패 (긍정 강화): {e}")
        text = backup_utterance(BACKUP_POSITIVE)
        _remember(text)
        return text
 
 
# =========================
# 오디오
# =========================
def play_alert_sound():
    def _play():
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"경고음 재생 실패: {e}")
    threading.Thread(target=_play, daemon=True).start()
 
 
def speak(text: str, rate: int = 180):
    def _double_alert():
        play_alert_sound()
        time.sleep(0.2)
        play_alert_sound()
    threading.Thread(target=_double_alert, daemon=True).start()
    time.sleep(0.3)
    os.system(f"say -r {int(rate)} '{text}'")
 
 
# =========================
# 시각화
# =========================
def visualize(image, detection_result) -> np.ndarray:
    TEXT_COLOR = (0, 255, 0)
    vis_image = np.copy(image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        cv2.rectangle(vis_image,
                      (bbox.origin_x, bbox.origin_y),
                      (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                      TEXT_COLOR, 3)
        cat = detection.categories[0]
        label = f"{cat.category_name} ({cat.score:.2f})"
        pos = (10 + bbox.origin_x, bbox.origin_y - 10)
        if pos[1] < 10:
            pos = (10 + bbox.origin_x, bbox.origin_y + 20)
        cv2.putText(vis_image, label, pos,
                    cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR, 1)
    return vis_image
 
 
# =========================
# 메인 루프
# =========================
def run(model_path: str, camera_id: int, width: int, height: int):
    fps_avg_frame_count = 10
 
    counter = 0
    fps = 0.0
    fps_timer = time.time()
 
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"ERROR: 웹캠 열기 실패 (카메라 ID {camera_id})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
 
    detection_result_list = []
 
    def visualize_callback(result: vision.ObjectDetectorResult,
                           output_image: mp.Image, timestamp_ms: int):
        result.timestamp_ms = timestamp_ms
        detection_result_list.append(result)
 
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        score_threshold=0.5,
        result_callback=visualize_callback,
    )
    try:
        detector = vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        sys.exit(f"ERROR: MediaPipe 초기화 실패: {e}")
 
    ws = WarningState()
 
    print("=== HitL DMS 시작 ===")
    print(f"  - 감지 임계값 : {WarningState.DETECT_THRESHOLD_SEC}초 지속 감지")
    print(f"  - 개선 판단   : 경고 후 {WarningState.IMPROVE_WINDOW_SEC}초 내 연속 {WarningState.IMPROVE_CONFIRM_FRAMES}프레임 미감지")
    print(f"  - 최대 경고   : {WarningState.MAX_WARNING_COUNT}회")
    print("  - ESC 키로 종료\n")
 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ERROR: 프레임 읽기 실패.")
            break
 
        counter += 1
        frame = cv2.flip(frame, 1)
        now = time.time()
 
        # ── 탐지 ──────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_image, counter)
        current_frame = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
 
        # FPS
        if counter % fps_avg_frame_count == 0:
            fps = fps_avg_frame_count / (now - fps_timer)
            fps_timer = now
        cv2.putText(current_frame, f"FPS: {fps:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
 
        # ── 결과 처리 ─────────────────────────────────────────────────
        phone_detected_this_frame = False
 
        if detection_result_list:
            result = detection_result_list.pop(0)
            vis = visualize(current_frame, result)
 
            for det in result.detections:
                if det.categories[0].category_name == "cell phone":
                    phone_detected_this_frame = True
                    break
 
            # 상태 표시
            status_color = (0, 255, 0)  # 기본 녹색
            status_text  = "MONITORING"
            if ws.state == WarningState.DETECTING:
                elapsed_detect = now - ws.detect_start_time
                status_text  = f"DETECTING {elapsed_detect:.1f}s / {WarningState.DETECT_THRESHOLD_SEC}s"
                status_color = (0, 165, 255)  # 주황
            elif ws.state == WarningState.WARNING:
                elapsed_window = now - ws.warning_issued_time
                remain = max(0, WarningState.IMPROVE_WINDOW_SEC - elapsed_window)
                status_text  = f"WARNING #{ws.warning_count} | improve window {remain:.1f}s"
                status_color = (0, 0, 255)  # 빨강
            elif ws.state in (WarningState.IMPROVED, WarningState.DONE):
                status_text  = "COOLDOWN"
                status_color = (255, 200, 0)  # 하늘
 
            cv2.putText(vis, status_text,
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow("HitL DMS", vis)
        else:
            cv2.putText(current_frame, "MONITORING",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("HitL DMS", current_frame)
 
        # ── HitL 상태 머신 업데이트 ───────────────────────────────────
        action, payload = ws.update(phone_detected_this_frame, now)
 
        if action == "warn":
            level = payload  # 1 / 2 / 3
            print(f"\n📱 [경고 #{level}] 휴대폰 {WarningState.DETECT_THRESHOLD_SEC}초 이상 감지")
            text = generate_warning(level)
            print(f"🗣️  {text}")
            threading.Thread(target=speak, args=(text,), daemon=True).start()
 
        elif action == "positive":
            print(f"\n✅ [개선 감지] 경고 후 {WarningState.IMPROVE_WINDOW_SEC}초 내 휴대폰 내려놓음 → 긍정 강화")
            text = generate_positive()
            print(f"🗣️  {text}")
            threading.Thread(target=speak, args=(text, 160), daemon=True).start()
 
        elif action == "log_done":
            print(f"\n⛔ [루프 종료] 최대 {WarningState.MAX_WARNING_COUNT}회 경고 후에도 미개선 → 로그 기록")
 
        # ESC 종료
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n프로그램을 종료합니다.")
            break
 
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
 
 
# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="HitL DMS: 휴대폰 감지 → 피드백 루프 경고 시스템"
    )
    parser.add_argument("--model",       default="efficientdet_lite0.tflite")
    parser.add_argument("--cameraId",    type=int, default=0)
    parser.add_argument("--frameWidth",  type=int, default=640)
    parser.add_argument("--frameHeight", type=int, default=480)
    args = parser.parse_args()
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight)
 
 
if __name__ == "__main__":
    main()
