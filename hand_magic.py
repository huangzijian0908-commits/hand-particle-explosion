from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

PARTICLE_COUNT = 4000
EXPLOSION_FORCE = 42.0
HAND_INFLUENCE = 0.018
FRICTION = 0.945
MAX_SPEED = 18.0
WINDOW_NAME = "Hand Particle Explosion"


@dataclass
class HandState:
    palm: tuple[float, float] | None = None
    closed: bool = False
    visible: bool = False


class ParticleField:
    def __init__(self, width: int, height: int, count: int) -> None:
        self.width = width
        self.height = height
        self.count = count
        self.positions = np.column_stack(
            (
                np.random.uniform(0, width, count),
                np.random.uniform(0, height, count),
            )
        ).astype(np.float32)
        self.home = self.positions.copy()
        self.velocity = np.random.uniform(-1, 1, (count, 2)).astype(np.float32)
        self.radius = np.random.uniform(1.0, 2.4, count).astype(np.float32)
        self.colors = np.column_stack(
            (
                np.random.uniform(180, 255, count),
                np.random.uniform(90, 230, count),
                np.random.uniform(80, 255, count),
            )
        ).astype(np.uint8)

    def resize(self, width: int, height: int) -> None:
        scale_x = width / max(self.width, 1)
        scale_y = height / max(self.height, 1)
        self.width = width
        self.height = height
        self.positions[:, 0] *= scale_x
        self.positions[:, 1] *= scale_y
        self.home[:, 0] *= scale_x
        self.home[:, 1] *= scale_y

    def update(self, hand: HandState, fist_energy: float) -> None:
        self.velocity += (self.home - self.positions) * 0.0022

        if hand.palm is not None:
            target = np.array(hand.palm, dtype=np.float32)
            delta = target - self.positions
            dist_sq = np.maximum(np.sum(delta * delta, axis=1, keepdims=True), 120.0)
            attraction = HAND_INFLUENCE * (1.0 + fist_energy * 0.9)
            self.velocity += (delta / dist_sq) * attraction * 900.0

        self.velocity *= FRICTION
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        fast = speed[:, 0] > MAX_SPEED
        self.velocity[fast] *= MAX_SPEED / speed[fast]
        self.positions += self.velocity

        over_x = (self.positions[:, 0] < -20) | (self.positions[:, 0] > self.width + 20)
        over_y = (self.positions[:, 1] < -20) | (self.positions[:, 1] > self.height + 20)
        bounced = over_x | over_y
        self.positions[:, 0] = np.clip(self.positions[:, 0], -10, self.width + 10)
        self.positions[:, 1] = np.clip(self.positions[:, 1], -10, self.height + 10)
        self.velocity[bounced] *= -0.45

    def explode(self, origin: tuple[float, float]) -> None:
        center = np.array(origin, dtype=np.float32)
        delta = self.positions - center
        distance = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 18.0)
        direction = delta / distance
        falloff = np.minimum(1.0, 420.0 / distance)
        random_force = np.random.uniform(0.45, 1.35, (self.count, 1)).astype(np.float32)
        sideways = np.column_stack((-direction[:, 1], direction[:, 0])).astype(np.float32)
        noise = np.random.uniform(-0.42, 0.42, (self.count, 1)).astype(np.float32)
        burst_direction = direction + sideways * noise
        self.velocity += burst_direction * EXPLOSION_FORCE * falloff * random_force

    def draw(self, frame: np.ndarray, fist_energy: float) -> None:
        glow = np.zeros_like(frame)
        size_boost = int(1 + fist_energy * 2)
        for index, (x, y) in enumerate(self.positions.astype(np.int32)):
            if 0 <= x < self.width and 0 <= y < self.height:
                radius = int(self.radius[index]) + size_boost
                color = tuple(int(v) for v in self.colors[index])
                cv2.circle(glow, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow, 0.88, frame, 1.0, 0, dst=frame)


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    message = (
        f"Missing MediaPipe model: {MODEL_PATH}\n\n"
        "Run this first:\n"
        "  python python/download_model.py\n"
    )
    raise FileNotFoundError(message)


def landmark_distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def estimate_closed_fist(landmarks) -> bool:
    palm_x = sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5
    palm_y = sum(landmarks[i].y for i in (0, 5, 9, 13, 17)) / 5
    palm_size = max(landmark_distance(landmarks[0], landmarks[9]), 0.001)

    folded = 0
    for tip, joint in zip((8, 12, 16, 20), (6, 10, 14, 18)):
        tip_distance = math.hypot(landmarks[tip].x - palm_x, landmarks[tip].y - palm_y) / palm_size
        joint_distance = math.hypot(landmarks[joint].x - palm_x, landmarks[joint].y - palm_y) / palm_size
        if tip_distance < joint_distance * 1.08 or tip_distance < 0.82:
            folded += 1

    thumb_distance = math.hypot(landmarks[4].x - palm_x, landmarks[4].y - palm_y) / palm_size
    return folded >= 3 and thumb_distance < 1.05


def read_hand_state(result, width: int, height: int) -> HandState:
    if not result.hand_landmarks:
        return HandState()

    landmarks = result.hand_landmarks[0]
    palm = landmarks[9]
    return HandState(
        palm=(palm.x * width, palm.y * height),
        closed=estimate_closed_fist(landmarks),
        visible=True,
    )


def draw_hud(frame: np.ndarray, hand: HandState, fps: float) -> None:
    status = "Fist" if hand.closed else "Open hand" if hand.visible else "No hand"
    lines = [
        "Hand Particle Explosion",
        f"{status}   {fps:>2.0f} fps",
        "Close your hand to explode. Press Q to quit.",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (8, 10, 16), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 244, 255), 1, cv2.LINE_AA)
        y += 30

    if hand.palm is not None:
        center = (int(hand.palm[0]), int(hand.palm[1]))
        cv2.circle(frame, center, 34 if hand.closed else 24, (125, 242, 194), 2, cv2.LINE_AA)
        cv2.circle(frame, center, 5, (255, 170, 80), -1, cv2.LINE_AA)


def main() -> None:
    ensure_model_exists()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("The webcam opened, but no frame was received.")

    height, width = frame.shape[:2]
    field = ParticleField(width, height, PARTICLE_COUNT)
    previous_closed = False
    fist_energy = 0.0
    last_time = time.perf_counter()
    fps = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if width != field.width or height != field.height:
                field.resize(width, height)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand = read_hand_state(result, width, height)

            if hand.closed and not previous_closed and hand.palm is not None:
                field.explode(hand.palm)
            previous_closed = hand.closed
            fist_energy = min(1.0, fist_energy + 0.14) if hand.closed else max(0.0, fist_energy - 0.07)

            canvas = np.zeros_like(frame)
            canvas[:] = (8, 9, 13)
            field.update(hand, fist_energy)
            field.draw(canvas, fist_energy)
            draw_hud(canvas, hand, fps)

            now = time.perf_counter()
            frame_time = max(now - last_time, 0.001)
            last_time = now
            fps = fps * 0.9 + (1.0 / frame_time) * 0.1

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
