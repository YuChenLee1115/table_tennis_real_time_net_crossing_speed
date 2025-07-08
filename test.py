#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v12.2 (最終修正版)
# Author: Refactored by AI Assistant
# Description: 高度模組化、職責分離、易於維護與擴展，並提升了跨平台相容性。

import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import csv
import threading
import queue
import concurrent.futures
from dataclasses import dataclass, field

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 10)
except AttributeError:
    cv2.setNumThreads(10)

# ==============================================================================
# 1. Configuration & Data Structures (設定與資料結構)
# ==============================================================================

class Config:
    """集中管理所有靜態設定參數"""
    # Basic Settings
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_TARGET_FPS = 60
    DEFAULT_FRAME_WIDTH = 1280
    DEFAULT_FRAME_HEIGHT = 720
    DEFAULT_TABLE_LENGTH_CM = 94
    OUTPUT_DATA_FOLDER = 'real_time_output'
    
    # Detection Parameters
    DEFAULT_DETECTION_TIMEOUT_S = 0.2
    DEFAULT_ROI_START_RATIO = 0.4
    DEFAULT_ROI_END_RATIO = 0.6
    DEFAULT_ROI_BOTTOM_RATIO = 0.85
    MAX_TRAJECTORY_POINTS = 120

    # Center Line Detection
    MAX_NET_SPEEDS_TO_COLLECT = 30
    NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left'
    AUTO_STOP_AFTER_COLLECTION = False
    EFFECTIVE_CROSSING_COOLDOWN_S = 0.2
    CENTER_ZONE_WIDTH_RATIO = 0.05

    # Perspective Correction
    NEAR_SIDE_WIDTH_CM_DEFAULT = 29
    FAR_SIDE_WIDTH_CM_DEFAULT = 72

    # FMO (Fast Moving Object) Parameters
    MAX_PREV_FRAMES_FMO = 10
    OPENING_KERNEL_SIZE_FMO = (12, 12)
    CLOSING_KERNEL_SIZE_FMO = (25, 25)
    THRESHOLD_VALUE_FMO = 9

    # Ball Detection Parameters
    MIN_BALL_AREA_PX = 10
    MAX_BALL_AREA_PX = 10000
    MIN_BALL_CIRCULARITY = 0.32
    
    # Speed & FPS Calculation
    SPEED_SMOOTHING_FACTOR = 0.3
    KMH_CONVERSION_FACTOR = 0.036
    FPS_SMOOTHING_FACTOR = 0.4
    MAX_FRAME_TIMES_FPS_CALC = 20

    # Visualization
    TRAJECTORY_COLOR_BGR = (0, 0, 255)
    BALL_COLOR_BGR = (0, 255, 255)
    CONTOUR_COLOR_BGR = (255, 0, 0)
    ROI_COLOR_BGR = (0, 255, 0)
    CENTER_LINE_COLOR_BGR = (0, 255, 255)
    FONT_SCALE_VIS = 1
    FONT_THICKNESS_VIS = 2
    VISUALIZATION_DRAW_INTERVAL = 2

    # Threading & Queue
    FRAME_QUEUE_SIZE = 30
    EVENT_BUFFER_SIZE_CENTER_CROSS = 200

    # Debug
    DEBUG_MODE_DEFAULT = False

@dataclass
class BallData:
    position_roi: tuple = None
    position_global: tuple = None
    contour_roi: np.ndarray = None
    timestamp: float = 0.0
    area: float = 0.0
    circularity: float = 0.0

@dataclass
class FrameInfo:
    frame_counter: int
    display_fps: float
    current_speed_kmh: float
    trajectory: deque
    ball_data: BallData = None
    debug_mode: bool = False

@dataclass
class SessionState:
    is_active: bool = False
    session_id: int = 0
    speeds: list = field(default_factory=list)
    times: list = field(default_factory=list)
    last_recorded_speed_kmh: float = 0.0
    output_generated: bool = True

# ==============================================================================
# 2. Core Logic Components (核心邏輯元件)
# ==============================================================================

class FrameReader:
    """在獨立執行緒中讀取攝影機或影片檔案的影像"""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        
        # 修正：移除 macOS 特定的 cv2.CAP_AVFOUNDATION，使其更具跨平台相容性
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            raise IOError(f"無法開啟影像來源: '{self.video_source}'。請確認攝影機已正確連接，或影片檔案路徑無誤。")

        if not self.use_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        self.frame_queue = queue.Queue(maxsize=Config.FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             self.actual_fps = self.target_fps

    def _read_frames(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                self.frame_queue.put((ret, frame))
                if not ret:
                    self.running = False
                    break
            else:
                time.sleep(1.0 / (self.target_fps * 2))

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()

class Detector:
    """專責球體偵測與影像前處理"""
    def __init__(self, frame_width, frame_height, trajectory):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.roi_start_x = int(frame_width * Config.DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(frame_width * Config.DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0
        self.roi_bottom_y = int(frame_height * Config.DEFAULT_ROI_BOTTOM_RATIO)
        
        self.prev_frames_gray_roi = deque(maxlen=Config.MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.CLOSING_KERNEL_SIZE_FMO)
        
        self.trajectory = trajectory
        self.last_detection_timestamp = 0
        
    def detect(self, frame, current_timestamp) -> (BallData, np.ndarray, np.ndarray):
        roi_sub_frame, _ = self._preprocess_frame(frame)
        motion_mask_roi = self._detect_fmo()
        
        ball_data = None
        if motion_mask_roi is not None:
            ball_data = self._find_best_ball_in_roi(motion_mask_roi, current_timestamp)
        
        if ball_data:
            self.last_detection_timestamp = time.monotonic()
            self.trajectory.append((ball_data.position_global[0], ball_data.position_global[1], ball_data.timestamp))
        
        return ball_data, roi_sub_frame

    def _preprocess_frame(self, frame):
        roi_sub_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY)
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        self.prev_frames_gray_roi.append(gray_roi_blurred)
        return roi_sub_frame, gray_roi_blurred

    def _detect_fmo(self):
        if len(self.prev_frames_gray_roi) < 3: return None
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        try:
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            _, thresh_mask = cv2.threshold(motion_mask, Config.THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        
        opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        return closed_mask

    def _find_best_ball_in_roi(self, motion_mask_roi, current_timestamp):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        potential_balls = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if Config.MIN_BALL_AREA_PX < area < Config.MAX_BALL_AREA_PX:
                w_roi, h_roi = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                circularity = 0; contour_to_store = None
                if max(w_roi, h_roi) > 0:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]; contour_to_store = cnt
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: circularity = 4 * math.pi * area / (perimeter * perimeter)
                potential_balls.append({'position_roi': (int(cx_roi), int(cy_roi)), 'area': area,
                                        'circularity': circularity, 'contour_roi': contour_to_store})
        if not potential_balls: return None

        best_ball_info = None
        if not self.trajectory:
            highly_circular = [b for b in potential_balls if b['circularity'] > Config.MIN_BALL_CIRCULARITY]
            best_ball_info = max(highly_circular, key=lambda b: b['circularity']) if highly_circular else max(potential_balls, key=lambda b: b['area'])
        else:
            last_x_global, last_y_global, _ = self.trajectory[-1]
            for ball_info in potential_balls:
                cx_global = ball_info['position_roi'][0] + self.roi_start_x
                cy_global = ball_info['position_roi'][1] + self.roi_top_y
                distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
                ball_info['distance_from_last'] = float('inf') if distance > self.frame_width * 0.4 else distance
                
                consistency_score = 0
                if len(self.trajectory) >= 2:
                    prev_x, prev_y, _ = self.trajectory[-2]
                    vec_hist = (last_x_global - prev_x, last_y_global - prev_y)
                    vec_curr = (cx_global - last_x_global, cy_global - last_y_global)
                    dot = vec_hist[0]*vec_curr[0] + vec_hist[1]*vec_curr[1]
                    mag_hist = math.sqrt(vec_hist[0]**2 + vec_hist[1]**2)
                    mag_curr = math.sqrt(vec_curr[0]**2 + vec_curr[1]**2)
                    if mag_hist > 0 and mag_curr > 0:
                        consistency_score = max(0, dot / (mag_hist * mag_curr))
                ball_info['consistency'] = consistency_score
                ball_info['score'] = (0.3 / (1.0 + ball_info['distance_from_last'])) + \
                                     (0.5 * ball_info['consistency']) + \
                                     (0.2 * ball_info['circularity'])
            best_ball_info = max(potential_balls, key=lambda b: b['score'])

        if not best_ball_info: return None
        
        cx_roi, cy_roi = best_ball_info['position_roi']
        return BallData(
            position_roi=(cx_roi, cy_roi),
            position_global=(cx_roi + self.roi_start_x, cy_roi + self.roi_top_y),
            contour_roi=best_ball_info.get('contour_roi'),
            timestamp=current_timestamp,
            area=best_ball_info['area'],
            circularity=best_ball_info['circularity']
        )

# ... (The rest of the classes: SessionManager, Visualizer, create_output_files remain the same)
class SessionManager:
    @dataclass
    class EventRecord:
        ball_x_global: float; timestamp: float; speed_kmh: float
        predicted: bool = False; processed: bool = False

    def __init__(self, frame_width, display_fps_provider, max_speeds, direction):
        self.state = SessionState()
        self.max_speeds_to_collect = max_speeds
        self.direction = direction
        self.frame_width = frame_width
        self.get_display_fps = display_fps_provider
        self.center_x_global = frame_width // 2
        self.center_zone_width = frame_width * Config.CENTER_ZONE_WIDTH_RATIO
        self.event_buffer = deque(maxlen=Config.EVENT_BUFFER_SIZE_CENTER_CROSS)
        self._reset_transient_state()

    def toggle_session(self):
        self.state.is_active = not self.state.is_active
        if self.state.is_active:
            self.state.session_id += 1
            self.state.output_generated = False
            self._reset_transient_state()
            print(f"計數啟動 (Session #{self.state.session_id}) - 目標: {self.max_speeds_to_collect} 次.")
        else:
            print(f"計數關閉 (Session #{self.state.session_id}).")
            if self.state.speeds and not self.state.output_generated:
                self.state.output_generated = True
                return True
        return False

    def _reset_transient_state(self):
        self.state.speeds.clear(); self.state.times.clear()
        self.state.last_recorded_speed_kmh = 0
        self.event_buffer.clear()
        self.last_committed_crossing_time = 0
        self.ball_on_left_of_center = False
        self.last_ball_x_global = None
        self.first_ball_crossing_timestamp = None

    def update(self, ball_data: BallData, speed_kmh: float, trajectory: deque, current_timestamp: float):
        if not self.state.is_active: return False
        
        if ball_data:
            self._record_potential_crossing(ball_data, speed_kmh, trajectory, current_timestamp)
            self.last_ball_x_global = ball_data.position_global[0]
        else: self.last_ball_x_global = None
        
        return self._process_crossing_events(current_timestamp)

    def _record_potential_crossing(self, ball_data, speed_kmh, trajectory, current_timestamp):
        ball_x = ball_data.position_global[0]
        if self.direction not in ['right_to_left', 'both']: return
        if current_timestamp - self.last_committed_crossing_time < Config.EFFECTIVE_CROSSING_COOLDOWN_S: return
        
        crossed = self.last_ball_x_global is not None and self.last_ball_x_global >= self.center_x_global and ball_x < self.center_x_global and not self.ball_on_left_of_center
        if crossed and speed_kmh > 0.1:
            self.event_buffer.append(self.EventRecord(ball_x, current_timestamp, speed_kmh))
        
        elif not crossed and not self.ball_on_left_of_center and len(trajectory) >= 2 and speed_kmh > 0.1:
            p1_x, _, p1_t = trajectory[-2]; p2_x, _, p2_t = trajectory[-1]
            if p1_x >= self.center_x_global and (p2_t - p1_t) > 0:
                vx = (p2_x - p1_x) / (p2_t - p1_t)
                fps = self.get_display_fps(); fps = fps if fps > 1 else 30
                if vx < -(self.frame_width * 0.005) / (1.0 / fps):
                    for lookahead in [1, 2, 3]:
                        pred_t = lookahead / fps; pred_x = ball_x + vx * pred_t
                        if pred_x < self.center_x_global:
                            pred_ts = current_timestamp + pred_t
                            if not any(e.predicted and abs(e.timestamp - pred_ts) < 1.0/fps for e in self.event_buffer):
                                self.event_buffer.append(self.EventRecord(pred_x, pred_ts, speed_kmh, predicted=True))
                                break
        
        if ball_x < self.center_x_global - self.center_zone_width: self.ball_on_left_of_center = True
        elif ball_x > self.center_x_global + self.center_zone_width: self.ball_on_left_of_center = False
            
    def _process_crossing_events(self, current_timestamp: float):
        if self.state.output_generated: return False

        events = sorted([e for e in self.event_buffer if not e.processed], key=lambda e: e.timestamp)
        committed_event = None
        for ev in events:
            if not ev.predicted and ev.timestamp - self.last_committed_crossing_time >= Config.EFFECTIVE_CROSSING_COOLDOWN_S:
                committed_event = ev; break
        if not committed_event:
            for ev in events:
                if ev.predicted and current_timestamp >= ev.timestamp and ev.timestamp - self.last_committed_crossing_time >= Config.EFFECTIVE_CROSSING_COOLDOWN_S:
                    committed_event = ev; break
        
        if committed_event and len(self.state.speeds) < self.max_speeds_to_collect:
            if not self.first_ball_crossing_timestamp: self.first_ball_crossing_timestamp = committed_event.timestamp
            rel_time = round(committed_event.timestamp - self.first_ball_crossing_timestamp, 2)
            self.state.speeds.append(committed_event.speed_kmh); self.state.times.append(rel_time)
            self.state.last_recorded_speed_kmh = committed_event.speed_kmh
            self.last_committed_crossing_time = committed_event.timestamp
            self.ball_on_left_of_center = True; committed_event.processed = True
            
            for ev in self.event_buffer:
                if not ev.processed and abs(ev.timestamp - committed_event.timestamp) < Config.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0: ev.processed = True

        self.event_buffer = deque([e for e in self.event_buffer if not e.processed and (current_timestamp - e.timestamp < 2.0)], maxlen=Config.EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        if len(self.state.speeds) >= self.max_speeds_to_collect and not self.state.output_generated:
            self.state.output_generated = True
            if Config.AUTO_STOP_AFTER_COLLECTION: self.state.is_active = False
            return True
        return False

class Visualizer:
    def __init__(self, frame_width, frame_height, roi_rect, max_speeds_to_collect):
        self.frame_width, self.frame_height = frame_width, frame_height
        self.roi_start_x, _, self.roi_end_x, self.roi_bottom_y = roi_rect
        self.max_speeds_to_collect = max_speeds_to_collect
        self.static_overlay = self._create_static_overlay()
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"
        
    def _create_static_overlay(self):
        overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(overlay, (self.roi_start_x, 0), (self.roi_start_x, self.roi_bottom_y), Config.ROI_COLOR_BGR, 2)
        cv2.line(overlay, (self.roi_end_x, 0), (self.roi_end_x, self.roi_bottom_y), Config.ROI_COLOR_BGR, 2)
        cv2.line(overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), Config.ROI_COLOR_BGR, 2)
        center_x = self.frame_width // 2
        cv2.line(overlay, (center_x, 0), (center_x, self.frame_height), Config.CENTER_LINE_COLOR_BGR, 2)
        return overlay

    def draw(self, frame: np.ndarray, f_info: FrameInfo, s_state: SessionState, roi_sub_frame: np.ndarray):
        vis = frame
        if f_info.frame_counter % Config.VISUALIZATION_DRAW_INTERVAL == 0:
            vis = cv2.addWeighted(vis, 1.0, self.static_overlay, 0.7, 0)
            if f_info.trajectory and len(f_info.trajectory) >= 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in f_info.trajectory], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=False, color=Config.TRAJECTORY_COLOR_BGR, thickness=2)
        
        if f_info.ball_data and roi_sub_frame is not None:
            cx_r, cy_r = f_info.ball_data.position_roi
            cv2.circle(roi_sub_frame, (cx_r, cy_r), 5, Config.BALL_COLOR_BGR, -1)
            if f_info.ball_data.contour_roi is not None:
                cv2.drawContours(roi_sub_frame, [f_info.ball_data.contour_roi], 0, Config.CONTOUR_COLOR_BGR, 2)
            cx_g, cy_g = f_info.ball_data.position_global
            cv2.circle(vis, (cx_g, cy_g), 8, Config.BALL_COLOR_BGR, -1)

        cv2.putText(vis, f"Speed: {f_info.current_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, (0,0,255), Config.FONT_THICKNESS_VIS)
        cv2.putText(vis, f"FPS: {f_info.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, (0,255,0), Config.FONT_THICKNESS_VIS)
        
        status, color = ("ON", (0,255,0)) if s_state.is_active else ("OFF", (0,0,255))
        cv2.putText(vis, f"Counting: {status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, color, Config.FONT_THICKNESS_VIS)
        if s_state.last_recorded_speed_kmh > 0: cv2.putText(vis, f"Last Net: {s_state.last_recorded_speed_kmh:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, (255,0,0), Config.FONT_THICKNESS_VIS)
        cv2.putText(vis, f"Recorded: {len(s_state.speeds)}/{self.max_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, (255,0,0), Config.FONT_THICKNESS_VIS)
        if s_state.times: cv2.putText(vis, f"Last Time: {s_state.times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_VIS, (255,0,0), Config.FONT_THICKNESS_VIS)
        cv2.putText(vis, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        return vis

def create_output_files(session_state: SessionState, video_file_path: str, use_video_file: bool):
    if not session_state.speeds: return
    
    out_dir, base_name = "", ""
    if use_video_file and video_file_path:
        try:
            out_dir = os.path.dirname(video_file_path)
            stem = os.path.splitext(os.path.basename(video_file_path))[0]
            parts = stem.split('_'); base_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem
        except Exception: use_video_file = False

    if not use_video_file or not out_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(Config.OUTPUT_DATA_FOLDER, ts)
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"speed_data_{ts}"

    print(f"正在將檔案儲存至 '{out_dir}'，前綴為 '{base_name}'")
    avg, M, m = np.mean(session_state.speeds), max(session_state.speeds), min(session_state.speeds)

    plt.figure(figsize=(12, 7))
    plt.plot(session_state.times, session_state.speeds, 'o-', label='Speed (km/h)')
    plt.axhline(y=avg, color='r', linestyle='--', label=f'Avg: {avg:.1f} km/h')
    plt.title(f'Net Crossing Speeds - Session {session_state.session_id}'); plt.xlabel('Relative Time (s)'); plt.ylabel('Speed (km/h)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{base_name}_chart.png'), dpi=150); plt.close()

    with open(os.path.join(out_dir, f'{base_name}_data.txt'), 'w') as f:
        f.write(f"Net Speeds - Session {session_state.session_id}\nAvg: {avg:.1f}, Max: {M:.1f}, Min: {m:.1f} km/h\n------------------\n")
        for t, s in zip(session_state.times, session_state.speeds): f.write(f"{t:.2f}s: {s:.1f} km/h\n")

    with open(os.path.join(out_dir, f'{base_name}_data.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Session ID', 'Relative Time (s)', 'Speed (km/h)'])
        for t, s in zip(session_state.times, session_state.speeds): w.writerow([session_state.session_id, f"{t:.2f}", f"{s:.1f}"])
    print(f"Session {session_state.session_id} 的輸出檔案已成功儲存。")


# ==============================================================================
# 3. Main Application Class (主應用程式類別)
# ==============================================================================

class PingPongSpeedTracker:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.debug_mode = args.debug
        self.reader = FrameReader(args.video if args.video else args.camera_idx, 
                                  args.fps, bool(args.video), args.width, args.height)
        
        self.frame_width, self.frame_height = self.reader.frame_width, self.reader.frame_height
        self.trajectory = deque(maxlen=Config.MAX_TRAJECTORY_POINTS)
        self.detector = Detector(self.frame_width, self.frame_height, self.trajectory)
        self.session_manager = SessionManager(self.frame_width, lambda: self.display_fps, 
                                              args.count, args.direction)
        roi_rect = (self.detector.roi_start_x, 0, self.detector.roi_end_x, self.detector.roi_bottom_y)
        self.visualizer = Visualizer(self.frame_width, self.frame_height, roi_rect, args.count)
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.frame_counter = 0; self.display_fps = self.reader.actual_fps
        self.frame_timestamps_for_fps = deque(maxlen=Config.MAX_FRAME_TIMES_FPS_CALC)
        self.current_speed_kmh = 0
        self._create_perspective_lookup_table()
        
    def _create_perspective_lookup_table(self):
        self.perspective_lookup = {}
        roi_h = self.detector.roi_bottom_y - self.detector.roi_top_y
        for y_roi_round in range(0, roi_h + 1, 10):
            y_g = y_roi_round + self.detector.roi_top_y
            rel_y = np.clip(y_g / self.detector.roi_bottom_y, 0, 1) if self.detector.roi_bottom_y > 0 else 0.5
            w_cm = self.args.far_width * (1 - rel_y) + self.args.near_width * rel_y
            w_px = self.detector.roi_end_x - self.detector.roi_start_x
            self.perspective_lookup[y_roi_round] = w_cm / w_px if w_cm > 0 and w_px > 0 else self.args.table_len / self.frame_width

    def run(self):
        print("=== 乒乓球速度追蹤系統 v12.2 (最終修正版) ===")
        self.running = True; self.reader.start()
        cv2.namedWindow('Ping Pong Tracker', cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    self._handle_shutdown(); break

                f_info, roi_sub = self.process_single_frame(frame)
                display_frame = self.visualizer.draw(frame, f_info, self.session_manager.state, roi_sub)
                cv2.imshow('Ping Pong Tracker', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: self.running = False
                elif key == ord(' '):
                    if self.session_manager.toggle_session(): self._generate_outputs_async()
                elif key == ord('d'): self.debug_mode = not self.debug_mode
        except KeyboardInterrupt: print("使用者中斷程式 (Ctrl+C)。")
        finally: self._handle_shutdown(); self.stop()

    def process_single_frame(self, frame):
        self.frame_counter += 1; self._update_display_fps()
        ts = self.frame_counter / self.reader.actual_fps if self.args.video else time.monotonic()
        
        ball, roi_sub = self.detector.detect(frame, ts)
        if ball: self._calculate_ball_speed()
        
        if time.monotonic() - self.detector.last_detection_timestamp > self.args.timeout:
            self.trajectory.clear(); self.current_speed_kmh = 0
            
        if self.session_manager.update(ball, self.current_speed_kmh, self.trajectory, ts):
            self._generate_outputs_async()

        return FrameInfo(self.frame_counter, self.display_fps, self.current_speed_kmh, self.trajectory, ball, self.debug_mode), roi_sub

    def _update_display_fps(self):
        if self.args.video: return
        now = time.monotonic()
        self.frame_timestamps_for_fps.append(now)
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed > 0:
                fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed
                self.display_fps = (1 - Config.FPS_SMOOTHING_FACTOR) * self.display_fps + Config.FPS_SMOOTHING_FACTOR * fps

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2: self.current_speed_kmh = 0; return
        p1, p2 = self.trajectory[-2], self.trajectory[-1]
        x1, y1, t1 = p1; x2, y2, t2 = p2
        delta_t = t2 - t1
        if delta_t > 0.0001:
            y1_r = max(0, y1 - self.detector.roi_top_y); y2_r = max(0, y2 - self.detector.roi_top_y)
            fb_ratio = list(self.perspective_lookup.values())[0] if self.perspective_lookup else 1
            r1 = self.perspective_lookup.get(round(y1_r / 10) * 10, fb_ratio)
            r2 = self.perspective_lookup.get(round(y2_r / 10) * 10, fb_ratio)
            dist_cm = math.hypot(x2 - x1, y2 - y1) * ((r1 + r2) / 2.0)
            speed_kmh = (dist_cm / delta_t) * Config.KMH_CONVERSION_FACTOR
            self.current_speed_kmh = (1 - Config.SPEED_SMOOTHING_FACTOR) * self.current_speed_kmh + Config.SPEED_SMOOTHING_FACTOR * speed_kmh
        else: self.current_speed_kmh *= (1 - Config.SPEED_SMOOTHING_FACTOR)

    def _handle_shutdown(self):
        if self.session_manager.state.is_active and self.session_manager.state.speeds and not self.session_manager.state.output_generated:
            self._generate_outputs_async(); self.session_manager.state.output_generated = True

    def _generate_outputs_async(self):
        state_copy = SessionState(**self.session_manager.state.__dict__)
        self.file_writer_executor.submit(create_output_files, state_copy, self.args.video, bool(self.args.video))

    def stop(self):
        print("正在關閉系統...")
        self.running = False
        self.reader.stop()
        self.file_writer_executor.shutdown(wait=True)
        cv2.destroyAllWindows()
        print("系統已完全關閉。")

def main():
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤系統 v12.2 (最終修正版)')
    parser.add_argument('--video', type=str, help='影片檔案路徑。若為 None，則使用攝影機。')
    parser.add_argument('--camera_idx', type=int, default=Config.DEFAULT_CAMERA_INDEX, help='攝影機索引。')
    parser.add_argument('--fps', type=int, default=Config.DEFAULT_TARGET_FPS, help='攝影機目標 FPS。')
    parser.add_argument('--width', type=int, default=Config.DEFAULT_FRAME_WIDTH, help='影像寬度。')
    parser.add_argument('--height', type=int, default=Config.DEFAULT_FRAME_HEIGHT, help='影像高度。')
    parser.add_argument('--table_len', type=float, default=Config.DEFAULT_TABLE_LENGTH_CM, help='球桌長度 (cm)。')
    parser.add_argument('--timeout', type=float, default=Config.DEFAULT_DETECTION_TIMEOUT_S, help='球體偵測超時 (s)。')
    parser.add_argument('--direction', type=str, default=Config.NET_CROSSING_DIRECTION_DEFAULT, choices=['left_to_right', 'right_to_left', 'both'], help='要記錄的過網方向。')
    parser.add_argument('--count', type=int, default=Config.MAX_NET_SPEEDS_TO_COLLECT, help='每個會話要收集的速度次數。')
    parser.add_argument('--near_width', type=float, default=Config.NEAR_SIDE_WIDTH_CM_DEFAULT, help='ROI 近端實際寬度 (cm)。')
    parser.add_argument('--far_width', type=float, default=Config.FAR_SIDE_WIDTH_CM_DEFAULT, help='ROI 遠端實際寬度 (cm)。')
    parser.add_argument('--debug', action='store_true', help='啟用除錯模式。')
    args = parser.parse_args()

    try:
        tracker = PingPongSpeedTracker(args)
        tracker.run()
    except IOError as e:
        print(f"初始化錯誤: {e}")
        print("程式無法啟動。")

if __name__ == '__main__':
    main()