#!/usr/bin/env python3
# 乒乓球速度追蹤系統 macOS 優化版 (使用 AVFoundation 後端)
# Optimized for macOS with AVFoundation backend for 120fps performance

import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import argparse
import matplotlib
matplotlib.use("Agg") # Must be before importing pyplot
import matplotlib.pyplot as plt
import os
import csv
import threading
import queue
import concurrent.futures

# —— Global Parameter Configuration ——
# Basic Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 60
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_TABLE_LENGTH_CM = 94

# Detection Parameters
DEFAULT_DETECTION_TIMEOUT = 0.3
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.55
MAX_TRAJECTORY_POINTS = 120

# Center Line Detection
CENTER_LINE_WIDTH_PIXELS = 1000
CENTER_DETECTION_COOLDOWN_S = 0.001
MAX_NET_SPEEDS_TO_COLLECT = 100
NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left' # 'left_to_right', 'right_to_left', 'both'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output'

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
# Speed Calculation
SPEED_SMOOTHING_FACTOR = 0.3
KMH_CONVERSION_FACTOR = 0.036

# FPS Calculation
FPS_SMOOTHING_FACTOR = 0.4
MAX_FRAME_TIMES_FPS_CALC = 20

# Visualization Parameters
TRAJECTORY_COLOR_BGR = (0, 0, 255)
BALL_COLOR_BGR = (0, 255, 255)
CONTOUR_COLOR_BGR = (255, 0, 0)
ROI_COLOR_BGR = (0, 255, 0)
SPEED_TEXT_COLOR_BGR = (0, 0, 255)
FPS_TEXT_COLOR_BGR = (0, 255, 0)
CENTER_LINE_COLOR_BGR = (0, 255, 255)
NET_SPEED_TEXT_COLOR_BGR = (255, 0, 0)
FONT_SCALE_VIS = 1
FONT_THICKNESS_VIS = 2
VISUALIZATION_DRAW_INTERVAL = 2 # Draw full visuals every N frames

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 30 # For FrameReader
EVENT_BUFFER_SIZE_CENTER_CROSS = 200
PREDICTION_LOOKAHEAD_FRAMES = 10

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 8)  # macOS 上通常有更好的多核心性能
except AttributeError:
    cv2.setNumThreads(8)

# 確保輸出目錄存在
os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

class FrameData:
    """Data structure for passing frame-related information."""
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0):
        self.frame = frame
        self.roi_sub_frame = roi_sub_frame  # The ROI portion of the frame
        self.ball_position_in_roi = ball_position_in_roi  # (x,y) relative to ROI
        self.ball_contour_in_roi = ball_contour_in_roi  # Contour points relative to ROI
        self.current_ball_speed_kmh = current_ball_speed_kmh
        self.display_fps = display_fps
        self.is_counting_active = is_counting_active
        self.collected_net_speeds = collected_net_speeds if collected_net_speeds is not None else []
        self.last_recorded_net_speed_kmh = last_recorded_net_speed_kmh
        self.collected_relative_times = collected_relative_times if collected_relative_times is not None else []
        self.debug_display_text = debug_display_text
        self.frame_counter = frame_counter
        self.trajectory_points_global = []  # (x,y) in global frame coordinates

class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False

class FrameReader:
    """Reads frames from camera or video file in a separate thread, optimized for macOS."""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        
        # 使用 AVFoundation 後端 (macOS 專用)
        if not self.use_video_file:
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
            print("使用 AVFoundation 後端以獲得最佳性能")
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            
        self._configure_capture(frame_width, frame_height)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             self.actual_fps = self.target_fps  # Use target if webcam FPS is unreliable

    def _configure_capture(self, frame_width, frame_height):
        if not self.use_video_file:
            # 設定 MJPG 格式以提高幀率
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # 設定解析度和幀率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 顯示設定信息
            actual_codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = ''.join([chr((actual_codec >> 8 * i) & 0xFF) for i in range(4)])
            print(f"攝影機編解碼器設為: {codec_str}")
            print(f"目標FPS: {self.target_fps}, 攝影機報告: {self.cap.get(cv2.CAP_PROP_FPS)}")
            
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")

    def _read_frames(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False  # End of video or camera error
                    self.frame_queue.put((False, None))  # Signal end
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(1.0 / (self.target_fps * 4))  # 避免在佇列已滿時過度使用 CPU

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)  # Wait up to 1s for a frame
        except queue.Empty:
            return False, None  # Timeout

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Wait for thread to finish
        if self.cap.isOpened():
            self.cap.release()

    def get_properties(self):
        return self.actual_fps, self.frame_width, self.frame_height

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM,
                 detection_timeout_s=DEFAULT_DETECTION_TIMEOUT, use_video_file=False,
                 target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=DEBUG_MODE_DEFAULT,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
                 max_net_speeds=MAX_NET_SPEEDS_TO_COLLECT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT):
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.target_fps = target_fps

        self.reader = FrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps  # Initial display FPS

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm  # Nominal, used if perspective fails

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0  # ROI starts from top of the frame
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y

        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)
        self.last_motion_mask = None  # 儲存上一幀的運動遮罩以節省計算

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2
        self.center_line_start_x = self.center_x_global - CENTER_LINE_WIDTH_PIXELS // 2
        self.center_line_end_x = self.center_x_global + CENTER_LINE_WIDTH_PIXELS // 2
        
        self.net_crossing_direction = net_crossing_direction
        self.max_net_speeds_to_collect = max_net_speeds
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_net_crossing_detection_time = 0
        self.last_recorded_net_speed_kmh = 0
        self.last_ball_x_global = None
        self.output_generated_for_session = False
        
        self.is_counting_active = False
        self.count_session_id = 0
        self.timing_started_for_session = False
        self.first_ball_crossing_timestamp = None
        
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 性能監控
        self.performance_history = deque(maxlen=10)

        self._precalculate_overlay()
        self._create_perspective_lookup_table()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10):  # step by 10px
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y) 
        
        if self.roi_bottom_y == 0:
            relative_y = 0.5
        else:
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)
        
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        
        roi_width_px = self.roi_end_x - self.roi_start_x
        if current_width_cm > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px  # cm per pixel
        else:
            pixel_to_cm_ratio = self.table_length_cm / self.frame_width  # Fallback

        return pixel_to_cm_ratio

    def _update_display_fps(self):
        if self.use_video_file:  # For video files, FPS is fixed
            self.display_fps = self.actual_fps
            return

        now = time.monotonic()  # More suitable for interval timing
        self.frame_timestamps_for_fps.append(now)
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 0:
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                self.display_fps = (1 - FPS_SMOOTHING_FACTOR) * self.display_fps + FPS_SMOOTHING_FACTOR * measured_fps
        self.last_frame_timestamp_for_fps = now

    def _preprocess_frame(self, frame):
        roi_sub_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY)
        
        # 使用較小的高斯模糊核心以提高性能
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        self.prev_frames_gray_roi.append(gray_roi_blurred)
        return roi_sub_frame, gray_roi_blurred

    def _detect_fmo(self):
        # 每2幀處理一次以提高性能
        if self.frame_counter % 2 != 0 and self.last_motion_mask is not None:
            return self.last_motion_mask
            
        if len(self.prev_frames_gray_roi) < 3:
            return None
        
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        
        # 降低處理解析度以提高速度
        scale_factor = 0.7  # 降低到70%解析度
        if scale_factor < 1.0:
            h, w = f1.shape
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            f1_small = cv2.resize(f1, (new_w, new_h))
            f2_small = cv2.resize(f2, (new_w, new_h))
            f3_small = cv2.resize(f3, (new_w, new_h))
            
            diff1 = cv2.absdiff(f1_small, f2_small)
            diff2 = cv2.absdiff(f2_small, f3_small)
            motion_mask_small = cv2.bitwise_and(diff1, diff2)
            
            # 處理完再放大回原始尺寸
            motion_mask = cv2.resize(motion_mask_small, (w, h))
        else:
            diff1 = cv2.absdiff(f1, f2)
            diff2 = cv2.absdiff(f2, f3)
            motion_mask = cv2.bitwise_and(diff1, diff2)
        
        try:
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:  # OTSU can fail on blank images
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        
        if OPENING_KERNEL_SIZE_FMO[0] > 0:  # Avoid error if kernel size is (0,0)
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else:
            opened_mask = thresh_mask
        
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        
        # 保存結果供下一幀使用
        self.last_motion_mask = closed_mask
        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        
        potential_balls = []
        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
                x_roi = stats[i, cv2.CC_STAT_LEFT]
                y_roi = stats[i, cv2.CC_STAT_TOP]
                w_roi = stats[i, cv2.CC_STAT_WIDTH]
                h_roi = stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                
                circularity = 0
                if max(w_roi, h_roi) > 0:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                potential_balls.append({
                    'position_roi': (int(cx_roi), int(cy_roi)),
                    'area': area,
                    'circularity': circularity,
                    'label_id': i,
                    'contour_roi': contours[0] if contours else None  # Store contour
                })

        if not potential_balls: return None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        # Convert ROI coordinates to global frame coordinates
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y  # y is from top of ROI
        
        current_timestamp = time.monotonic()
        if self.use_video_file:  # For video files, use frame count for timing
            current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()  # System time for timeout
        
        if self.is_counting_active:
            self.check_center_crossing(cx_global, current_timestamp)
        
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')

    def _select_best_ball_candidate(self, candidates):
        if not candidates: return None

        if not self.trajectory:  # No history, pick most circular one if good enough
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular:
                return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])  # Fallback to largest

        last_x_global, last_y_global, _ = self.trajectory[-1]

        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y

            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            
            # Penalize balls too far from last known position
            if distance > self.frame_width * 0.2:  # Heuristic: 20% of frame width
                ball_info['distance_from_last'] = float('inf')

            consistency_score = 0
            if len(self.trajectory) >= 2:  # Need at least two previous points for direction
                prev_x_global, prev_y_global, _ = self.trajectory[-2]
                # Vector from trajectory[-2] to trajectory[-1]
                vec_hist_dx = last_x_global - prev_x_global
                vec_hist_dy = last_y_global - prev_y_global
                # Vector from trajectory[-1] to current candidate
                vec_curr_dx = cx_global - last_x_global
                vec_curr_dy = cy_global - last_y_global

                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist = math.sqrt(vec_hist_dx**2 + vec_hist_dy**2)
                mag_curr = math.sqrt(vec_curr_dx**2 + vec_curr_dy**2)

                if mag_hist > 0 and mag_curr > 0:
                    # Cosine similarity: 1 for same direction, -1 for opposite, 0 for orthogonal
                    cosine_similarity = dot_product / (mag_hist * mag_curr)
                    consistency_score = max(0, cosine_similarity)  # We only care about forward consistency
            ball_info['consistency'] = consistency_score
        
        # Scoring: Higher is better
        # Weights: distance (lower is better), consistency (higher), circularity (higher)
        for ball_info in candidates:
            score = (0.4 / (1.0 + ball_info['distance_from_last'])) + \
                    (0.4 * ball_info['consistency']) + \
                    (0.2 * ball_info['circularity'])
            ball_info['score'] = score
        
        return max(candidates, key=lambda b: b['score'])

    def toggle_counting(self):
        self.is_counting_active = not self.is_counting_active
        if self.is_counting_active:
            self.count_session_id += 1
            self.collected_net_speeds = []
            self.collected_relative_times = []
            self.timing_started_for_session = False
            self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear()
            self.output_generated_for_session = False
            print(f"Counting ON (Session #{self.count_session_id}) - Target: {self.max_net_speeds_to_collect} speeds.")
        else:
            print(f"Counting OFF (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_outputs_async()  # Use async version
            self.output_generated_for_session = True  # Prevent re-generating if toggled off/on quickly

    def check_center_crossing(self, ball_x_global, current_timestamp):
        if self.last_ball_x_global is None:
            self.last_ball_x_global = ball_x_global
            return

        time_since_last_net_cross = current_timestamp - self.last_net_crossing_detection_time
        if time_since_last_net_cross < CENTER_DETECTION_COOLDOWN_S:
            self.last_ball_x_global = ball_x_global
            return

        self._record_potential_crossing(ball_x_global, current_timestamp)
        self.last_ball_x_global = ball_x_global

    def _record_potential_crossing(self, ball_x_global, current_timestamp):
        # 更寬鬆的判定邏輯（解決遺漏紀錄）
        crossed_l_to_r = (self.last_ball_x_global < self.center_x_global and ball_x_global >= self.center_x_global)
        crossed_r_to_l = (self.last_ball_x_global > self.center_x_global and ball_x_global <= self.center_x_global)
        
        actual_crossing_detected = False
        if self.net_crossing_direction == 'left_to_right' and crossed_l_to_r: actual_crossing_detected = True
        elif self.net_crossing_direction == 'right_to_left' and crossed_r_to_l: actual_crossing_detected = True
        elif self.net_crossing_direction == 'both' and (crossed_l_to_r or crossed_r_to_l): actual_crossing_detected = True

        if actual_crossing_detected and self.current_ball_speed_kmh > 0:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            return

        # Predictive crossing (simplified)
        if len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 0:
            pt1_x, _, pt1_t = self.trajectory[-2]
            pt2_x, _, pt2_t = self.trajectory[-1]
            delta_t = pt2_t - pt1_t
            if delta_t > 0:
                vx_pixels_per_time_unit = (pt2_x - pt1_x) / delta_t
                
                prediction_horizon_time = PREDICTION_LOOKAHEAD_FRAMES / self.display_fps if self.display_fps > 0 else 0.1
                
                predicted_x_future = ball_x_global + vx_pixels_per_time_unit * prediction_horizon_time
                predicted_timestamp_future = current_timestamp + prediction_horizon_time

                predict_l_to_r = (ball_x_global < self.center_x_global and predicted_x_future >= self.center_x_global)
                predict_r_to_l = (ball_x_global > self.center_x_global and predicted_x_future <= self.center_x_global)
                
                prediction_valid_for_direction = False
                if self.net_crossing_direction == 'left_to_right' and predict_l_to_r: prediction_valid_for_direction = True
                elif self.net_crossing_direction == 'right_to_left' and predict_r_to_l: prediction_valid_for_direction = True
                elif self.net_crossing_direction == 'both' and (predict_l_to_r or predict_r_to_l): prediction_valid_for_direction = True

                if prediction_valid_for_direction:
                    # Avoid duplicate predictions too close in time
                    can_add_prediction = True
                    for ev in self.event_buffer_center_cross:
                        if ev.predicted and abs(ev.timestamp - predicted_timestamp_future) < 0.1:  # 100ms
                            can_add_prediction = False
                            break
                    if can_add_prediction:
                        event = EventRecord(predicted_x_future, predicted_timestamp_future, self.current_ball_speed_kmh, predicted=True)
                        self.event_buffer_center_cross.append(event)

    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_eval_time = time.monotonic()
        if self.use_video_file: current_eval_time = self.frame_counter / self.actual_fps
        
        events_to_commit = []
        
        processed_indices = []
        # First, find actual events and mark nearby predictions as handled
        for i, event in enumerate(self.event_buffer_center_cross):
            if event.processed: continue
            if not event.predicted:  # Actual event
                events_to_commit.append(event)
                event.processed = True  # Mark for removal from buffer or just ignore later
                # Invalidate nearby predictions
                for j, other_event in enumerate(self.event_buffer_center_cross):
                    if i !=j and other_event.predicted and not other_event.processed and \
                       abs(event.timestamp - other_event.timestamp) < 0.2:  # 200ms window
                        other_event.processed = True  # This prediction is covered by an actual event

        # Then, process timed-out predictions not covered by actual events
        for event in self.event_buffer_center_cross:
            if event.processed: continue
            if event.predicted and (current_eval_time - event.timestamp) > 0.05:  # Prediction considered "due" or "past due"
                events_to_commit.append(event)
                event.processed = True

        # Sort events by timestamp before committing
        events_to_commit.sort(key=lambda e: e.timestamp)

        for event in events_to_commit:
            if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                break  # Stop if limit reached

            if not self.timing_started_for_session:
                self.timing_started_for_session = True
                self.first_ball_crossing_timestamp = event.timestamp
                relative_time = 0.0
            else:
                relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2)
            
            self.last_recorded_net_speed_kmh = event.speed_kmh
            self.collected_net_speeds.append(event.speed_kmh)
            self.collected_relative_times.append(relative_time)
            self.last_net_crossing_detection_time = event.timestamp  # System time based on monotonic() or frame time
            
            status_msg = "Pred" if event.predicted else "Actual"
            print(f"Net Speed #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg})")

        # Clean up processed events from buffer (optional, as maxlen handles size)
        self.event_buffer_center_cross = deque(
            [e for e in self.event_buffer_center_cross if not e.processed],
            maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS
        )

        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Target {self.max_net_speeds_to_collect} speeds collected. Generating output.")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION:
                self.is_counting_active = False  # Optionally stop counting

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return

        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob

        # Distances are calculated in global frame using y_global for perspective.
        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        
        delta_t = t2 - t1
        if delta_t > 0:
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR 
            
            if self.current_ball_speed_kmh > 0:  # Apply smoothing if there's a previous speed
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else:
                self.current_ball_speed_kmh = speed_kmh
            
            if self.debug_mode:
                print(f"Speed: {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else:  # Should not happen if timestamps are monotonic
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)  # Decay speed if no movement or bad dt

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        # y_g are global y coordinates. Convert to y_in_roi for lookup.
        y1_roi = y1_g - self.roi_top_y
        y2_roi = y2_g - self.roi_top_y

        y1_roi_rounded = round(y1_roi / 10) * 10
        y2_roi_rounded = round(y2_roi / 10) * 10
        
        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * avg_px_to_cm_ratio
        return real_distance_cm

    def _generate_outputs_async(self):
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
        
        # Create copies for the thread to work on, preventing race conditions
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id

        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        """This method runs in a separate thread via ThreadPoolExecutor."""
        if not net_speeds: return

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
        os.makedirs(output_dir_path, exist_ok=True)

        avg_speed = sum(net_speeds) / len(net_speeds)
        max_speed = max(net_speeds)
        min_speed = min(net_speeds)

        # Generate Chart
        chart_filename = f'{output_dir_path}/speed_chart_{timestamp_str}.png'
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        for t, s in zip(relative_times, net_speeds):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Net Crossing Speeds - {timestamp_str}', fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(min_speed - y_range*0.1, max_speed + y_range*0.1)
        plt.figtext(0.02, 0.02, f"Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for title and figtext
        plt.savefig(chart_filename, dpi=150)
        plt.close()  # Important to close figures when done in non-interactive mode

        # Generate TXT
        txt_filename = f'{output_dir_path}/speed_data_{timestamp_str}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Net Speeds - Session {session_id} - {timestamp_str}\n")
            f.write("---------------------------------------\n")
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write("---------------------------------------\n")
            f.write(f"Total Points: {len(net_speeds)}\n")
            f.write(f"Average Speed: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
            f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")

        # Generate CSV
        csv_filename = f'{output_dir_path}/speed_data_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                writer.writerow([timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Total Points', len(net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        
        print(f"Output files saved to {output_dir_path}")

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        vis_frame = display_frame
        
        # 降低視覺化頻率以提高性能
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0

        if is_full_draw:
            # Add pre-calculated static overlay (ROI lines, center line)
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            # Draw trajectory
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                # 只使用一部分軌跡點以提高繪製效率
                if len(frame_data_obj.trajectory_points_global) > 15:
                    trajectory_points = frame_data_obj.trajectory_points_global[::2]  # 每隔一個點取樣
                else:
                    trajectory_points = frame_data_obj.trajectory_points_global
                
                pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)

        # Draw ball on ROI (if ball detected)
        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            # Only draw detailed contour in full visualization frames
            if is_full_draw and frame_data_obj.ball_contour_in_roi is not None:
                cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            
            # Always draw the ball marker on main frame
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1)

        # Text overlays (always drawn for fresh info)
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        if frame_data_obj.last_recorded_net_speed_kmh > 0:
            cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        if frame_data_obj.collected_relative_times:
            cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)

        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if self.debug_mode and frame_data_obj.debug_display_text and is_full_draw:
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        start_time = time.monotonic()  # 開始性能監控
        
        self.frame_counter += 1
        self._update_display_fps()
         
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame) 
        
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi)
            self._calculate_ball_speed() 
        
        self._check_timeout_and_reset()
        
        if self.is_counting_active:
            self._process_crossing_events()

        frame_data = FrameData(
            frame=frame,
            roi_sub_frame=roi_sub_frame,
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi,
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=f"Traj: {len(self.trajectory)}, Events: {len(self.event_buffer_center_cross)}" if self.debug_mode else None,
            frame_counter=self.frame_counter
        )
        if self.trajectory:  # Global trajectory points
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        # 性能監控
        frame_processing_time = time.monotonic() - start_time
        self.performance_history.append(frame_processing_time)
        
        if self.debug_mode and self.frame_counter % 60 == 0:
            avg_time = sum(self.performance_history) / len(self.performance_history)
            print(f"平均每幀處理時間: {avg_time*1000:.2f}ms (目標: {1000/self.target_fps:.2f}ms)")
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker (macOS Optimized) ===")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction}")
        print(f"Target speeds to collect: {self.max_net_speeds_to_collect}")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker (macOS Optimized)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # Or WINDOW_NORMAL for resizable

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("Video ended or frame read error.")
                    else: print("Camera error or stream ended.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("End of stream with pending data. Generating output.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    self.running = False
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("Quitting with pending data. Generating output.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                elif key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        except KeyboardInterrupt:
            print("Process interrupted by user (Ctrl+C).")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                print("Interrupted with pending data. Generating output.")
                self._generate_outputs_async()
                self.output_generated_for_session = True
        finally:
            self.running = False
            print("Shutting down...")
            self.reader.stop()
            print("Frame reader stopped.")
            self.file_writer_executor.shutdown(wait=True)
            print("File writer stopped.")
            cv2.destroyAllWindows()
            print("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker (macOS Optimized)')
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='Frame height.')
    parser.add_argument('--table_len', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='Table length (cm) for nominal px/cm.')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='Ball detection timeout (s).')
    
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                        choices=['left_to_right', 'right_to_left', 'both'], help='Net crossing direction to record.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='Number of net speeds to collect per session.')
    
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at near side.')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at far side.')
    
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='Enable debug printouts.')

    args = parser.parse_args()

    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False

    tracker = PingPongSpeedTracker(
        video_source=video_source_arg,
        table_length_cm=args.table_len,
        detection_timeout_s=args.timeout,
        use_video_file=use_video_file_arg,
        target_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        debug_mode=args.debug,
        net_crossing_direction=args.direction,
        max_net_speeds=args.count,
        near_width_cm=args.near_width,
        far_width_cm=args.far_width
    )
    tracker.run()

if __name__ == '__main__':
    main()