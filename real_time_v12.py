#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v12 (優化 Webcam 實時追蹤)
# 針對網路攝像頭下高速球體遺漏紀錄問題進行改進

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
DEFAULT_TARGET_FPS = 120
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_TABLE_LENGTH_CM = 142

# Detection Parameters
DEFAULT_DETECTION_TIMEOUT = 0.2
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.8
MAX_TRAJECTORY_POINTS = 120

# Center Line Detection
CENTER_LINE_WIDTH_PIXELS = 55
CENTER_DETECTION_COOLDOWN_S = 0.01          # 原始冷卻時間
WEBCAM_CENTER_DETECTION_COOLDOWN_S = 0.005  # Webcam 模式下更短的冷卻時間
MAX_NET_SPEEDS_TO_COLLECT = 27
NET_CROSSING_DIRECTION_DEFAULT = 'left_to_right' # 'left_to_right', 'right_to_left', 'both'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output'

# 增加 Webcam 專用參數
WEBCAM_PREDICTION_LEVELS = 6                # Webcam 模式下的預測級別 (原來只有1個)
WEBCAM_CENTER_TOLERANCE_PX = 12             # Webcam 模式下中心線檢測的容差像素
WEBCAM_BACKUP_DETECTION_ENABLED = True      # 是否啟用備用檢測機制
WEBCAM_CROSS_CONFIDENCE_THRESHOLD = 0.65    # 交叉檢測的信心閾值

# Perspective Correction
NEAR_SIDE_WIDTH_CM_DEFAULT = 29
FAR_SIDE_WIDTH_CM_DEFAULT = 72

# FMO (Fast Moving Object) Parameters
MAX_PREV_FRAMES_FMO = 10
OPENING_KERNEL_SIZE_FMO = (10, 10)
CLOSING_KERNEL_SIZE_FMO = (25, 25)
THRESHOLD_VALUE_FMO = 8

# Ball Detection Parameters
MIN_BALL_AREA_PX = 5
MAX_BALL_AREA_PX = 10000
MIN_BALL_CIRCULARITY = 0.4
# Speed Calculation
SPEED_SMOOTHING_FACTOR = 0.3
WEBCAM_SPEED_SMOOTHING_FACTOR = 0.15       # Webcam 模式下更低的平滑因子，提高穩定性
KMH_CONVERSION_FACTOR = 0.036

# FPS Calculation
FPS_SMOOTHING_FACTOR = 0.4
WEBCAM_FPS_SMOOTHING_FACTOR = 0.25         # Webcam 模式下更低的 FPS 平滑因子
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
FRAME_QUEUE_SIZE = 10 # For FrameReader
EVENT_BUFFER_SIZE_CENTER_CROSS = 70
WEBCAM_EVENT_BUFFER_SIZE = 150              # 增加 Webcam 模式下的事件緩衝區大小
PREDICTION_LOOKAHEAD_FRAMES = 15
WEBCAM_PREDICTION_LOOKAHEAD_FRAMES = 20     # Webcam 模式下更遠的預測幀數

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 10) # Fallback if os.cpu_count() is None/0
except AttributeError: # os.cpu_count() might not be available on all os modules
    cv2.setNumThreads(10) # Default to 10 threads if os.cpu_count() fails

class FrameData:
    """Data structure for passing frame-related information."""
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0):
        self.frame = frame
        self.roi_sub_frame = roi_sub_frame # The ROI portion of the frame
        self.ball_position_in_roi = ball_position_in_roi # (x,y) relative to ROI
        self.ball_contour_in_roi = ball_contour_in_roi # Contour points relative to ROI
        self.current_ball_speed_kmh = current_ball_speed_kmh
        self.display_fps = display_fps
        self.is_counting_active = is_counting_active
        self.collected_net_speeds = collected_net_speeds if collected_net_speeds is not None else []
        self.last_recorded_net_speed_kmh = last_recorded_net_speed_kmh
        self.collected_relative_times = collected_relative_times if collected_relative_times is not None else []
        self.debug_display_text = debug_display_text
        self.frame_counter = frame_counter
        self.trajectory_points_global = [] # (x,y) in global frame coordinates

class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False, confidence=1.0):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False
        self.confidence = confidence  # 新增: 事件信心度 (0.0-1.0)
        self.detection_method = "standard"  # 新增: 檢測方法標識

class FrameReader:
    """Reads frames from camera or video file in a separate thread."""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        self.cap = cv2.VideoCapture(self.video_source)
        self._configure_capture(frame_width, frame_height)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             self.actual_fps = self.target_fps # Use target if webcam FPS is unreliable

    def _configure_capture(self, frame_width, frame_height):
        if not self.use_video_file:
            # 網路攝像頭特殊設置，設置較高的優先級和緩衝區大小
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 嘗試設置網路攝像頭特有的緩衝區設定 (如果支援)
            try:
                # 降低緩衝區以減少延遲
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")

    def _read_frames(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False # End of video or camera error
                    self.frame_queue.put((False, None)) # Signal end
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(1.0 / (self.target_fps * 2)) # Avoid busy-waiting if queue is full

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0) # Wait up to 1s for a frame
        except queue.Empty:
            return False, None # Timeout

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0) # Wait for thread to finish
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
        self.target_fps = target_fps # For webcam FPS calculation if needed

        self.reader = FrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps # Initial display FPS

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm # Nominal, used if perspective fails

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0 # ROI starts from top of the frame
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y

        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2
        self.center_line_start_x = self.center_x_global - CENTER_LINE_WIDTH_PIXELS // 2
        self.center_line_end_x = self.center_x_global + CENTER_LINE_WIDTH_PIXELS // 2
        
        # 根據是否為網路攝像頭來設定中心檢測參數
        self.center_detection_cooldown_s = WEBCAM_CENTER_DETECTION_COOLDOWN_S if not use_video_file else CENTER_DETECTION_COOLDOWN_S
        self.center_tolerance_px = WEBCAM_CENTER_TOLERANCE_PX if not use_video_file else 0
        
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
        
        # 網路攝像頭模式下使用更大的事件緩衝區
        buffer_size = WEBCAM_EVENT_BUFFER_SIZE if not use_video_file else EVENT_BUFFER_SIZE_CENTER_CROSS
        self.event_buffer_center_cross = deque(maxlen=buffer_size)
        
        # 網路攝像頭模式下的預測參數
        self.prediction_lookahead_frames = WEBCAM_PREDICTION_LOOKAHEAD_FRAMES if not use_video_file else PREDICTION_LOOKAHEAD_FRAMES
        self.prediction_levels = WEBCAM_PREDICTION_LEVELS if not use_video_file else 1
        
        # 速度平滑參數
        self.speed_smoothing_factor = WEBCAM_SPEED_SMOOTHING_FACTOR if not use_video_file else SPEED_SMOOTHING_FACTOR
        
        # 額外的冗余檢測功能
        self.backup_detection_enabled = WEBCAM_BACKUP_DETECTION_ENABLED and not use_video_file
        self.recent_predicted_x_points = deque(maxlen=10)  # 用於保存最近的球預測位置
        
        # 交叉信心閾值
        self.cross_confidence_threshold = WEBCAM_CROSS_CONFIDENCE_THRESHOLD
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self._precalculate_overlay()
        self._create_perspective_lookup_table()
        
        # 網路攝像頭模式下顯示額外提示
        if not self.use_video_file:
            print("使用網路攝像頭模式，已啟用特殊優化設定")

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: 切換計數 | D: 除錯模式 | Q/ESC: 退出"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        # ROI y is from 0 (top) to self.roi_height_px (bottom of ROI)
        # Global y is from 0 (top of frame) to self.frame_height (bottom of frame)
        # Perspective is calculated based on y within ROI
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10): # step by 10px
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        # y_global is the y-coordinate in the full frame.
        # We need y relative to the top of the defined visual field for perspective (which is self.roi_bottom_y)
        # Let's assume perspective change happens from frame top (far) to roi_bottom_y (nearer part of table visible)
        
        # y_eff is y relative to top of frame, capped at roi_bottom_y for perspective calc
        y_eff = min(y_global, self.roi_bottom_y) 
        
        # relative_y: 0 at frame_top (far), 1 at roi_bottom_y (near)
        # Ensure roi_bottom_y is not zero to prevent division by zero
        if self.roi_bottom_y == 0: # Should not happen if ROI is configured
            relative_y = 0.5 # Default to a mid-value
        else:
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)

        # Assuming table_length_cm corresponds to the depth visible on screen,
        # and frame_width is the on-screen width representation.
        # This needs careful thought: pixels_per_cm should vary with y.
        # The original ratios were for width variation. Let's use that.
        # A simpler model: assume far_side_width_cm is at y=0, near_side_width_cm is at y=roi_bottom_y
        
        # Effective width at this y, then pixels_per_cm_horizontal
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        
        # If ROI width is representative of this current_width_cm
        roi_width_px = self.roi_end_x - self.roi_start_x
        if current_width_cm > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px # cm per pixel
        else:
            pixel_to_cm_ratio = self.table_length_cm / self.frame_width # Fallback

        return pixel_to_cm_ratio


    def _update_display_fps(self):
        if self.use_video_file: # For video files, FPS is fixed
            self.display_fps = self.actual_fps
            return

        now = time.monotonic() # More suitable for interval timing
        self.frame_timestamps_for_fps.append(now)
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 0:
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                # 使用網路攝像頭專用的平滑係數
                fps_smoothing = WEBCAM_FPS_SMOOTHING_FACTOR if not self.use_video_file else FPS_SMOOTHING_FACTOR
                self.display_fps = (1 - fps_smoothing) * self.display_fps + fps_smoothing * measured_fps
                
                # 為網路攝像頭設置最小 FPS 下限，確保速度計算不會過度敏感
                if not self.use_video_file and self.display_fps < 15:
                    self.display_fps = 15
        self.last_frame_timestamp_for_fps = now

    def _preprocess_frame(self, frame):
        # ROI slicing. Make sure to copy if modifications are made to roi or gray_roi later.
        roi_sub_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur can help reduce noise before FMO
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        self.prev_frames_gray_roi.append(gray_roi_blurred)
        return roi_sub_frame, gray_roi_blurred # Return the original ROI for drawing, blurred for processing

    def _detect_fmo(self):
        if len(self.prev_frames_gray_roi) < 3:
            return None
        
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        
        try:
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error: # OTSU can fail on blank images
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        
        if OPENING_KERNEL_SIZE_FMO[0] > 0: # Avoid error if kernel size is (0,0)
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else:
            opened_mask = thresh_mask
        
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        
        potential_balls = []
        for i in range(1, num_labels): # Skip background label 0
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
                    'contour_roi': contours[0] if contours else None # Store contour
                })

        if not potential_balls: return None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        # Convert ROI coordinates to global frame coordinates
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y # y is from top of ROI
        
        current_timestamp = time.monotonic()
        if self.use_video_file: # For video files, use frame count for timing
            current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic() # System time for timeout
        
        if self.is_counting_active:
            self.check_center_crossing(cx_global, cy_global, current_timestamp)
        
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')


    def _select_best_ball_candidate(self, candidates):
        if not candidates: return None

        if not self.trajectory: # No history, pick most circular one if good enough
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular:
                return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area']) # Fallback to largest

        last_x_global, last_y_global, _ = self.trajectory[-1]

        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y

            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            
            # 網路攝像頭模式下使用更寬鬆的距離容忍度
            max_distance_ratio = 0.25 if not self.use_video_file else 0.2
            if distance > self.frame_width * max_distance_ratio:
                ball_info['distance_from_last'] = float('inf')

            consistency_score = 0
            if len(self.trajectory) >= 2: # Need at least two previous points for direction
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
                    consistency_score = max(0, cosine_similarity) # We only care about forward consistency
            ball_info['consistency'] = consistency_score
        
        # 網路攝像頭模式下調整評分權重
        if not self.use_video_file:
            # 在網路攝像頭模式下，更偏向連續性和圓形度
            for ball_info in candidates:
                score = (0.3 / (1.0 + ball_info['distance_from_last'])) + \
                        (0.5 * ball_info['consistency']) + \
                        (0.2 * ball_info['circularity'])
                ball_info['score'] = score
        else:
            # 原始評分計算
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
            print(f"計數開啟 (Session #{self.count_session_id}) - 目標: {self.max_net_speeds_to_collect} 筆速度.")
        else:
            print(f"計數關閉 (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"已收集 {len(self.collected_net_speeds)} 筆速度. 產生輸出...")
                self._generate_outputs_async() # Use async version
            self.output_generated_for_session = True # Prevent re-generating if toggled off/on quickly

    def check_center_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        """增強的中心線穿越檢測，針對網路攝像頭優化"""
        if self.last_ball_x_global is None:
            self.last_ball_x_global = ball_x_global
            return

        # 檢查冷卻時間 - 網路攝像頭模式下使用較短的冷卻時間
        time_since_last_net_cross = current_timestamp - self.last_net_crossing_detection_time
        if time_since_last_net_cross < self.center_detection_cooldown_s:
            self.last_ball_x_global = ball_x_global
            return

        # 記錄實際交叉事件
        self._record_actual_crossing(ball_x_global, ball_y_global, current_timestamp)
        
        # 預測未來交叉事件
        if not self.use_video_file or len(self.trajectory) >= 3:
            self._record_predicted_crossing(ball_x_global, ball_y_global, current_timestamp)
        
        # 網路攝像頭模式下使用額外的備用檢測
        if self.backup_detection_enabled and len(self.trajectory) >= 3:
            self._perform_backup_detection(ball_x_global, ball_y_global, current_timestamp)
        
        self.last_ball_x_global = ball_x_global

    def _record_actual_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        """檢測球是否實際穿越中心線"""
        # 使用容差值增強檢測 - 網路攝像頭模式下使用更大的容差
        tolerance = self.center_tolerance_px
        
        # 根據方向檢測穿越
        crossed_l_to_r = (self.last_ball_x_global < (self.center_line_end_x - tolerance) and 
                          ball_x_global >= (self.center_line_end_x - tolerance))
        crossed_r_to_l = (self.last_ball_x_global > (self.center_line_start_x + tolerance) and 
                          ball_x_global <= (self.center_line_start_x + tolerance))
        
        actual_crossing_detected = False
        if self.net_crossing_direction == 'left_to_right' and crossed_l_to_r: 
            actual_crossing_detected = True
        elif self.net_crossing_direction == 'right_to_left' and crossed_r_to_l: 
            actual_crossing_detected = True
        elif self.net_crossing_direction == 'both' and (crossed_l_to_r or crossed_r_to_l): 
            actual_crossing_detected = True

        if actual_crossing_detected and self.current_ball_speed_kmh > 0:
            # 建立一個實際穿越事件，信心度最高
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, 
                               predicted=False, confidence=1.0)
            event.detection_method = "actual_crossing"
            self.event_buffer_center_cross.append(event)

    def _record_predicted_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        """基於當前軌跡預測未來穿越"""
        if len(self.trajectory) < 3 or self.current_ball_speed_kmh <= 0:
            return
        
        # 取得多個時間點進行速度計算，提高預測精度
        positions = []
        timestamps = []
        
        for i in range(min(3, len(self.trajectory))):
            x, y, t = self.trajectory[-(i+1)]
            positions.append((x, y))
            timestamps.append(t)
        
        # 計算平均速度向量
        vx_sum, vy_sum = 0, 0
        time_diffs = 0
        
        for i in range(len(positions)-1):
            dt = timestamps[i] - timestamps[i+1]
            if dt > 0:
                vx = (positions[i][0] - positions[i+1][0]) / dt
                vy = (positions[i][1] - positions[i+1][1]) / dt
                vx_sum += vx
                vy_sum += vy
                time_diffs += 1
        
        if time_diffs == 0:
            return
            
        # 平均速度向量
        vx_avg = vx_sum / time_diffs
        vy_avg = vy_sum / time_diffs
        
        # 存儲已創建的預測事件的時間戳，避免重複
        predicted_timestamps = set()
        
        # 多級預測 - 網路攝像頭模式下預測更多點
        for i in range(self.prediction_levels):
            # 預測時間跨度 - 隨著 i 增加而增加
            prediction_horizon_time = (i + 1) * self.prediction_lookahead_frames / self.display_fps
            
            # 預測未來位置
            predicted_x = ball_x_global + vx_avg * prediction_horizon_time
            predicted_y = ball_y_global + vy_avg * prediction_horizon_time
            predicted_timestamp = current_timestamp + prediction_horizon_time
            
            # 根據預測級別降低信心度
            confidence = 1.0 - (i * 0.1)  # 每級降低 0.1 的信心度
            
            # 避免重複時間點
            rounded_timestamp = round(predicted_timestamp, 2)
            if rounded_timestamp in predicted_timestamps:
                continue
            
            # 保存此預測點
            self.recent_predicted_x_points.append((predicted_x, predicted_timestamp))
            
            # 判斷穿越方向
            predict_l_to_r = (ball_x_global < self.center_x_global and predicted_x >= self.center_x_global)
            predict_r_to_l = (ball_x_global > self.center_x_global and predicted_x <= self.center_x_global)
            
            # 根據設定方向過濾
            prediction_valid_for_direction = False
            if self.net_crossing_direction == 'left_to_right' and predict_l_to_r: 
                prediction_valid_for_direction = True
            elif self.net_crossing_direction == 'right_to_left' and predict_r_to_l: 
                prediction_valid_for_direction = True
            elif self.net_crossing_direction == 'both' and (predict_l_to_r or predict_r_to_l): 
                prediction_valid_for_direction = True
            
            if prediction_valid_for_direction:
                # 避免在短時間內添加多個預測
                can_add_prediction = True
                for ev in self.event_buffer_center_cross:
                    if ev.predicted and abs(ev.timestamp - predicted_timestamp) < 0.15:  # 150ms
                        can_add_prediction = False
                        break
                
                if can_add_prediction:
                    event = EventRecord(predicted_x, predicted_timestamp, self.current_ball_speed_kmh, 
                                       predicted=True, confidence=confidence)
                    event.detection_method = f"prediction_level_{i+1}"
                    self.event_buffer_center_cross.append(event)
                    predicted_timestamps.add(rounded_timestamp)
                    
                    if self.debug_mode:
                        print(f"預測穿越: x={predicted_x:.1f}, t={predicted_timestamp:.3f}, conf={confidence:.2f}")

    def _perform_backup_detection(self, ball_x_global, ball_y_global, current_timestamp):
        """備用檢測機制 - 基於軌跡分析的多點檢測"""
        if len(self.trajectory) < 4:
            return
        
        # 使用最近的 N 個軌跡點進行分析
        recent_points = list(self.trajectory)[-4:]
        recent_x = [p[0] for p in recent_points]
        
        # 檢查是否有跨越中心線的趨勢
        left_side_points = [x for x in recent_x if x < self.center_x_global]
        right_side_points = [x for x in recent_x if x > self.center_x_global]
        
        # 至少要有兩個點在每一側才夠有意義
        has_significant_distribution = len(left_side_points) >= 1 and len(right_side_points) >= 1
        
        # 檢查是否可能穿越中心線
        possible_crossing = False
        crossing_direction = None
        
        if has_significant_distribution:
            # 查看點的排序，判斷方向
            if recent_x[0] < self.center_x_global and recent_x[-1] > self.center_x_global:
                crossing_direction = 'left_to_right'
                possible_crossing = True
            elif recent_x[0] > self.center_x_global and recent_x[-1] < self.center_x_global:
                crossing_direction = 'right_to_left'
                possible_crossing = True
        
        if possible_crossing:
            # 檢查方向是否符合設定
            valid_direction = (self.net_crossing_direction == 'both' or 
                              self.net_crossing_direction == crossing_direction)
            
            if valid_direction and self.current_ball_speed_kmh > 0:
                # 計算信心度 - 基於點分佈的一致性
                x_diff = max(recent_x) - min(recent_x)
                y_values = [p[1] for p in recent_points]
                y_diff = max(y_values) - min(y_values)
                
                # 使用 x 和 y 的差異計算一致性得分
                if x_diff > 0 and y_diff > 0:
                    # 較大的 x 差異和較小的 y 差異表示更直的軌跡
                    trajectory_linearity = min(1.0, x_diff / (y_diff + 1e-5))
                    confidence = min(0.9, 0.5 + 0.4 * trajectory_linearity)  # 最高 0.9
                else:
                    confidence = 0.5  # 默認中等信心度
                
                # 添加時間戳，與常規預測有所區分
                crossing_timestamp = current_timestamp - 0.05  # 略早於當前時間
                
                # 檢查是否已經有非常接近的穿越事件
                can_add_backup = True
                for ev in self.event_buffer_center_cross:
                    if abs(ev.timestamp - crossing_timestamp) < 0.2:  # 200ms
                        can_add_backup = False
                        break
                
                if can_add_backup:
                    # 建立備用穿越事件
                    event = EventRecord(self.center_x_global, crossing_timestamp, 
                                       self.current_ball_speed_kmh, predicted=False, 
                                       confidence=confidence)
                    event.detection_method = "backup_trajectory_analysis"
                    self.event_buffer_center_cross.append(event)
                    
                    if self.debug_mode:
                        print(f"備用檢測: 方向={crossing_direction}, 信心度={confidence:.2f}")

    def _process_crossing_events(self):
        """優化的穿越事件處理方法，支持信心度篩選和優先處理"""
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_eval_time = time.monotonic()
        if self.use_video_file: 
            current_eval_time = self.frame_counter / self.actual_fps
        
        # 對事件進行分組和評分
        actual_events = []
        predicted_events = []
        
        # 按實際/預測分類事件
        for event in self.event_buffer_center_cross:
            if event.processed:
                continue
                
            # 高信心度事件或實際穿越
            if event.confidence >= self.cross_confidence_threshold or not event.predicted:
                if event.predicted:
                    predicted_events.append(event)
                else:
                    actual_events.append(event)
        
        # 檢查預測事件是否應該提交
        timed_out_predictions = []
        for event in predicted_events:
            # 檢查預測事件是否已"過期"
            if current_eval_time >= event.timestamp:
                event_age = current_eval_time - event.timestamp
                
                # 對於剛好過期的事件，立即處理
                if event_age < 0.15:  # 150ms 內優先處理
                    timed_out_predictions.append(event)
                # 對於過期較久的預測，降低優先級
                elif event_age < 0.3:  # 300ms 內仍考慮處理
                    # 額外降低信心度
                    event.confidence *= max(0.5, 1.0 - event_age)
                    if event.confidence >= self.cross_confidence_threshold * 0.7:
                        timed_out_predictions.append(event)
        
        # 合併有效事件清單
        events_to_commit = actual_events + timed_out_predictions
        
        # 根據信心度和時間戳排序
        events_to_commit.sort(key=lambda e: (e.timestamp, -e.confidence))
        
        # 標記所有要處理的事件為已處理
        for event in events_to_commit:
            event.processed = True
        
        # 標記與已處理事件時間接近的其他事件
        for i, event in enumerate(events_to_commit):
            for j, other_event in enumerate(self.event_buffer_center_cross):
                if not other_event.processed and abs(event.timestamp - other_event.timestamp) < 0.2:
                    other_event.processed = True
        
        # 處理事件並記錄速度
        for event in events_to_commit:
            if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                break # Stop if limit reached

            if not self.timing_started_for_session:
                self.timing_started_for_session = True
                self.first_ball_crossing_timestamp = event.timestamp
                relative_time = 0.0
            else:
                relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2)
            
            self.last_recorded_net_speed_kmh = event.speed_kmh
            self.collected_net_speeds.append(event.speed_kmh)
            self.collected_relative_times.append(relative_time)
            self.last_net_crossing_detection_time = event.timestamp
            
            status_msg = f"{event.detection_method}" if self.debug_mode else ("預測" if event.predicted else "實際")
            confidence_str = f", 信心度: {event.confidence:.2f}" if self.debug_mode else ""
            print(f"網速 #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg}{confidence_str})")

        # 清理已處理事件
        self.event_buffer_center_cross = deque(
            [e for e in self.event_buffer_center_cross if not e.processed],
            maxlen=len(self.event_buffer_center_cross)
        )

        # 檢查是否已達到目標數量
        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"已達到目標 {self.max_net_speeds_to_collect} 筆速度。生成輸出。")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION:
                self.is_counting_active = False

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return

        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob

        # 距離計算使用透視調整
        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        
        delta_t = t2 - t1
        if delta_t > 0:
            speed_cm_per_time_unit = dist_cm / delta_t
            # 轉換為 km/h
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR 
            
            if self.current_ball_speed_kmh > 0: # Apply smoothing if there's a previous speed
                # 使用網路攝像頭專用的平滑因子
                self.current_ball_speed_kmh = (1 - self.speed_smoothing_factor) * self.current_ball_speed_kmh + \
                                           self.speed_smoothing_factor * speed_kmh
            else:
                self.current_ball_speed_kmh = speed_kmh
            
            if self.debug_mode:
                print(f"Speed: {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else: # Should not happen if timestamps are monotonic
            self.current_ball_speed_kmh *= (1 - self.speed_smoothing_factor) # Decay speed if no movement or bad dt


    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        # y_g are global y coordinates. Convert to y_in_roi for lookup.
        y1_roi = y1_g - self.roi_top_y
        y2_roi = y2_g - self.roi_top_y

        y1_roi_rounded = round(y1_roi / 10) * 10
        y2_roi_rounded = round(y2_roi / 10) * 10
        
        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        
        # Calculate 2D pixel distance, then scale
        # This assumes the ratio applies isotropically (same for dx and dy)
        # More accurate would be to scale dx and dy separately if perspective affects them differently,
        # but that complicates the ratio calculation significantly (needs depth).
        # For now, use avg ratio on the hypotenuse.
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * avg_px_to_cm_ratio
        return real_distance_cm

    def _generate_outputs_async(self):
        if not self.collected_net_speeds:
            print("沒有速度數據可生成輸出。")
            return
        
        # Create copies for the thread to work on, preventing race conditions
        # if the main lists are modified (though they shouldn't be for a finished session)
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
        plt.title(f'網速計測 - {timestamp_str}', fontsize=16)
        plt.xlabel('相對時間 (秒)', fontsize=12)
        plt.ylabel('速度 (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(min_speed - y_range*0.1, max_speed + y_range*0.1)
        plt.figtext(0.02, 0.02, f"總數: {len(net_speeds)}, 最大: {max_speed:.1f}, 最小: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for title and figtext
        plt.savefig(chart_filename, dpi=150)
        plt.close() # Important to close figures when done in non-interactive mode

        # Generate TXT
        txt_filename = f'{output_dir_path}/speed_data_{timestamp_str}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"網速數據 - Session {session_id} - {timestamp_str}\n")
            f.write("---------------------------------------\n")
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write("---------------------------------------\n")
            f.write(f"總測量點: {len(net_speeds)}\n")
            f.write(f"平均速度: {avg_speed:.1f} km/h\n")
            f.write(f"最高速度: {max_speed:.1f} km/h\n")
            f.write(f"最低速度: {min_speed:.1f} km/h\n")

        # Generate CSV
        csv_filename = f'{output_dir_path}/speed_data_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['時間戳', '序號', '相對時間 (秒)', '速度 (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                writer.writerow([timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([])
            writer.writerow(['統計', '數值'])
            writer.writerow(['總測量點', len(net_speeds)])
            writer.writerow(['平均速度 (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['最高速度 (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['最低速度 (km/h)', f"{min_speed:.1f}"])
        
        print(f"輸出文件已儲存到 {output_dir_path}")


    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        # Operate on a copy for drawing if display_frame is the original from reader
        vis_frame = display_frame # Assume display_frame is safe to draw on
        
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0

        if is_full_draw:
            # Add pre-calculated static overlay (ROI lines, center line)
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            # Draw trajectory
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)

        # Draw ball on ROI (if ball detected)
        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            # Draw on the sub-frame (roi_sub_frame) that will be placed back
            cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
            if frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            
            # Place modified ROI back into vis_frame. This is already handled if roi_sub_frame is a view.
            # If it's a copy, it needs to be put back:
            # vis_frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x] = frame_data_obj.roi_sub_frame

            # Also draw a marker on the main frame for the ball
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1)

            # 網路攝像頭模式顯示中心檢測容差區域
            if not self.use_video_file and self.center_tolerance_px > 0:
                cv2.line(vis_frame, 
                         (self.center_line_start_x + self.center_tolerance_px, 0), 
                         (self.center_line_start_x + self.center_tolerance_px, self.frame_height), 
                         (100, 100, 100), 1)
                cv2.line(vis_frame, 
                         (self.center_line_end_x - self.center_tolerance_px, 0), 
                         (self.center_line_end_x - self.center_tolerance_px, self.frame_height), 
                         (100, 100, 100), 1)


        # Text overlays (always drawn for fresh info)
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        count_status_text = "On" if frame_data_obj.is_counting_active else "off"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Count: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        if frame_data_obj.last_recorded_net_speed_kmh > 0:
            cv2.putText(vis_frame, f"Last Time: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        if frame_data_obj.collected_relative_times:
            cv2.putText(vis_frame, f"Recent time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)

        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 如果在網路攝像頭模式下，顯示額外的模式指示
        if not self.use_video_file:
            cv2.putText(vis_frame, "Webcam Mode", (self.frame_width - 220, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.debug_mode and frame_data_obj.debug_display_text:
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0
            # self.crossed_center = False # This state is handled by cooldown

    def process_single_frame(self, frame):
        self.frame_counter += 1
        self._update_display_fps()
            
        # Make a copy for processing if original frame needs to be pristine for display
        # If frame is directly from reader and will be drawn upon, it's fine.
        # For safety, let's assume `frame` is the one to be displayed, process on a copy.
        # processing_frame = frame.copy() # If preprocess_frame modifies input

        # roi_sub_frame is a view into `frame`. Modifying it modifies `frame`.
        # gray_roi is a new array.
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame) 
        
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        if motion_mask_roi is not None:
            # _detect_ball_in_roi updates self.trajectory, self.last_detection_timestamp,
            # and calls check_center_crossing which updates more state.
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi)
            # _calculate_ball_speed updates self.current_ball_speed_kmh
            self._calculate_ball_speed() 
        
        self._check_timeout_and_reset()
        
        # If counting, process any crossing events from buffer
        if self.is_counting_active:
            self._process_crossing_events()

        # Prepare FrameData for visualization
        # Pass copies of mutable lists to FrameData
        frame_data = FrameData(
            frame=frame, # The frame that will be drawn upon
            roi_sub_frame=roi_sub_frame, # The ROI slice (potentially modified with ball drawing)
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi,
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=f"軌跡: {len(self.trajectory)}, 事件: {len(self.event_buffer_center_cross)}" if self.debug_mode else None,
            frame_counter=self.frame_counter
        )
        if self.trajectory: # Global trajectory points
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self):
        print("=== 乒乓球速度追蹤系統 v12 (網路攝像頭優化版) ===")
        print(self.instruction_text)
        print(f"透視參數: 近端 {self.near_side_width_cm}cm, 遠端 {self.far_side_width_cm}cm")
        print(f"網線穿越方向: {self.net_crossing_direction}")
        print(f"目標收集速度筆數: {self.max_net_speeds_to_collect}")
        if not self.use_video_file:
            print(f"網路攝像頭模式已啟用以下優化:")
            print(f" - 多級預測 ({self.prediction_levels} 級)")
            print(f" - 中心線容差: {self.center_tolerance_px} 像素")
            print(f" - 備用檢測: {'啟用' if self.backup_detection_enabled else '停用'}")
            print(f" - 冷卻時間: {self.center_detection_cooldown_s:.3f}s")
        if self.debug_mode: print("除錯模式 已啟用.")

        self.running = True
        self.reader.start()
        
        window_name = '乒乓球速度追蹤系統 v12'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Or WINDOW_NORMAL for resizable

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("影片結束或幀讀取錯誤.")
                    else: print("相機錯誤或串流結束.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("流程結束但還有待處理數據. 生成輸出.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                
                # frame is the original frame from camera.
                # process_single_frame will internally handle views/copies as needed
                # and return a FrameData object.
                # The `frame` attribute in FrameData will be the same `frame` passed here.
                frame_data_obj = self.process_single_frame(frame)
                
                # _draw_visualizations will draw ON the frame_data_obj.frame
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # ESC
                    self.running = False
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("退出但還有待處理數據. 生成輸出.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True # Mark as generated
                    break
                elif key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"除錯模式: {'開啟' if self.debug_mode else '關閉'}")

        except KeyboardInterrupt:
            print("使用者中斷處理 (Ctrl+C).")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                print("被中斷但還有待處理數據. 生成輸出.")
                self._generate_outputs_async()
                self.output_generated_for_session = True # Mark as generated
        finally:
            self.running = False
            print("關閉系統...")
            self.reader.stop()
            print("影像讀取器已停止.")
            # Wait for file writing tasks to complete
            self.file_writer_executor.shutdown(wait=True)
            print("檔案寫入器已停止.")
            cv2.destroyAllWindows()
            print("系統關閉完成.")


def main():
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤系統 v12 (網路攝像頭優化版)')
    parser.add_argument('--video', type=str, default=None, help='影片檔案路徑. 如果不指定則使用網路攝像頭.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='網路攝像頭索引.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='目標幀率.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='幀寬度.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='幀高度.')
    parser.add_argument('--table_len', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='球桌長度 (cm).')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='球檢測超時 (秒).')
    
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                        choices=['left_to_right', 'right_to_left', 'both'], help='網線穿越方向.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='每次收集速度筆數.')
    
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='ROI 近端實際寬度 (cm).')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='ROI 遠端實際寬度 (cm).')
    
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='啟用除錯輸出.')

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