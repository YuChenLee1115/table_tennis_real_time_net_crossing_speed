#!/usr/bin/env python3
# 乒乓球速度追蹤系統 macOS 優化版 (使用 AVFoundation 後端)
# Optimized for macOS with AVFoundation backend for 120fps performance
# 球體偵測方法已更新

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

# --- NEW Ball Detection Parameters (Replaces FMO Parameters) ---
# HSV Color Range for White Ball (Adjust for your specific lighting and ball color)
LOWER_HSV_BALL = np.array([0, 0, 180])    # Lower HSV for white
UPPER_HSV_BALL = np.array([179, 70, 255]) # Upper HSV for white
# Gaussian Blur for Preprocessing (Frame Differencing)
GAUSSIAN_BLUR_KERNEL_SIZE_FD = (11, 11) # Kernel size for Gaussian blur for frame differencing
# Color Mask Morphology
COLOR_MASK_ERODE_KERNEL_SIZE = (3,3)
COLOR_MASK_DILATE_KERNEL_SIZE = (7,7)
# Frame Difference Thresholding
FRAME_DIFF_THRESHOLD_VALUE = 25
# Final Mask Morphology (after thresholding frame difference)
FINAL_MASK_CLOSE_KERNEL_SIZE = (9,9)
FINAL_MASK_DILATE_ITERATIONS = 1 # Optional: additional dilation for the final mask

# Ball Properties (Contour Filtering)
MIN_BALL_AREA_PX = 30 # Adjusted from 10
MAX_BALL_AREA_PX = 8000 # Adjusted, can be dynamic based on ROI size
MIN_BALL_CIRCULARITY = 0.5 # Adjusted from 0.32, for more ball-like shapes

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
    cv2.setNumThreads(os.cpu_count() or 8)
except AttributeError:
    cv2.setNumThreads(8)

os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

class FrameData:
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0,
                 debug_mask_color=None, debug_mask_diff=None): # For debugging masks
        self.frame = frame
        self.roi_sub_frame = roi_sub_frame
        self.ball_position_in_roi = ball_position_in_roi
        self.ball_contour_in_roi = ball_contour_in_roi
        self.current_ball_speed_kmh = current_ball_speed_kmh
        self.display_fps = display_fps
        self.is_counting_active = is_counting_active
        self.collected_net_speeds = collected_net_speeds if collected_net_speeds is not None else []
        self.last_recorded_net_speed_kmh = last_recorded_net_speed_kmh
        self.collected_relative_times = collected_relative_times if collected_relative_times is not None else []
        self.debug_display_text = debug_display_text
        self.frame_counter = frame_counter
        self.trajectory_points_global = []
        self.debug_mask_color = debug_mask_color # Store color mask for debugging
        self.debug_mask_diff = debug_mask_diff   # Store final diff mask for debugging


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
             self.actual_fps = self.target_fps

    def _configure_capture(self, frame_width, frame_height):
        if not self.use_video_file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
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
                    self.running = False
                    self.frame_queue.put((False, None))
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(1.0 / (self.target_fps * 4))

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
        self.display_fps = self.actual_fps

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y
        self.roi_width_px = self.roi_end_x - self.roi_start_x


        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        # === NEW: Parameters for the new detection method ===
        self.lower_hsv_ball = LOWER_HSV_BALL
        self.upper_hsv_ball = UPPER_HSV_BALL
        self.gaussian_blur_kernel_fd = GAUSSIAN_BLUR_KERNEL_SIZE_FD
        self.color_mask_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, COLOR_MASK_ERODE_KERNEL_SIZE)
        self.color_mask_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, COLOR_MASK_DILATE_KERNEL_SIZE)
        self.frame_diff_thresh_val = FRAME_DIFF_THRESHOLD_VALUE
        self.final_mask_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FINAL_MASK_CLOSE_KERNEL_SIZE)
        self.final_mask_dilate_iterations = FINAL_MASK_DILATE_ITERATIONS
        # Store previous gray ROI for frame differencing
        self.prev_gray_roi_blurred = None
        # ====================================================


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
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10):
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y) 
        relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0) if self.roi_bottom_y > 0 else 0.5
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        return current_width_cm / self.roi_width_px if current_width_cm > 0 and self.roi_width_px > 0 else self.table_length_cm / self.frame_width

    def _update_display_fps(self):
        if self.use_video_file:
            self.display_fps = self.actual_fps
            return
        now = time.monotonic()
        self.frame_timestamps_for_fps.append(now)
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 0:
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                self.display_fps = (1 - FPS_SMOOTHING_FACTOR) * self.display_fps + FPS_SMOOTHING_FACTOR * measured_fps
        self.last_frame_timestamp_for_fps = now

    def _preprocess_frame_roi(self, frame):
        """Extracts ROI, converts to gray, and blurs."""
        if self.roi_top_y >= self.roi_bottom_y or self.roi_start_x >= self.roi_end_x:
            # Invalid ROI, return None or handle error
            return None, None 
            
        roi_sub_frame_color = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        if roi_sub_frame_color.size == 0: return None, None # ROI resulted in empty slice

        gray_roi = cv2.cvtColor(roi_sub_frame_color, cv2.COLOR_BGR2GRAY)
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, self.gaussian_blur_kernel_fd, 0)
        return roi_sub_frame_color, gray_roi_blurred

    def _generate_ball_candidate_mask(self, current_color_roi, current_gray_roi_blurred, prev_gray_roi_blurred):
        """Generates a binary mask of potential ball locations using new method."""
        if prev_gray_roi_blurred is None:
            return None, None # Not enough frames yet

        # 1. Color Filtering
        hsv_roi = cv2.cvtColor(current_color_roi, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_roi, self.lower_hsv_ball, self.upper_hsv_ball)
        color_mask = cv2.erode(color_mask, self.color_mask_erode_kernel, iterations=1)
        color_mask = cv2.dilate(color_mask, self.color_mask_dilate_kernel, iterations=2)

        # 2. Frame Differencing
        frame_delta = cv2.absdiff(prev_gray_roi_blurred, current_gray_roi_blurred)
        
        # 3. Apply Color Mask to Difference
        frame_delta_masked = cv2.bitwise_and(frame_delta, frame_delta, mask=color_mask)

        # 4. Thresholding
        _, thresh_mask = cv2.threshold(frame_delta_masked, self.frame_diff_thresh_val, 255, cv2.THRESH_BINARY)

        # 5. Morphological Operations on final mask
        processed_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, self.final_mask_close_kernel, iterations=2)
        if self.final_mask_dilate_iterations > 0:
             processed_mask = cv2.dilate(processed_mask, None, iterations=self.final_mask_dilate_iterations)
        
        # For debugging
        debug_mask_color_resized = cv2.resize(color_mask, (processed_mask.shape[1], processed_mask.shape[0])) if color_mask is not None else None
        
        return processed_mask, debug_mask_color_resized


    def _find_and_select_ball_from_mask(self, processed_mask_roi):
        """Finds contours in the processed mask and selects the best ball candidate."""
        if processed_mask_roi is None:
            return None, None

        contours, _ = cv2.findContours(processed_mask_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_balls = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Adjust MAX_BALL_AREA_PX dynamically if ROI is small
            current_max_ball_area = min(MAX_BALL_AREA_PX, self.roi_width_px * self.roi_height_px * 0.5) # Max 50% of ROI area

            if MIN_BALL_AREA_PX < area < current_max_ball_area:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                # Use minEnclosingCircle to get center for position
                ((cx_roi, cy_roi), radius) = cv2.minEnclosingCircle(cnt)

                if circularity >= MIN_BALL_CIRCULARITY: # Pre-filter by circularity before heavier selection
                    potential_balls.append({
                        'position_roi': (int(cx_roi), int(cy_roi)),
                        'area': area,
                        'circularity': circularity,
                        'contour_roi': cnt, 
                        'radius_roi': radius
                    })
        
        if not potential_balls: return None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls) # Use existing selection logic
        if not best_ball_info: return None, None

        # Update last detection timestamp (moved here to be more specific to successful detection)
        current_timestamp = time.monotonic()
        if self.use_video_file:
            current_timestamp = self.frame_counter / self.actual_fps if self.actual_fps > 0 else current_timestamp
        self.last_detection_timestamp = time.monotonic() # System time for timeout

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        if self.is_counting_active:
            self.check_center_crossing(cx_global, current_timestamp)
        
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball (New): ROI({cx_roi},{cy_roi}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')


    def _select_best_ball_candidate(self, candidates):
        # This function is largely kept from the original, as its logic is sound.
        # It expects candidates to have 'position_roi', 'area', 'circularity'.
        # We added 'contour_roi' and 'radius_roi' which can be ignored by this specific selection part if not used.
        if not candidates: return None

        if not self.trajectory:
            highly_circular_and_good_area = [
                b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY and MIN_BALL_AREA_PX < b['area'] < MAX_BALL_AREA_PX / 2 # Prefer not too large initially
            ]
            if highly_circular_and_good_area:
                 return max(highly_circular_and_good_area, key=lambda b: b['circularity']) # Prefer most circular
            return max(candidates, key=lambda b: b['area'])


        last_x_global, last_y_global, _ = self.trajectory[-1]

        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y

            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            
            max_expected_dist = self.frame_width * 0.3 # Heuristic: 30% of frame width
            if distance > max_expected_dist:
                ball_info['distance_from_last'] = float('inf')

            consistency_score = 0
            if len(self.trajectory) >= 2:
                prev_x_global, prev_y_global, _ = self.trajectory[-2]
                vec_hist_dx = last_x_global - prev_x_global
                vec_hist_dy = last_y_global - prev_y_global
                vec_curr_dx = cx_global - last_x_global
                vec_curr_dy = cy_global - last_y_global
                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist = math.sqrt(vec_hist_dx**2 + vec_hist_dy**2)
                mag_curr = math.sqrt(vec_curr_dx**2 + vec_curr_dy**2)
                if mag_hist > 0 and mag_curr > 0:
                    cosine_similarity = dot_product / (mag_hist * mag_curr)
                    consistency_score = max(0, cosine_similarity) 
            ball_info['consistency'] = consistency_score
        
        for ball_info in candidates:
            score = (0.5 / (1.0 + ball_info['distance_from_last'] / (max_expected_dist or 1.0) )) + \
                    (0.3 * ball_info['consistency']) + \
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
                self._generate_outputs_async()
            self.output_generated_for_session = True

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
        crossed_l_to_r = (self.last_ball_x_global < self.center_x_global and ball_x_global >= self.center_x_global)
        crossed_r_to_l = (self.last_ball_x_global > self.center_x_global and ball_x_global <= self.center_x_global)
        
        actual_crossing_detected = False
        # 決定穿越方向是否符合設定
        if self.net_crossing_direction == 'left_to_right' and crossed_l_to_r: actual_crossing_detected = True
        elif self.net_crossing_direction == 'right_to_left' and crossed_r_to_l: actual_crossing_detected = True
        elif self.net_crossing_direction == 'both' and (crossed_l_to_r or crossed_r_to_l): actual_crossing_detected = True

        # --- 調整點 1: 放寬實際穿越的速度限制 ---
        # 即使速度計算暫時為0，只要發生了物理穿越，也應考慮記錄
        # 可以設定一個極低的速度門檻，或者允許速度為0時也記錄（後續處理時再過濾）
        # 此處改為>=0，即只要有計算出的速度（包括0）就記錄實際穿越
        if actual_crossing_detected and self.current_ball_speed_kmh >= 0: # 原為 > 0
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            if self.debug_mode: print(f"Debug: Actual crossing recorded. Speed: {self.current_ball_speed_kmh:.1f}")
            return # 實際穿越優先，記錄後直接返回

        # Predictive crossing (如果沒有實際穿越被記錄)
        # 預測仍然可以保留一個較低的速度門檻，因為預測依賴於有效的速度向量
        if len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 1.0: # 保留一個低速門檻給預測
            pt1_x, _, pt1_t = self.trajectory[-2]
            pt2_x, _, pt2_t = self.trajectory[-1]
            delta_t = pt2_t - pt1_t
            if delta_t > 0.001: # 避免除以零或過小的時間差
                vx_pixels_per_time_unit = (pt2_x - pt1_x) / delta_t
                
                prediction_horizon_time = PREDICTION_LOOKAHEAD_FRAMES / self.display_fps if self.display_fps > 1 else 0.1
                
                predicted_x_future = ball_x_global + vx_pixels_per_time_unit * prediction_horizon_time
                predicted_timestamp_future = current_timestamp + prediction_horizon_time

                predict_l_to_r = (ball_x_global < self.center_x_global and predicted_x_future >= self.center_x_global)
                predict_r_to_l = (ball_x_global > self.center_x_global and predicted_x_future <= self.center_x_global)
                
                prediction_valid_for_direction = False
                if self.net_crossing_direction == 'left_to_right' and predict_l_to_r: prediction_valid_for_direction = True
                elif self.net_crossing_direction == 'right_to_left' and predict_r_to_l: prediction_valid_for_direction = True
                elif self.net_crossing_direction == 'both' and (predict_l_to_r or predict_r_to_l): prediction_valid_for_direction = True

                if prediction_valid_for_direction:
                    # --- 調整點 2: 稍微縮短預測重複抑制的時間窗口 ---
                    can_add_prediction = True
                    # 檢查最近的預測事件，避免過於頻繁的相同預測
                    for ev in reversed(list(self.event_buffer_center_cross)): # 從最近的開始檢查
                        if ev.predicted and abs(ev.timestamp - predicted_timestamp_future) < 0.075:  # 原為 0.1 (100ms)，改為 75ms
                            can_add_prediction = False
                            break
                        if not ev.predicted and abs(ev.timestamp - predicted_timestamp_future) < 0.15: # 如果附近有實際事件，也不再做預測
                            can_add_prediction = False
                            break
                    
                    if can_add_prediction:
                        event = EventRecord(predicted_x_future, predicted_timestamp_future, self.current_ball_speed_kmh, predicted=True)
                        self.event_buffer_center_cross.append(event)
                        if self.debug_mode: print(f"Debug: Predicted crossing recorded. Speed: {self.current_ball_speed_kmh:.1f}")

    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_eval_time = time.monotonic()
        if self.use_video_file: 
            current_eval_time = self.frame_counter / self.actual_fps if self.actual_fps > 0 else current_eval_time
        
        events_to_commit = []
        
        # 遍歷事件緩衝區進行處理
        # 使用索引來安全地修改（或標記待刪除）
        temp_event_buffer = list(self.event_buffer_center_cross) # 操作副本以避免迭代時修改問題
        processed_indices_in_temp = []


        # 優先處理實際事件
        for i, event in enumerate(temp_event_buffer):
            if event.processed: continue # 跳過已處理的

            if not event.predicted:  # 這是實際事件
                # --- 調整點 3: 實際事件的速度過濾 ---
                # 即使記錄時速度為0，在這裡最終確認時可以加一個過濾
                if event.speed_kmh < 0.5 and len(self.collected_net_speeds) > 0 : # 例如，速度小於0.5km/h且不是第一次記錄，則忽略
                    if self.debug_mode: print(f"Debug: Actual event speed {event.speed_kmh:.1f} too low, ignoring.")
                    event.processed = True # 標記為已處理（忽略）
                    processed_indices_in_temp.append(i)
                    continue

                events_to_commit.append(event)
                event.processed = True 
                processed_indices_in_temp.append(i)
                
                # 使附近的預測事件失效
                for j, other_event in enumerate(temp_event_buffer):
                    if i != j and other_event.predicted and not other_event.processed:
                        # 如果預測事件的時間戳與實際事件非常接近，則認為此預測已被實際事件覆蓋
                        if abs(event.timestamp - other_event.timestamp) < 0.15:  # 150ms 窗口，原為0.2
                            other_event.processed = True
                            processed_indices_in_temp.append(j)
                            if self.debug_mode: print(f"Debug: Prediction at {other_event.timestamp:.3f} invalidated by actual event at {event.timestamp:.3f}")


        # 處理尚未被實際事件覆蓋的預測事件
        for i, event in enumerate(temp_event_buffer):
            if event.processed: continue # 跳過已處理的

            if event.predicted:
                # --- 調整點 4: 修改預測事件的採納條件 ---
                # 原條件: (current_eval_time - event.timestamp) > 0.05 (預測時間點已過50ms)
                # 新條件: current_eval_time >= event.timestamp (預測時間點已到或剛過)
                # 並且增加一個最大等待時間，如果預測時間點過去太久還沒被實際事件覆蓋，則採納
                prediction_age = current_eval_time - event.timestamp
                
                # 如果預測時間已到，或者預測時間稍未到但很接近 (例如在1幀的誤差內)
                time_threshold_for_acceptance = 1.5 / self.display_fps if self.display_fps > 0 else 0.025 # e.g. 1.5 frame time
                
                if prediction_age >= -time_threshold_for_acceptance : # 允許一點點提前（在預測時間點附近就考慮）
                    # 再次確認此預測事件附近沒有更優的（例如非預測的）事件
                    is_still_valid_prediction = True
                    for k, other_event_check in enumerate(temp_event_buffer):
                        if other_event_check.processed and not other_event_check.predicted and abs(event.timestamp - other_event_check.timestamp) < 0.1:
                            is_still_valid_prediction = False # 有一個已處理的實際事件離得很近
                            break
                    
                    if is_still_valid_prediction:
                         # 對於預測事件，也加上速度過濾
                        if event.speed_kmh < 1.0 and len(self.collected_net_speeds) > 0: # 預測事件的速度門檻可以稍高
                            if self.debug_mode: print(f"Debug: Predicted event speed {event.speed_kmh:.1f} too low, ignoring.")
                            event.processed = True
                            processed_indices_in_temp.append(i)
                            continue

                        events_to_commit.append(event)
                        if self.debug_mode: print(f"Debug: Committing PREDICTED event. Age: {prediction_age:.3f}s")
                    else:
                        if self.debug_mode: print(f"Debug: Predicted event at {event.timestamp:.3f} was superseded or too old, ignoring.")
                    
                    event.processed = True # 無論是否採納，都標記為已處理
                    processed_indices_in_temp.append(i)


        # 清理原始事件緩衝區中已處理的事件
        # 必須從後往前刪除，或者重建列表
        new_event_buffer = []
        for i, event in enumerate(self.event_buffer_center_cross):
             is_processed_in_original = False
             for temp_idx, temp_event in enumerate(temp_event_buffer):
                 if event is temp_event and temp_event.processed: # 檢查是否是同一個對象且被標記處理
                     is_processed_in_original = True
                     break
             if not is_processed_in_original:
                 new_event_buffer.append(event)
        self.event_buffer_center_cross = deque(new_event_buffer, maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)


        # 按時間戳排序準備提交的事件
        events_to_commit.sort(key=lambda e: e.timestamp)

        for event in events_to_commit:
            if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                if self.debug_mode: print("Debug: Max speeds collected, skipping further event commits.")
                break 

            if not self.timing_started_for_session:
                self.timing_started_for_session = True
                self.first_ball_crossing_timestamp = event.timestamp
                relative_time = 0.0
            else:
                # 確保relative_time不為負 (如果事件時間戳因某種原因早於first_ball_crossing_timestamp)
                relative_time = max(0.0, round(event.timestamp - self.first_ball_crossing_timestamp, 2))
            
            self.last_recorded_net_speed_kmh = event.speed_kmh
            self.collected_net_speeds.append(event.speed_kmh)
            self.collected_relative_times.append(relative_time)
            # 更新 last_net_crossing_detection_time 應該使用 event.timestamp，因為這是事件的實際或預測發生時間
            self.last_net_crossing_detection_time = event.timestamp 
            
            status_msg = "Pred" if event.predicted else "Actual"
            print(f"Net Speed #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg})")

        # 不需要這行了，因為上面已經通過重建列表的方式更新了
        # self.event_buffer_center_cross = deque(
        #     [e for e in self.event_buffer_center_cross if not e.processed],
        #     maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS
        # )

        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"目標 {self.max_net_speeds_to_collect} 個速度已收集。正在生成輸出。")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION:
                self.is_counting_active = False

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2: self.current_ball_speed_kmh = 0; return
        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob; x2_glob, y2_glob, t2 = p2_glob
        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        delta_t = t2 - t1
        if delta_t > 0:
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR 
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + SPEED_SMOOTHING_FACTOR * speed_kmh
            else: self.current_ball_speed_kmh = speed_kmh
            if self.debug_mode: print(f"Speed: {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else: self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        y1_roi = y1_g - self.roi_top_y; y2_roi = y2_g - self.roi_top_y
        y1_roi_rounded = round(y1_roi / 10) * 10; y2_roi_rounded = round(y2_roi / 10) * 10
        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        return pixel_distance * avg_px_to_cm_ratio

    def _generate_outputs_async(self):
        if not self.collected_net_speeds: print("No speed data to generate output."); return
        speeds_copy = list(self.collected_net_speeds); times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        """This method runs in a separate thread via ThreadPoolExecutor."""
        if not net_speeds: return
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
        os.makedirs(output_dir_path, exist_ok=True)
        avg_speed = sum(net_speeds) / len(net_speeds); max_speed = max(net_speeds); min_speed = min(net_speeds)
        chart_filename = f'{output_dir_path}/speed_chart_{timestamp_str}.png'
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        for t, s in zip(relative_times, net_speeds): plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Net Crossing Speeds - {timestamp_str}', fontsize=16); plt.xlabel('Relative Time (s)', fontsize=12); plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7); plt.legend()
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(min_speed - y_range*0.1, max_speed + y_range*0.1)
        plt.figtext(0.02, 0.02, f"Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(chart_filename, dpi=150); plt.close()
        txt_filename = f'{output_dir_path}/speed_data_{timestamp_str}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Net Speeds - Session {session_id} - {timestamp_str}\n---------------------------------------\n")
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write(f"---------------------------------------\nTotal Points: {len(net_speeds)}\nAverage Speed: {avg_speed:.1f} km/h\nMaximum Speed: {max_speed:.1f} km/h\nMinimum Speed: {min_speed:.1f} km/h\n")
        csv_filename = f'{output_dir_path}/speed_data_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): writer.writerow([timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([]); writer.writerow(['Statistic', 'Value']); writer.writerow(['Total Points', len(net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"]); writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"]); writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        print(f"Output files saved to {output_dir_path}")

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        vis_frame = display_frame
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0
        if is_full_draw:
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                trajectory_points = frame_data_obj.trajectory_points_global[::2] if len(frame_data_obj.trajectory_points_global) > 15 else frame_data_obj.trajectory_points_global
                pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)
        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            if is_full_draw and frame_data_obj.ball_contour_in_roi is not None: # Draw contour only on full draw
                # Draw on the ROI sub-frame for debugging if needed, but here we draw on main
                # cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
                # cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
                pass # Contour drawing on sub_frame can be enabled for detailed ROI view
            cx_global = cx_roi + self.roi_start_x; cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1) # Always draw ball marker
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"; count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        if frame_data_obj.last_recorded_net_speed_kmh > 0: cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        if frame_data_obj.collected_relative_times: cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if self.debug_mode and frame_data_obj.debug_display_text and is_full_draw: cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        # --- Debug Visualizations for Masks ---
        if self.debug_mode and is_full_draw:
            if frame_data_obj.debug_mask_color is not None:
                # Resize mask to a smaller fixed size for display
                h_roi, w_roi = frame_data_obj.debug_mask_color.shape[:2]
                debug_w, debug_h = 160, int(160 * h_roi / w_roi) if w_roi > 0 else 120
                
                mask_color_resized = cv2.resize(frame_data_obj.debug_mask_color, (debug_w, debug_h))
                cv2.imshow("Debug Color Mask (ROI)", mask_color_resized)
            if frame_data_obj.debug_mask_diff is not None:
                h_roi, w_roi = frame_data_obj.debug_mask_diff.shape[:2]
                debug_w, debug_h = 160, int(160 * h_roi / w_roi) if w_roi > 0 else 120

                mask_diff_resized = cv2.resize(frame_data_obj.debug_mask_diff, (debug_w, debug_h))
                cv2.imshow("Debug Final Mask (ROI)", mask_diff_resized)
        # --- End Debug Visualizations ---
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        start_time = time.monotonic()
        self.frame_counter += 1
        self._update_display_fps()
         
        # 1. Preprocess: Get color ROI and current blurred gray ROI
        roi_sub_frame_color, current_gray_roi_blurred = self._preprocess_frame_roi(frame)
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        processed_mask_for_contours = None # For debugging
        debug_color_mask_roi = None # For debugging

        if roi_sub_frame_color is not None and current_gray_roi_blurred is not None:
            # 2. Generate candidate mask using the new method
            # This needs the *previous* gray blurred ROI, which is stored in self.prev_gray_roi_blurred
            processed_mask_for_contours, debug_color_mask_roi = self._generate_ball_candidate_mask(
                roi_sub_frame_color, current_gray_roi_blurred, self.prev_gray_roi_blurred
            )
            
            # 3. Find and select ball from this mask
            if processed_mask_for_contours is not None:
                ball_pos_in_roi, ball_contour_in_roi = self._find_and_select_ball_from_mask(processed_mask_for_contours)
                if ball_pos_in_roi: # If a ball is successfully found and selected
                    self._calculate_ball_speed() # Calculate speed based on this new position
            
            # 4. Update previous frame for next iteration
            self.prev_gray_roi_blurred = current_gray_roi_blurred.copy() 
        
        self._check_timeout_and_reset() # If no ball found for a while
        
        if self.is_counting_active:
            self._process_crossing_events()

        frame_data = FrameData(
            frame=frame,
            roi_sub_frame=roi_sub_frame_color, # Pass color ROI
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi,
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=f"Traj:{len(self.trajectory)} Evt:{len(self.event_buffer_center_cross)}" if self.debug_mode else None,
            frame_counter=self.frame_counter,
            debug_mask_color=debug_color_mask_roi,      # Pass color mask for debug
            debug_mask_diff=processed_mask_for_contours # Pass final diff mask for debug
        )
        if self.trajectory:
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        frame_processing_time = time.monotonic() - start_time
        self.performance_history.append(frame_processing_time)
        if self.debug_mode and self.frame_counter % 60 == 0 and self.performance_history:
            avg_time = sum(self.performance_history) / len(self.performance_history)
            print(f"平均每幀處理時間: {avg_time*1000:.2f}ms (目標: {1000/self.target_fps:.2f}ms)")
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker (macOS Optimized) ===")
        print(self.instruction_text); print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing: {self.net_crossing_direction}, Target speeds: {self.max_net_speeds_to_collect}")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True; self.reader.start()
        window_name = 'Ping Pong Speed Tracker (macOS Optimized)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        if self.debug_mode:
            cv2.namedWindow("Debug Color Mask (ROI)", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Debug Final Mask (ROI)", cv2.WINDOW_AUTOSIZE)


        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("Video ended.") 
                    else: print("Camera error.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        self._generate_outputs_async(); self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        self._generate_outputs_async(); self.output_generated_for_session = True
                    break
                elif key == ord(' '): self.toggle_counting()
                elif key == ord('d'): self.debug_mode = not self.debug_mode; print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl+C).")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                self._generate_outputs_async(); self.output_generated_for_session = True
        finally:
            self.running = False; print("Shutting down...")
            self.reader.stop(); print("Frame reader stopped.")
            self.file_writer_executor.shutdown(wait=True); print("File writer stopped.")
            cv2.destroyAllWindows(); print("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker (macOS Optimized)')
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='Frame height.')
    parser.add_argument('--table_len', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='Table length (cm) for nominal px/cm.')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='Ball detection timeout (s).')
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT, choices=['left_to_right', 'right_to_left', 'both'], help='Net crossing direction.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='Number of net speeds to collect.')
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at near side.')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at far side.')
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='Enable debug printouts.')
    args = parser.parse_args()
    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False
    tracker = PingPongSpeedTracker(video_source=video_source_arg, table_length_cm=args.table_len, detection_timeout_s=args.timeout,
                                 use_video_file=use_video_file_arg, target_fps=args.fps, frame_width=args.width, frame_height=args.height,
                                 debug_mode=args.debug, net_crossing_direction=args.direction, max_net_speeds=args.count,
                                 near_width_cm=args.near_width, far_width_cm=args.far_width)
    tracker.run()

if __name__ == '__main__':
    main()