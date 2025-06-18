#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v16 (MODIFIED FOR AREA-BASED PERSPECTIVE CALIBRATION)
# Lightweight, optimized, multi-threaded, with area-based depth estimation.

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
DEFAULT_DETECTION_TIMEOUT = 0.2
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.85
MAX_TRAJECTORY_POINTS = 120

# Center Line Detection Related
MAX_NET_SPEEDS_TO_COLLECT = 30
NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output'

# Perspective Correction (Now used for calibration, not direct calculation)
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
# NEW: Area Smoothing
AREA_SMOOTHING_WINDOW = 5 # Use the average of the last 5 detected areas for stability

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
VISUALIZATION_DRAW_INTERVAL = 2

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 30
EVENT_BUFFER_SIZE_CENTER_CROSS = 200

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 10)
except AttributeError:
    cv2.setNumThreads(10)

class FrameData:
    """Data structure for passing frame-related information."""
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0, viz_text=""):
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
        self.viz_text = viz_text # For displaying calibration status etc.

class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False

class FrameReader:
    """Reads frames from camera or video file in a separate thread."""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
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

    def get_properties(self):
        return self.actual_fps, self.frame_width, self.frame_height

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM,
                 detection_timeout_s=DEFAULT_DETECTION_TIMEOUT, use_video_file=False,
                 video_file_path=None, target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=DEBUG_MODE_DEFAULT,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
                 max_net_speeds=MAX_NET_SPEEDS_TO_COLLECT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT):
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.video_file_path = video_file_path
        self.target_fps = target_fps

        self.reader = FrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_width_px = self.roi_end_x - self.roi_start_x

        # MODIFIED: Trajectory now stores area as well for the new calculation method
        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()
        
        # NEW: For area-based perspective correction
        self.area_history = deque(maxlen=AREA_SMOOTHING_WINDOW)
        self.calibration_mode = 'none' # 'none', 'near', 'far'
        self.calibrated_area_near_px = None
        self.calibrated_area_far_px = None
        self.calibrated_ratio_near = near_width_cm / self.roi_width_px if self.roi_width_px > 0 else 0
        self.calibrated_ratio_far = far_width_cm / self.roi_width_px if self.roi_width_px > 0 else 0
        self.nominal_pixel_to_cm_ratio = self.table_length_cm / self.frame_width

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2
        self.net_crossing_direction = net_crossing_direction
        self.max_net_speeds_to_collect = max_net_speeds
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_recorded_net_speed_kmh = 0
        self.last_ball_x_global = None
        self.output_generated_for_session = False
        
        self.is_counting_active = False
        self.count_session_id = 0
        self.timing_started_for_session = False
        self.first_ball_crossing_timestamp = None
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self.ball_on_left_of_center = False
        self.last_committed_crossing_time = 0
        self.EFFECTIVE_CROSSING_COOLDOWN_S = 0.2
        self.CENTER_ZONE_WIDTH_PIXELS = self.frame_width * 0.05

        self._precalculate_overlay()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        # MODIFIED instruction text
        self.instruction_text = "SPACE: Toggle Count | C: Calibrate | D: Debug | Q/ESC: Quit"
        
    # REMOVED/OBSOLETE: The Y-based perspective logic is replaced by area-based logic.
    # def _create_perspective_lookup_table(self): ...
    # def _get_pixel_to_cm_ratio(self, y_global): ...

    # NEW: Area-based perspective correction function
    def _get_ratio_from_smoothed_area(self, smoothed_area):
        if self.calibrated_area_near_px is None or self.calibrated_area_far_px is None or self.calibrated_area_near_px <= self.calibrated_area_far_px:
            # 如果未校準或校準值無效，返回一個基於全域的標稱值
            return self.nominal_pixel_to_cm_ratio

        # 面積與距離的平方成反比，使用面積的平方根進行線性內插會更準確
        sqrt_area = math.sqrt(max(1, smoothed_area))
        sqrt_area_near = math.sqrt(self.calibrated_area_near_px)
        sqrt_area_far = math.sqrt(self.calibrated_area_far_px)
        
        # 計算內插比例 (0 代表在最遠處, 1 代表在最近處)
        interp_factor = (sqrt_area - sqrt_area_far) / (sqrt_area_near - sqrt_area_far)
        
        # 將因子限制在 [0, 1] 範圍內，以處理超出校準範圍的面積值
        interp_factor = np.clip(interp_factor, 0.0, 1.0)
        
        # 線性內插計算當前的 cm/pixel 比例
        current_ratio = self.calibrated_ratio_far * (1 - interp_factor) + self.calibrated_ratio_near * interp_factor
        return current_ratio

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
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        if OPENING_KERNEL_SIZE_FMO[0] > 0:
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else: opened_mask = thresh_mask
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        potential_balls = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
                w_roi, h_roi = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                circularity = 0; contour_to_store = None
                if max(w_roi, h_roi) > 0:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]
                        contour_to_store = cnt
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: circularity = 4 * math.pi * area / (perimeter * perimeter)
                potential_balls.append({'position_roi': (int(cx_roi), int(cy_roi)), 'area': area,
                                        'circularity': circularity, 'contour_roi': contour_to_store})
        if not potential_balls: return None, None, None, None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None, None, None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file: current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()
        # MODIFIED: Store area in trajectory
        ball_area = best_ball_info['area']
        self.trajectory.append((cx_global, cy_global, current_timestamp, ball_area))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{ball_area:.1f}, Circ:{best_ball_info['circularity']:.3f}, T:{current_timestamp:.3f}")
        
        # MODIFIED: Return ball area
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi'), (cx_global, cy_global), current_timestamp, ball_area

    def _select_best_ball_candidate(self, candidates):
        if not candidates: return None
        if not self.trajectory:
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular: return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])

        last_x_global, last_y_global, _, _ = self.trajectory[-1]
        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global, cy_global = cx_roi + self.roi_start_x, cy_roi + self.roi_top_y
            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            if distance > self.frame_width * 0.4: ball_info['distance_from_last'] = float('inf')
            consistency_score = 0
            if len(self.trajectory) >= 2:
                prev_x_global, prev_y_global, _, _ = self.trajectory[-2]
                vec_hist_dx, vec_hist_dy = last_x_global - prev_x_global, last_y_global - prev_y_global
                vec_curr_dx, vec_curr_dy = cx_global - last_x_global, cy_global - last_y_global
                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist, mag_curr = math.sqrt(vec_hist_dx**2 + vec_hist_dy**2), math.sqrt(vec_curr_dx**2 + vec_curr_dy**2)
                if mag_hist > 0 and mag_curr > 0: cosine_similarity = dot_product / (mag_hist * mag_curr)
                else: cosine_similarity = 0
                consistency_score = max(0, cosine_similarity)
            ball_info['consistency'] = consistency_score
        for ball_info in candidates:
            ball_info['score'] = (0.3 / (1.0 + ball_info['distance_from_last'])) + \
                                (0.5 * ball_info['consistency']) + \
                                (0.2 * ball_info['circularity'])
        return max(candidates, key=lambda b: b['score'])

    def toggle_counting(self):
        if self.calibration_mode != 'none':
            print("Please exit calibration mode first (press 'C' until normal).")
            return
        self.is_counting_active = not self.is_counting_active
        if self.is_counting_active:
            self.count_session_id += 1; self.collected_net_speeds = []; self.collected_relative_times = []
            self.timing_started_for_session = False; self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear(); self.output_generated_for_session = False
            self.ball_on_left_of_center = False; self.last_committed_crossing_time = 0; self.last_ball_x_global = None
            print(f"Counting ON (Session #{self.count_session_id}) - Target: {self.max_net_speeds_to_collect} speeds.")
        else:
            print(f"Counting OFF (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_outputs_async()
            self.output_generated_for_session = True

    # NEW: Handle calibration key presses
    def handle_calibration_keypress(self, key):
        if key == ord('c'):
            if self.calibration_mode == 'none':
                self.calibration_mode = 'near'
                print("--- CALIBRATION START: Place ball at NEAR side of ROI, then press 'S' to set. ---")
            elif self.calibration_mode == 'near':
                self.calibration_mode = 'far'
                print("--- CALIBRATION: Place ball at FAR side of ROI, then press 'S' to set. ---")
            elif self.calibration_mode == 'far':
                self.calibration_mode = 'none'
                print("--- CALIBRATION ENDED. ---")
        
        elif key == ord('s') and self.calibration_mode != 'none':
            if not self.area_history:
                print("!!! Calibration Error: No ball detected. Cannot set area. Make sure ball is visible.")
                return

            smoothed_area = sum(self.area_history) / len(self.area_history)
            if self.calibration_mode == 'near':
                self.calibrated_area_near_px = smoothed_area
                print(f"+++ NEAR area set to: {self.calibrated_area_near_px:.2f} px^2 +++")
                self.calibration_mode = 'far' # Auto-advance
                print("--- CALIBRATION: Now place ball at FAR side of ROI, then press 'S' to set. ---")
            elif self.calibration_mode == 'far':
                self.calibrated_area_far_px = smoothed_area
                print(f"+++ FAR area set to: {self.calibrated_area_far_px:.2f} px^2 +++")
                if self.calibrated_area_near_px is not None and self.calibrated_area_near_px <= self.calibrated_area_far_px:
                    print("!!! WARNING: Near area is smaller or equal to Far area. Calibration might be incorrect. Recalibrate if needed. !!!")
                self.calibration_mode = 'none' # End calibration
                print("--- CALIBRATION COMPLETE. Ready for tracking. ---")


    def _record_potential_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        # This function's logic remains the same as v15
        if not self.is_counting_active: self.last_ball_x_global = ball_x_global; return
        if self.net_crossing_direction not in ['right_to_left', 'both']: self.last_ball_x_global = ball_x_global; return
        if current_timestamp - self.last_committed_crossing_time < self.EFFECTIVE_CROSSING_COOLDOWN_S: self.last_ball_x_global = ball_x_global; return

        crossed_r_to_l_strictly = False
        if self.last_ball_x_global is not None and self.last_ball_x_global >= self.center_x_global and ball_x_global < self.center_x_global and not self.ball_on_left_of_center:
            crossed_r_to_l_strictly = True
        if crossed_r_to_l_strictly and self.current_ball_speed_kmh > 0.1:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)

        if not crossed_r_to_l_strictly and not self.ball_on_left_of_center and len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 0.1:
            pt1_x, _, pt1_t, _ = self.trajectory[-2]; pt2_x, _, pt2_t, _ = self.trajectory[-1]
            if pt1_x >= self.center_x_global:
                delta_t_hist = pt2_t - pt1_t
                if delta_t_hist > 0:
                    vx_pixels_per_time_unit = (pt2_x - pt1_x) / delta_t_hist
                    min_vx_for_prediction = - (self.frame_width * 0.005) * (delta_t_hist / (1.0 / (self.display_fps if self.display_fps > 1 else self.target_fps)))
                    if vx_pixels_per_time_unit < min_vx_for_prediction:
                        for lookahead_frames in [1, 2, 3]:
                            time_to_predict = lookahead_frames / (self.display_fps if self.display_fps > 0 else self.target_fps)
                            predicted_x_at_crossing_time = ball_x_global + vx_pixels_per_time_unit * time_to_predict
                            predicted_timestamp = current_timestamp + time_to_predict
                            if predicted_x_at_crossing_time < self.center_x_global:
                                can_add_prediction = True
                                for ev in self.event_buffer_center_cross:
                                    if ev.predicted and abs(ev.timestamp - predicted_timestamp) < (1.0 / (self.display_fps if self.display_fps > 0 else self.target_fps)): can_add_prediction = False; break
                                if can_add_prediction:
                                    event = EventRecord(predicted_x_at_crossing_time, predicted_timestamp, self.current_ball_speed_kmh, predicted=True)
                                    self.event_buffer_center_cross.append(event); break
        
        if ball_x_global < self.center_x_global - self.CENTER_ZONE_WIDTH_PIXELS: self.ball_on_left_of_center = True
        elif ball_x_global > self.center_x_global + self.CENTER_ZONE_WIDTH_PIXELS: self.ball_on_left_of_center = False
        self.last_ball_x_global = ball_x_global

    def _process_crossing_events(self):
        # This function's logic remains the same as v15
        if not self.is_counting_active or self.output_generated_for_session: return
        current_processing_time = time.monotonic();
        if self.use_video_file: current_processing_time = self.frame_counter / self.actual_fps
        processed_event_this_cycle = False; temp_event_list = sorted(list(self.event_buffer_center_cross), key=lambda e: e.timestamp)
        new_event_buffer = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS); committed_event_ts = -1
        
        actual_event_to_commit = None
        for event in temp_event_list:
            if not event.processed and not event.predicted and event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                actual_event_to_commit = event; break
        if actual_event_to_commit:
            event = actual_event_to_commit
            if len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
                if not self.timing_started_for_session: self.timing_started_for_session = True; self.first_ball_crossing_timestamp = event.timestamp
                relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2)
                self.last_recorded_net_speed_kmh = event.speed_kmh; self.collected_net_speeds.append(event.speed_kmh)
                self.collected_relative_times.append(relative_time); self.last_committed_crossing_time = event.timestamp
                self.ball_on_left_of_center = True; event.processed = True; processed_event_this_cycle = True; committed_event_ts = event.timestamp
            else: event.processed = True
        
        if not processed_event_this_cycle:
            predicted_event_to_commit = None
            for event in temp_event_list:
                if not event.processed and event.predicted and current_processing_time >= event.timestamp and event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                    predicted_event_to_commit = event; break
            if predicted_event_to_commit:
                event = predicted_event_to_commit
                if len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
                    if not self.timing_started_for_session: self.timing_started_for_session = True; self.first_ball_crossing_timestamp = event.timestamp
                    relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2)
                    self.last_recorded_net_speed_kmh = event.speed_kmh; self.collected_net_speeds.append(event.speed_kmh)
                    self.collected_relative_times.append(relative_time); self.last_committed_crossing_time = event.timestamp
                    self.ball_on_left_of_center = True; event.processed = True; processed_event_this_cycle = True; committed_event_ts = event.timestamp
                else: event.processed = True

        if committed_event_ts > 0:
            for event_in_list in temp_event_list:
                if not event_in_list.processed and abs(event_in_list.timestamp - committed_event_ts) < self.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0:
                    event_in_list.processed = True
        
        for event_in_list in temp_event_list:
            if not event_in_list.processed and (current_processing_time - event_in_list.timestamp < 2.0):
                new_event_buffer.append(event_in_list)
        self.event_buffer_center_cross = new_event_buffer

        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Collected {self.max_net_speeds_to_collect} net speeds. Generating output.")
            self._generate_outputs_async(); self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION: self.is_counting_active = False

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2: self.current_ball_speed_kmh = 0; return
        
        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1, area1 = p1_glob
        x2_glob, y2_glob, t2, area2 = p2_glob
        
        # MODIFIED: Use the new area-based distance calculation
        dist_cm = self._calculate_real_distance_cm_area_based(x1_glob, y1_glob, x2_glob, y2_glob)
        
        delta_t = t2 - t1
        if delta_t > 0.0001:
            speed_cm_per_s = dist_cm / delta_t
            speed_kmh = speed_cm_per_s * KMH_CONVERSION_FACTOR
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + SPEED_SMOOTHING_FACTOR * speed_kmh
            else: self.current_ball_speed_kmh = speed_kmh
        else: self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    # REPLACED: This is the new core calculation function
    def _calculate_real_distance_cm_area_based(self, x1_g, y1_g, x2_g, y2_g):
        if not self.area_history: return 0.0
        
        # 使用平滑後的面積來計算比例
        smoothed_area = sum(self.area_history) / len(self.area_history)
        pixel_to_cm_ratio = self._get_ratio_from_smoothed_area(smoothed_area)
        
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * pixel_to_cm_ratio
        
        if self.debug_mode:
            print(f"DistCalc: SmArea={smoothed_area:.1f}, Ratio={pixel_to_cm_ratio:.4f}, PxDist={pixel_distance:.1f}, CmDist={real_distance_cm:.2f}")

        return real_distance_cm

    def _generate_outputs_async(self):
        # This function's logic remains the same as v15
        if not self.collected_net_speeds: print("No speed data to generate output."); return
        speeds_copy = list(self.collected_net_speeds); times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        # This function's logic remains the same as v15
        if not net_speeds: return
        output_dir_path = ""; base_filename = ""
        if self.use_video_file and self.video_file_path:
            try:
                output_dir_path = os.path.dirname(self.video_file_path)
                video_filename_stem = os.path.splitext(os.path.basename(self.video_file_path))[0]
                parts = video_filename_stem.split('_'); base_filename = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else video_filename_stem
            except Exception: self.use_video_file = False
        if not self.use_video_file or not self.video_file_path:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"; os.makedirs(output_dir_path, exist_ok=True)
            base_filename = f"speed_data_{timestamp_str}"
        avg_speed = sum(net_speeds) / len(net_speeds); max_speed = max(net_speeds); min_speed = min(net_speeds)
        chart_filename = os.path.join(output_dir_path, f'{base_filename}_chart.png')
        txt_filename = os.path.join(output_dir_path, f'{base_filename}_data.txt')
        csv_filename = os.path.join(output_dir_path, f'{base_filename}_data.csv')
        plt.figure(figsize=(12, 7)); plt.plot(relative_times, net_speeds, 'o-', label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        for t, s in zip(relative_times, net_speeds): plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Net Crossing Speeds - Session {session_id} - File: {base_filename}'); plt.xlabel('Relative Time (s)'); plt.ylabel('Speed (km/h)'); plt.grid(True); plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(chart_filename, dpi=150); plt.close()
        with open(txt_filename, 'w') as f:
            f.write(f"Net Speeds - Session {session_id}\n");
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write(f"\nStats:\nAvg: {avg_speed:.1f} km/h, Max: {max_speed:.1f}, Min: {min_speed:.1f}\n")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['Session ID', 'Point', 'Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): writer.writerow([session_id, i+1, f"{t:.2f}", f"{s:.1f}"])
        print(f"Output files for session {session_id} saved to '{output_dir_path}'.")

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        vis_frame = display_frame
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0
        if is_full_draw:
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)
        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
            if frame_data_obj.ball_contour_in_roi is not None: cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            cx_global_vis = cx_roi + self.roi_start_x; cy_global_vis = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global_vis, cy_global_vis), 8, BALL_COLOR_BGR, -1)
        
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"; count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        if frame_data_obj.last_recorded_net_speed_kmh > 0: cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # NEW: Display calibration/debug info
        if frame_data_obj.viz_text:
             cv2.putText(vis_frame, frame_data_obj.viz_text, (10, self.frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.debug_mode and frame_data_obj.debug_display_text:
             cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.area_history.clear() # NEW: Clear area history as well
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        # === START OF FIX ===
        # 檢查傳入影格的尺寸是否與系統初始化的尺寸相符
        h, w, _ = frame.shape
        if h != self.frame_height or w != self.frame_width:
            # 如果不符，強制將其調整為基準尺寸
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        # === END OF FIX ===

        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_global_coords, ball_timestamp = None, None

        if motion_mask_roi is not None:
            # MODIFIED: Get area from detection
            ball_pos_in_roi, ball_contour_in_roi, ball_global_coords, ball_timestamp, ball_area = self._detect_ball_in_roi(motion_mask_roi)
            if ball_pos_in_roi: # Ball was detected
                self.area_history.append(ball_area) # NEW: Add raw area to history for smoothing
                self._calculate_ball_speed()
                if self.is_counting_active: self._record_potential_crossing(ball_global_coords[0], ball_global_coords[1], ball_timestamp)
            else: self.last_ball_x_global = None
        else: self.last_ball_x_global = None

        self._check_timeout_and_reset()
        if self.is_counting_active: self._process_crossing_events()

        # NEW: Prepare visualization text for calibration and debug
        viz_text = ""
        if self.calibration_mode == 'near': viz_text = "CALIBRATING (NEAR): Place ball & press 'S'"
        elif self.calibration_mode == 'far': viz_text = "CALIBRATING (FAR): Place ball & press 'S'"
        elif self.calibrated_area_near_px is None: viz_text = "NOT CALIBRATED! Press 'C' to start."
        
        debug_text = None
        if self.debug_mode:
            smoothed_area = sum(self.area_history) / len(self.area_history) if self.area_history else 0
            debug_text = f"SmArea:{smoothed_area:.1f} NearA:{self.calibrated_area_near_px} FarA:{self.calibrated_area_far_px}"

        frame_data = FrameData(
            frame=frame, roi_sub_frame=roi_sub_frame, ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi, current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps, is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=debug_text, frame_counter=self.frame_counter, viz_text=viz_text
        )
        if self.trajectory: frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker (v16 - Area Calibration) ===")
        print(self.instruction_text)
        print(f"!!! IMPORTANT: System requires calibration. Press 'C' to begin. !!!")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True; self.reader.start()
        window_name = 'Ping Pong Speed Tracker v16 (Area Cal)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        self._generate_outputs_async(); self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False; break
                elif key == ord(' '): self.toggle_counting()
                elif key == ord('d'): self.debug_mode = not self.debug_mode; print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('c') or key == ord('s'):
                    self.handle_calibration_keypress(key)
        
        except KeyboardInterrupt: print("Process interrupted by user (Ctrl+C).")
        finally:
            self.running = False; print("Shutting down...")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                print("Exiting with pending data. Generating output.")
                self._generate_outputs_async()
            self.reader.stop(); print("Frame reader stopped.")
            self.file_writer_executor.shutdown(wait=True); print("File writer stopped.")
            cv2.destroyAllWindows(); print("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker v16 (Area Calibration)')
    # Arguments are the same as v15
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
    # 修正了這裡的 add_-argument
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='Frame height.')
    parser.add_argument('--table_len', type=float, default=DEFAULT_TABLE_LENGTH_CM, help='Table length (cm) for nominal px/cm.')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='Ball detection timeout (s).')
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                        choices=['left_to_right', 'right_to_left', 'both'], help='Net crossing direction to record.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='Number of net speeds to collect per session.')
    parser.add_argument('--near_width', type=float, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at near side.')
    parser.add_argument('--far_width', type=float, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at far side.')
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='Enable debug printouts.')
    args = parser.parse_args()

    tracker = PingPongSpeedTracker(
        video_source=args.video if args.video else args.camera_idx, table_length_cm=args.table_len,
        detection_timeout_s=args.timeout, use_video_file=True if args.video else False,
        video_file_path=args.video, target_fps=args.fps, frame_width=args.width, frame_height=args.height,
        debug_mode=args.debug, net_crossing_direction=args.direction,
        max_net_speeds=args.count, near_width_cm=args.near_width, far_width_cm=args.far_width
    )
    tracker.run()

if __name__ == '__main__':
    main()