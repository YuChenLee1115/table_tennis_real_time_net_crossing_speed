#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11 (MODIFIED FOR ENHANCED R-L RECORDING & REDUCED DUPLICATES)
# Lightweight, optimized, multi-threaded (acquisition & I/O), macOS compatible

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
DEFAULT_DETECTION_TIMEOUT = 0.2 # From user's v11
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.85
MAX_TRAJECTORY_POINTS = 120

# Center Line Detection Related (MAX_NET_SPEEDS_TO_COLLECT is primary for count)
MAX_NET_SPEEDS_TO_COLLECT = 30
NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left' # Focus of this modification
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
VISUALIZATION_DRAW_INTERVAL = 2

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 30
EVENT_BUFFER_SIZE_CENTER_CROSS = 200 # Keep buffer size, but processing logic will change
# PREDICTION_LOOKAHEAD_FRAMES = 10 # Will use a more dynamic small lookahead

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
                 debug_display_text=None, frame_counter=0):
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

class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False # Important for new processing logic

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
                 video_file_path=None,  # <--- 新增此參數
                 target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=DEBUG_MODE_DEFAULT,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
                 max_net_speeds=MAX_NET_SPEEDS_TO_COLLECT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT):
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.video_file_path = video_file_path # <--- 新增此行以儲存路徑
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
        
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self.ball_on_left_of_center = False 
        self.last_committed_crossing_time = 0 
        self.EFFECTIVE_CROSSING_COOLDOWN_S = 0.2 
        self.CENTER_ZONE_WIDTH_PIXELS = self.frame_width * 0.05 

        self._precalculate_overlay()
        self._create_perspective_lookup_table()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # ROI Box lines
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        # Center line
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10):
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y)
        if self.roi_bottom_y == 0: relative_y = 0.5
        else: relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        roi_width_px = self.roi_end_x - self.roi_start_x
        if current_width_cm > 0 and roi_width_px > 0: # roi_width_px added for safety
            pixel_to_cm_ratio = current_width_cm / roi_width_px
        else:
            pixel_to_cm_ratio = self.table_length_cm / self.frame_width
        return pixel_to_cm_ratio

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

    def _detect_ball_in_roi(self, motion_mask_roi): # This method now primarily updates trajectory and returns ball info
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        potential_balls = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
                x_roi, y_roi = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w_roi, h_roi = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                circularity = 0; contour_to_store = None
                if max(w_roi, h_roi) > 0:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]
                        contour_to_store = cnt # Store the actual contour
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: circularity = 4 * math.pi * area / (perimeter * perimeter)
                potential_balls.append({'position_roi': (int(cx_roi), int(cy_roi)), 'area': area,
                                        'circularity': circularity, 'contour_roi': contour_to_store})
        if not potential_balls: return None, None, None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None, None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file: current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}, T:{current_timestamp:.3f}")
        
        # Return global coordinates and timestamp as they are needed by process_single_frame
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi'), (cx_global, cy_global), current_timestamp


    def _select_best_ball_candidate(self, candidates): # Logic from v11 seems fine
        if not candidates: return None
        if not self.trajectory:
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular: return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])

        last_x_global, last_y_global, _ = self.trajectory[-1]
        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global, cy_global = cx_roi + self.roi_start_x, cy_roi + self.roi_top_y
            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            if distance > self.frame_width * 0.4: ball_info['distance_from_last'] = float('inf')
            consistency_score = 0
            if len(self.trajectory) >= 2:
                prev_x_global, prev_y_global, _ = self.trajectory[-2]
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
        self.is_counting_active = not self.is_counting_active
        if self.is_counting_active:
            self.count_session_id += 1
            self.collected_net_speeds = []
            self.collected_relative_times = []
            self.timing_started_for_session = False
            self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear()
            self.output_generated_for_session = False
            # MODIFIED: Reset new state variables
            self.ball_on_left_of_center = False
            self.last_committed_crossing_time = 0
            self.last_ball_x_global = None 
            print(f"Counting ON (Session #{self.count_session_id}) - Target: {self.max_net_speeds_to_collect} speeds.")
        else:
            print(f"Counting OFF (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_outputs_async()
            self.output_generated_for_session = True

    # MODIFIED: New _record_potential_crossing, replaces old check_center_crossing and _record_potential_crossing
    def _record_potential_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        if not self.is_counting_active:
            self.last_ball_x_global = ball_x_global # Still update for next frame if counting starts
            return

        if self.net_crossing_direction not in ['right_to_left', 'both']:
            self.last_ball_x_global = ball_x_global
            return

        # Cooldown based on the last *committed* crossing
        if current_timestamp - self.last_committed_crossing_time < self.EFFECTIVE_CROSSING_COOLDOWN_S:
            if self.debug_mode: print(f"DEBUG REC: In cooldown. CT: {current_timestamp:.3f}, LastCommitT: {self.last_committed_crossing_time:.3f}")
            self.last_ball_x_global = ball_x_global
            return

        # --- Actual Crossing Detection (Right-to-Left) ---
        # *** MODIFICATION 1: We perform the crossing check BEFORE updating the ball's on_left_of_center state.
        # This prevents a race condition where a fast ball jumps over the center and the margin in one frame.
        crossed_r_to_l_strictly = False
        if self.last_ball_x_global is not None and \
           self.last_ball_x_global >= self.center_x_global and \
           ball_x_global < self.center_x_global and \
           not self.ball_on_left_of_center: # Check against PREVIOUS state
            crossed_r_to_l_strictly = True
            if self.debug_mode:
                print(f"DEBUG REC: Strict R-L Actual Crossing Detected. PrevX: {self.last_ball_x_global:.1f}, CurrX: {ball_x_global:.1f}. Speed: {self.current_ball_speed_kmh:.1f}")

        if crossed_r_to_l_strictly and self.current_ball_speed_kmh > 0.1: # Ensure some minimal speed
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            if self.debug_mode: print(f"DEBUG REC: Added ACTUAL event to buffer. Buffer size: {len(self.event_buffer_center_cross)}")

        # --- REVISED Prediction Logic (Right-to-Left) ---
        # The main condition for prediction is now based on the PREVIOUS frame's position.
        if not crossed_r_to_l_strictly and not self.ball_on_left_of_center and \
           len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 0.1:

            pt1_x, _, pt1_t = self.trajectory[-2] # Previous point
            pt2_x, _, pt2_t = self.trajectory[-1] # Current point (ball_x_global, current_timestamp)

            if pt1_x >= self.center_x_global:
                delta_t_hist = pt2_t - pt1_t
                if delta_t_hist > 0:
                    vx_pixels_per_time_unit = (pt2_x - pt1_x) / delta_t_hist
                    min_vx_for_prediction = - (self.frame_width * 0.005) * (delta_t_hist / (1.0 / (self.display_fps if self.display_fps > 1 else self.target_fps)))

                    if vx_pixels_per_time_unit < min_vx_for_prediction:
                        # *** MODIFICATION 2: Increase lookahead frames to be more aggressive for fast balls.
                        for lookahead_frames in [1, 2, 3]: # Changed from [1, 2]
                            time_to_predict = lookahead_frames / (self.display_fps if self.display_fps > 0 else self.target_fps)
                            predicted_x_at_crossing_time = ball_x_global + vx_pixels_per_time_unit * time_to_predict
                            predicted_timestamp = current_timestamp + time_to_predict

                            if predicted_x_at_crossing_time < self.center_x_global: # Predicted to cross
                                can_add_prediction = True
                                for ev in self.event_buffer_center_cross:
                                    if ev.predicted and abs(ev.timestamp - predicted_timestamp) < (1.0 / (self.display_fps if self.display_fps > 0 else self.target_fps)):
                                        can_add_prediction = False; break

                                if can_add_prediction:
                                    if self.debug_mode: print(f"DEBUG REC: Added PREDICTED event (lookahead {lookahead_frames}f). PredX: {predicted_x_at_crossing_time:.1f}")
                                    event = EventRecord(predicted_x_at_crossing_time, predicted_timestamp, self.current_ball_speed_kmh, predicted=True)
                                    self.event_buffer_center_cross.append(event)
                                break # One prediction per frame is enough
        
        # *** MODIFICATION 1 (Part 2): Update the state AFTER all checks for this frame are done.
        # This ensures the state reflects the beginning of the NEXT frame.
        if ball_x_global < self.center_x_global - self.CENTER_ZONE_WIDTH_PIXELS:
            if not self.ball_on_left_of_center and self.debug_mode: print(f"DEBUG STATE UPDATE: Ball now clearly on left (X={ball_x_global}).")
            self.ball_on_left_of_center = True
        elif ball_x_global > self.center_x_global + self.CENTER_ZONE_WIDTH_PIXELS:
            if self.ball_on_left_of_center and self.debug_mode: print(f"DEBUG STATE UPDATE: Ball returned to right (X={ball_x_global}), resetting left flag.")
            self.ball_on_left_of_center = False

        self.last_ball_x_global = ball_x_global # Update for next frame's comparison
        
    # MODIFIED: New _process_crossing_events
    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_processing_time = time.monotonic()
        if self.use_video_file: current_processing_time = self.frame_counter / self.actual_fps

        processed_event_this_cycle = False
        
        # Create a temporary list to iterate over, allowing modification of the deque
        temp_event_list = sorted(list(self.event_buffer_center_cross), key=lambda e: e.timestamp)
        new_event_buffer = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)

        committed_event_ts = -1

        # --- Stage 1: Try to commit an ACTUAL event ---
        actual_event_to_commit = None
        for event in temp_event_list:
            if event.processed: continue
            if not event.predicted: # It's an actual crossing event
                # Check cooldown against last *committed* event
                if event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                    actual_event_to_commit = event
                    break # Found the best actual event to commit this cycle
        
        if actual_event_to_commit:
            event = actual_event_to_commit
            if len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
                if not self.timing_started_for_session:
                    self.timing_started_for_session = True
                    self.first_ball_crossing_timestamp = event.timestamp
                relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2) if self.timing_started_for_session else 0.0

                self.last_recorded_net_speed_kmh = event.speed_kmh
                self.collected_net_speeds.append(event.speed_kmh)
                self.collected_relative_times.append(relative_time)
                
                self.last_committed_crossing_time = event.timestamp # Update for cooldown
                self.ball_on_left_of_center = True # Confirm ball is now on the left

                if self.debug_mode: print(f"--- COMMITTED ACTUAL Event #{len(self.collected_net_speeds)}: Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. New cooldown starts from {event.timestamp:.3f} ---")
                event.processed = True
                processed_event_this_cycle = True
                committed_event_ts = event.timestamp
            else: # Max speeds collected
                event.processed = True # Mark as processed to remove from buffer
        
        # --- Stage 2: If no ACTUAL event was committed, consider a PREDICTED event ---
        if not processed_event_this_cycle:
            predicted_event_to_commit = None
            for event in temp_event_list:
                if event.processed: continue
                if event.predicted:
                    # Predicted event's time must be reached, and not in cooldown from a previous commit
                    if current_processing_time >= event.timestamp and \
                       event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S :
                        predicted_event_to_commit = event
                        break # Found the best predicted event
            
            if predicted_event_to_commit:
                event = predicted_event_to_commit
                if len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
                    if not self.timing_started_for_session:
                        self.timing_started_for_session = True
                        self.first_ball_crossing_timestamp = event.timestamp
                    relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2) if self.timing_started_for_session else 0.0

                    self.last_recorded_net_speed_kmh = event.speed_kmh
                    self.collected_net_speeds.append(event.speed_kmh)
                    self.collected_relative_times.append(relative_time)
                    
                    self.last_committed_crossing_time = event.timestamp 
                    self.ball_on_left_of_center = True 

                    if self.debug_mode: print(f"--- COMMITTED PREDICTED Event #{len(self.collected_net_speeds)}: Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. New cooldown from {event.timestamp:.3f} ---")
                    event.processed = True
                    processed_event_this_cycle = True
                    committed_event_ts = event.timestamp
                else: # Max speeds collected
                    event.processed = True

        # --- Stage 3: Clean up buffer ---
        # If an event was committed, nullify other events very close to its timestamp
        if committed_event_ts > 0:
            for event_in_list in temp_event_list:
                if not event_in_list.processed and abs(event_in_list.timestamp - committed_event_ts) < self.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0:
                    event_in_list.processed = True
                    if self.debug_mode: print(f"DEBUG PROC: Nullified nearby event (Pred: {event_in_list.predicted}, T: {event_in_list.timestamp:.3f}) due to commit at {committed_event_ts:.3f}")
        
        # Rebuild the main buffer with events that are not processed and not too old
        for event_in_list in temp_event_list:
            if not event_in_list.processed and (current_processing_time - event_in_list.timestamp < 2.0): # Keep events up to 2s old
                new_event_buffer.append(event_in_list)
        self.event_buffer_center_cross = new_event_buffer

        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Collected {self.max_net_speeds_to_collect} net speeds. Generating output.")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION: self.is_counting_active = False
            
    def _calculate_ball_speed(self): # Logic from v11 seems fine
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return
        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob
        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        delta_t = t2 - t1
        if delta_t > 0.0001: # Avoid division by zero/tiny dt
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else: self.current_ball_speed_kmh = speed_kmh
            if self.debug_mode and speed_kmh > 1: # Print only significant raw speeds
                 pass # print(f"Speed: {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else: self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR) # Decay if dt is bad

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g): # Logic from v11 seems fine
        y1_roi = max(0, min(self.roi_height_px, y1_g - self.roi_top_y))
        y2_roi = max(0, min(self.roi_height_px, y2_g - self.roi_top_y))
        y1_roi_rounded = round(y1_roi / 10) * 10
        y2_roi_rounded = round(y2_roi / 10) * 10
        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * avg_px_to_cm_ratio
        return real_distance_cm

    def _generate_outputs_async(self): # Logic from v11 seems fine
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        if not net_speeds: return

        output_dir_path = ""
        base_filename = ""

        # Case 1: Video file input. Use video's path and name prefix.
        if self.use_video_file and self.video_file_path:
            try:
                # 獲取影片所在的目錄
                output_dir_path = os.path.dirname(self.video_file_path)
                # 獲取不含副檔名的影片檔名
                video_filename_stem = os.path.splitext(os.path.basename(self.video_file_path))[0]
                # 根據 "_" 分割檔名，並取前兩部分作為前綴
                parts = video_filename_stem.split('_')
                if len(parts) >= 2:
                    base_filename = f"{parts[0]}_{parts[1]}"
                else:
                    # 如果檔名格式不符，則退回使用完整檔名作為基礎
                    base_filename = video_filename_stem
                print(f"Video mode: Saving files with prefix '{base_filename}' to '{output_dir_path}'")
            except Exception as e:
                print(f"Error parsing video file path. Falling back to default naming. Error: {e}")
                # Fallback to default timestamp naming if path parsing fails
                self.use_video_file = False # Force fallback logic

        # Case 2: Real-time camera input OR fallback from video mode error. Use timestamp.
        if not self.use_video_file or not self.video_file_path:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
            os.makedirs(output_dir_path, exist_ok=True)
            base_filename = f"speed_data_{timestamp_str}"
            print(f"Real-time mode: Saving files to '{output_dir_path}'")

        avg_speed = sum(net_speeds) / len(net_speeds)
        max_speed = max(net_speeds)
        min_speed = min(net_speeds)

        # 組合出最終的檔案路徑
        chart_filename = os.path.join(output_dir_path, f'{base_filename}_chart.png')
        txt_filename = os.path.join(output_dir_path, f'{base_filename}_data.txt')
        csv_filename = os.path.join(output_dir_path, f'{base_filename}_data.csv')

        # --- 以下圖表與檔案寫入邏輯不變 ---
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        for t, s in zip(relative_times, net_speeds): plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Net Crossing Speeds - Session {session_id} - File: {base_filename}', fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12); plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7); plt.legend()
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if len(relative_times) > 1 and max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
        if net_speeds:
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(max(0, min_speed - y_range*0.1), max_speed + y_range*0.1)
        plt.figtext(0.02, 0.02, f"Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(chart_filename, dpi=150); plt.close()

        with open(txt_filename, 'w') as f:
            f.write(f"Net Speeds - Session {session_id} - File: {base_filename}\n"); f.write("---------------------------------------\n")
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write("---------------------------------------\n"); f.write(f"Total Points: {len(net_speeds)}\n")
            f.write(f"Average Speed: {avg_speed:.1f} km/h\n"); f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
            f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")

        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Session ID', 'File Prefix', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): writer.writerow([session_id, base_filename, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([]); writer.writerow(['Statistic', 'Value']); writer.writerow(['Total Points', len(net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"]); writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        print(f"Output files for session {session_id} saved successfully.")

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData): # Logic from v11 seems fine
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
            if frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            cx_global_vis = cx_roi + self.roi_start_x; cy_global_vis = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global_vis, cy_global_vis), 8, BALL_COLOR_BGR, -1)
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"; count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        if frame_data_obj.last_recorded_net_speed_kmh > 0: cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        if frame_data_obj.collected_relative_times: cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if self.debug_mode and frame_data_obj.debug_display_text: cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        return vis_frame

    def _check_timeout_and_reset(self): # Logic from v11 seems fine
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0
            # self.ball_on_left_of_center = False # This state should persist until ball clearly returns to right
            # No need to clear self.last_ball_x_global here, it's for inter-frame diff

    def process_single_frame(self, frame):
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_global_coords, ball_timestamp = None, None

        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi, ball_global_coords, ball_timestamp = self._detect_ball_in_roi(motion_mask_roi)
            if ball_pos_in_roi: # Ball was detected
                self._calculate_ball_speed()
                # MODIFIED: Call new record potential crossing here
                if self.is_counting_active:
                    self._record_potential_crossing(ball_global_coords[0], ball_global_coords[1], ball_timestamp)
            else: # No ball detected by _detect_ball_in_roi (e.g. candidate selection failed)
                self.last_ball_x_global = None # No current ball to be last_ball_x_global
        else: # No motion mask
            self.last_ball_x_global = None


        self._check_timeout_and_reset() # Handles general ball detection timeout
        
        if self.is_counting_active:
            self._process_crossing_events() # Process buffer regardless of current frame detection

        debug_text = None
        if self.debug_mode:
            on_left_text = "Y" if self.ball_on_left_of_center else "N"
            debug_text = f"Traj:{len(self.trajectory)} EvtBuf:{len(self.event_buffer_center_cross)} OnLeft:{on_left_text} LastCommitT:{self.last_committed_crossing_time:.2f}"
        
        frame_data = FrameData(
            frame=frame, roi_sub_frame=roi_sub_frame, ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi, current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps, is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=debug_text, frame_counter=self.frame_counter
        )
        if self.trajectory: frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self): # Overall run loop from v11 seems fine
        print("=== Ping Pong Speed Tracker (v11 MODIFIED) ===")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction} (Focus on Right-to-Left)")
        print(f"Target speeds to collect: {self.max_net_speeds_to_collect}")
        print(f"Effective Crossing Cooldown: {self.EFFECTIVE_CROSSING_COOLDOWN_S}s")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker v11 (MODIFIED)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("Video ended or frame read error.")
                    else: print("Camera error or stream ended.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("End of stream with pending data. Generating output.")
                        self._generate_outputs_async(); self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # ESC
                    self.running = False
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("Quitting with pending data. Generating output.")
                        self._generate_outputs_async(); self.output_generated_for_session = True
                    break
                elif key == ord(' '): self.toggle_counting()
                elif key == ord('d'): self.debug_mode = not self.debug_mode; print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        except KeyboardInterrupt:
            print("Process interrupted by user (Ctrl+C).")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                print("Interrupted with pending data. Generating output.")
                self._generate_outputs_async(); self.output_generated_for_session = True
        finally:
            self.running = False; print("Shutting down...")
            self.reader.stop(); print("Frame reader stopped.")
            self.file_writer_executor.shutdown(wait=True); print("File writer stopped.")
            cv2.destroyAllWindows(); print("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker v11 (MODIFIED)')
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
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

    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False

    tracker = PingPongSpeedTracker(
        video_source=video_source_arg, table_length_cm=args.table_len,
        detection_timeout_s=args.timeout, use_video_file=use_video_file_arg,
        video_file_path=args.video,  # <--- 將影片路徑傳遞給新參數
        target_fps=args.fps, frame_width=args.width, frame_height=args.height,
        debug_mode=args.debug, net_crossing_direction=args.direction,
        max_net_speeds=args.count, near_width_cm=args.near_width, far_width_cm=args.far_width
    )
    tracker.run()

if __name__ == '__main__':
    main()