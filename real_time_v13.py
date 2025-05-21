#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11.1 (ROI Right-to-Left Average Speed)
# Lightweight, optimized, multi-threaded (acquisition & I/O), macOS compatible
# MODIFIED: To maximize recording chances by relaxing transit detection criteria.

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
DEFAULT_TABLE_LENGTH_CM = 94 # Used for fallback perspective calculation

# Detection Parameters
DEFAULT_DETECTION_TIMEOUT = 0.1 # Seconds
DEFAULT_ROI_START_RATIO = 0.4 # Left edge of ROI as fraction of frame width
DEFAULT_ROI_END_RATIO = 0.6   # Right edge of ROI as fraction of frame width
DEFAULT_ROI_BOTTOM_RATIO = 0.55 # Bottom edge of ROI as fraction of frame height
MAX_TRAJECTORY_POINTS = 150

# Data Collection
MAX_TRANSIT_SPEEDS_TO_COLLECT = 30
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output' # Changed folder name

# Perspective Correction
NEAR_SIDE_WIDTH_CM_DEFAULT = 29 # Real width of observed scene at ROI bottom
FAR_SIDE_WIDTH_CM_DEFAULT = 72  # Real width of observed scene at frame top (or ROI top if perspective limited to ROI)

# FMO (Fast Moving Object) Parameters
MAX_PREV_FRAMES_FMO = 15
OPENING_KERNEL_SIZE_FMO = (12, 12)
CLOSING_KERNEL_SIZE_FMO = (25, 25)
THRESHOLD_VALUE_FMO = 8

# Ball Detection Parameters
MIN_BALL_AREA_PX = 10
MAX_BALL_AREA_PX = 10000
MIN_BALL_CIRCULARITY = 0.32
# Speed Calculation
SPEED_SMOOTHING_FACTOR = 0.3 # For instantaneous speed, average transit speed is not smoothed
KMH_CONVERSION_FACTOR = 0.036

# FPS Calculation
FPS_SMOOTHING_FACTOR = 0.4
MAX_FRAME_TIMES_FPS_CALC = 20

# Visualization Parameters
TRAJECTORY_COLOR_BGR = (0, 0, 255)
BALL_COLOR_BGR = (0, 255, 255)
CONTOUR_COLOR_BGR = (255, 0, 0)
ROI_COLOR_BGR = (0, 255, 0)
SPEED_TEXT_COLOR_BGR = (0, 0, 255) # For instantaneous speed
FPS_TEXT_COLOR_BGR = (0, 255, 0)
TRANSIT_SPEED_TEXT_COLOR_BGR = (255, 0, 0) # For recorded average transit speed
FONT_SCALE_VIS = 1
FONT_THICKNESS_VIS = 2
VISUALIZATION_DRAW_INTERVAL = 2 # Draw full visuals every N frames

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 30 # For FrameReader

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
                 display_fps=0, is_counting_active=False, collected_transit_speeds=None,
                 last_recorded_transit_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0):
        self.frame = frame
        self.roi_sub_frame = roi_sub_frame
        self.ball_position_in_roi = ball_position_in_roi
        self.ball_contour_in_roi = ball_contour_in_roi
        self.current_ball_speed_kmh = current_ball_speed_kmh # Instantaneous
        self.display_fps = display_fps
        self.is_counting_active = is_counting_active
        self.collected_transit_speeds = collected_transit_speeds if collected_transit_speeds is not None else []
        self.last_recorded_transit_speed_kmh = last_recorded_transit_speed_kmh # Avg speed of last recorded transit
        self.collected_relative_times = collected_relative_times if collected_relative_times is not None else []
        self.debug_display_text = debug_display_text
        self.frame_counter = frame_counter
        self.trajectory_points_global = []

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
                 target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=DEBUG_MODE_DEFAULT,
                 max_transit_speeds=MAX_TRANSIT_SPEEDS_TO_COLLECT,
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
        self.current_ball_speed_kmh = 0 # Instantaneous speed
        self.last_detection_timestamp = time.time()

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2 

        self.max_transit_speeds_to_collect = max_transit_speeds
        self.collected_transit_speeds = []
        self.collected_relative_times = []
        self.last_recorded_transit_speed_kmh = 0 
        self.output_generated_for_session = False
        
        self.is_counting_active = False
        self.count_session_id = 0
        self.timing_started_for_session = False
        self.first_ball_event_timestamp_in_session = None 
        
        self.is_tracking_r_to_l_transit = False
        self.current_r_to_l_transit_data = [] 
        
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # MODIFICATION: Parameters for relaxed transit detection
        self.X_MOVEMENT_PIXEL_THRESHOLD_TRANSIT = 3  # Pixel difference to confirm horizontal movement
        self.transit_ball_lost_frames_count = 0      # Counter for consecutive frames ball is lost during transit
        self.MAX_TRANSIT_BALL_LOST_FRAMES_TOLERANCE = 3 # Max consecutive lost frames tolerated during transit

        self._precalculate_overlay()
        self._create_perspective_lookup_table()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.rectangle(self.static_overlay, (self.roi_start_x, self.roi_top_y), 
                      (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
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
        
        roi_width_px_for_ratio_calc = self.roi_width_px 
        if current_width_cm > 0 and roi_width_px_for_ratio_calc > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px_for_ratio_calc
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

    def _detect_ball_in_roi(self, motion_mask_roi):
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
                        contour_to_store = cnt
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: circularity = 4 * math.pi * area / (perimeter * perimeter)
                potential_balls.append({'position_roi': (int(cx_roi), int(cy_roi)), 'area': area,
                                        'circularity': circularity, 'contour_roi': contour_to_store})
        if not potential_balls: return None, None
        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file: current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic() 
        self.trajectory.append((cx_global, cy_global, current_timestamp)) 
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}, T:{current_timestamp:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')

    def _select_best_ball_candidate(self, candidates):
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
            if distance > self.frame_width * 0.2: ball_info['distance_from_last'] = float('inf')
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
            ball_info['score'] = (0.4 / (1.0 + ball_info['distance_from_last'])) + \
                                 (0.4 * ball_info['consistency']) + \
                                 (0.2 * ball_info['circularity'])
        return max(candidates, key=lambda b: b['score'])

    def toggle_counting(self):
        self.is_counting_active = not self.is_counting_active
        if self.is_counting_active:
            self.count_session_id += 1
            self.collected_transit_speeds = []
            self.collected_relative_times = []
            self.timing_started_for_session = False
            self.first_ball_event_timestamp_in_session = None
            self.output_generated_for_session = False
            self.is_tracking_r_to_l_transit = False
            self.current_r_to_l_transit_data = []
            self.transit_ball_lost_frames_count = 0 # MODIFICATION: Reset on new count session
            print(f"Counting ON (Session #{self.count_session_id}) - Target: {self.max_transit_speeds_to_collect} transit speeds.")
        else:
            if self.is_tracking_r_to_l_transit:
                 if self.debug_mode: print("DEBUG: Counting toggled OFF during active transit. Finalizing...")
                 last_known_time = self.current_r_to_l_transit_data[-1]['t'] if self.current_r_to_l_transit_data else \
                                   (self.trajectory[-1][2] if self.trajectory else \
                                   (self.frame_counter / self.actual_fps if self.use_video_file else time.monotonic()))
                 self._finalize_r_to_l_transit(last_known_time, "counting_toggled_off")

            print(f"Counting OFF (Session #{self.count_session_id}).")
            if self.collected_transit_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_transit_speeds)} transit speeds. Generating output...")
                self._generate_outputs_async()
            self.output_generated_for_session = True

    def _update_r_to_l_transit_tracking(self, cx_global, cy_global, current_timestamp, current_speed_kmh):
        if not self.is_counting_active:
            if self.is_tracking_r_to_l_transit: 
                self._finalize_r_to_l_transit(current_timestamp, "counting_stopped_unexpectedly")
            return

        if len(self.trajectory) < 2: 
            if self.is_tracking_r_to_l_transit: 
                self._finalize_r_to_l_transit(current_timestamp, "ball_trajectory_lost")
            return

        prev_x_global, _, _ = self.trajectory[-2] 
        
        # MODIFICATION: Use relaxed pixel threshold for movement detection
        is_moving_left = (cx_global < prev_x_global - self.X_MOVEMENT_PIXEL_THRESHOLD_TRANSIT) 
        is_moving_right = (cx_global > prev_x_global + self.X_MOVEMENT_PIXEL_THRESHOLD_TRANSIT)

        if is_moving_left:
            if not self.is_tracking_r_to_l_transit:
                if self.debug_mode: print(f"DEBUG: R-to-L transit STARTED at ({cx_global},{cy_global}) T:{current_timestamp:.3f}")
                self.is_tracking_r_to_l_transit = True
                self.current_r_to_l_transit_data = [] 
                self.transit_ball_lost_frames_count = 0 # MODIFICATION: Reset lost frames count on new transit start

            self.current_r_to_l_transit_data.append({'x': cx_global, 'y': cy_global, 't': current_timestamp})
            
            if cx_global < (self.roi_start_x + self.roi_width_px * 0.05):
                if self.debug_mode: print(f"DEBUG: R-to-L transit MET EXIT LEFT condition at x={cx_global}")
                self._finalize_r_to_l_transit(current_timestamp, "exited_left_roi_boundary")
        
        elif is_moving_right:
            if self.is_tracking_r_to_l_transit:
                if self.debug_mode: print(f"DEBUG: R-to-L transit ENDED by direction change to RIGHT at x={cx_global}")
                self._finalize_r_to_l_transit(current_timestamp, "direction_change_to_right")
        
        else: 
            if self.is_tracking_r_to_l_transit:
                self.current_r_to_l_transit_data.append({'x': cx_global, 'y': cy_global, 't': current_timestamp})


    def _finalize_r_to_l_transit(self, end_timestamp, reason):
        if not self.is_tracking_r_to_l_transit or not self.current_r_to_l_transit_data or len(self.current_r_to_l_transit_data) < 2:
            if self.debug_mode and self.is_tracking_r_to_l_transit: 
                print(f"DEBUG: Finalize '{reason}' called, but insufficient transit data ({len(self.current_r_to_l_transit_data)} pts). Resetting.")
            self.is_tracking_r_to_l_transit = False
            self.current_r_to_l_transit_data = []
            self.transit_ball_lost_frames_count = 0 # MODIFICATION: Reset lost frames count
            return

        if self.debug_mode: print(f"DEBUG: Finalizing R-to-L transit ({reason}). Points: {len(self.current_r_to_l_transit_data)}")

        start_point_data = self.current_r_to_l_transit_data[0]
        end_point_data = self.current_r_to_l_transit_data[-1] 

        total_dist_cm = self._calculate_real_distance_cm_global(
            start_point_data['x'], start_point_data['y'],
            end_point_data['x'], end_point_data['y']
        )
        
        delta_t = end_point_data['t'] - start_point_data['t']

        avg_speed_kmh = 0
        if delta_t > 0.001: 
            avg_speed_cm_per_s = total_dist_cm / delta_t
            avg_speed_kmh = avg_speed_cm_per_s * KMH_CONVERSION_FACTOR
        else:
            if self.debug_mode: print(f"DEBUG: Transit duration too short ({delta_t:.4f}s) or invalid. Speed set to 0.")

        self.is_tracking_r_to_l_transit = False
        self.current_r_to_l_transit_data = []
        self.transit_ball_lost_frames_count = 0 # MODIFICATION: Reset lost frames count
        
        if not self.is_counting_active: 
            if self.debug_mode: print("DEBUG: Counting is OFF during finalize. Speed not recorded.")
            return

        if avg_speed_kmh > 0.1: 
            if len(self.collected_transit_speeds) >= self.max_transit_speeds_to_collect:
                if not self.output_generated_for_session: 
                    print(f"Max transit speeds ({self.max_transit_speeds_to_collect}) already collected. Further speeds ignored for this session.")
                    self._generate_outputs_async()
                    self.output_generated_for_session = True
                    if AUTO_STOP_AFTER_COLLECTION: self.is_counting_active = False
                return

            if not self.timing_started_for_session:
                self.timing_started_for_session = True
                self.first_ball_event_timestamp_in_session = start_point_data['t']
                relative_time = 0.0
            else:
                relative_time = round(start_point_data['t'] - self.first_ball_event_timestamp_in_session, 2)

            self.last_recorded_transit_speed_kmh = avg_speed_kmh
            self.collected_transit_speeds.append(avg_speed_kmh)
            self.collected_relative_times.append(relative_time)
            
            print(f"ROI R-L Transit #{len(self.collected_transit_speeds)}: Avg Speed {avg_speed_kmh:.1f} km/h @ Rel.T {relative_time:.2f}s (Dur: {delta_t:.2f}s, Dist: {total_dist_cm:.1f}cm, Reason: {reason})")

            if len(self.collected_transit_speeds) >= self.max_transit_speeds_to_collect and not self.output_generated_for_session:
                print(f"Collected {self.max_transit_speeds_to_collect} transit speeds. Generating output.")
                self._generate_outputs_async()
                self.output_generated_for_session = True
                if AUTO_STOP_AFTER_COLLECTION: self.is_counting_active = False
        elif self.debug_mode:
             print(f"DEBUG: Calculated transit speed {avg_speed_kmh:.1f} km/h is too low. Discarding.")


    def _calculate_ball_speed(self): 
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return

        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob

        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        delta_t = t2 - t1
        if delta_t > 0.0001: 
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else: self.current_ball_speed_kmh = speed_kmh
            if self.debug_mode and speed_kmh > 1: 
                 pass 
        else: self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)


    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        y1_roi = max(0, min(self.roi_height_px, y1_g - self.roi_top_y)) 
        y2_roi = max(0, min(self.roi_height_px, y2_g - self.roi_top_y))
        
        y1_roi_rounded = round(y1_roi / 10) * 10
        y2_roi_rounded = round(y2_roi / 10) * 10
        
        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        avg_cm_per_pixel_ratio = (ratio1 + ratio2) / 2.0 
        
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * avg_cm_per_pixel_ratio
        return real_distance_cm

    def _generate_outputs_async(self):
        if not self.collected_transit_speeds:
            print("No transit speed data to generate output.")
            return
        speeds_copy = list(self.collected_transit_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, transit_speeds, relative_times, session_id):
        if not transit_speeds: return
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
        os.makedirs(output_dir_path, exist_ok=True)

        avg_speed = sum(transit_speeds) / len(transit_speeds)
        max_speed = max(transit_speeds)
        min_speed = min(transit_speeds)

        chart_filename = f'{output_dir_path}/transit_speed_chart_{timestamp_str}.png'
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, transit_speeds, 'o-', linewidth=2, markersize=6, label='Avg Transit Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Overall Avg: {avg_speed:.1f} km/h')
        for t, s in zip(relative_times, transit_speeds):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Avg R-L Transit Speeds in ROI - Session {session_id} - {timestamp_str}', fontsize=16)
        plt.xlabel('Relative Time of Transit Start (s)', fontsize=12)
        plt.ylabel('Average Transit Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if len(relative_times) > 1 and max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
        if transit_speeds:
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(max(0, min_speed - y_range*0.1), max_speed + y_range*0.1) 
        plt.figtext(0.02, 0.02, f"Count: {len(transit_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_filename, dpi=150)
        plt.close()

        txt_filename = f'{output_dir_path}/transit_speed_data_{timestamp_str}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Avg R-L Transit Speeds - Session {session_id} - {timestamp_str}\n")
            f.write("---------------------------------------\n")
            for i, (t, s) in enumerate(zip(relative_times, transit_speeds)):
                f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write("---------------------------------------\n")
            f.write(f"Total Transits Recorded: {len(transit_speeds)}\n")
            f.write(f"Overall Average Speed: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum Avg Transit Speed: {max_speed:.1f} km/h\n")
            f.write(f"Minimum Avg Transit Speed: {min_speed:.1f} km/h\n")

        csv_filename = f'{output_dir_path}/transit_speed_data_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Session ID', 'Timestamp File', 'Transit Number', 'Relative Start Time (s)', 'Avg Transit Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, transit_speeds)):
                writer.writerow([session_id, timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Total Transits Recorded', len(transit_speeds)])
            writer.writerow(['Overall Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Avg Transit Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Avg Transit Speed (km/h)', f"{min_speed:.1f}"])
        
        print(f"Output files for session {session_id} saved to {output_dir_path}")


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
            if frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)

        cv2.putText(vis_frame, f"Inst. Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        if frame_data_obj.last_recorded_transit_speed_kmh > 0:
            cv2.putText(vis_frame, f"Last Avg Transit: {frame_data_obj.last_recorded_transit_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, TRANSIT_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_transit_speeds)}/{self.max_transit_speeds_to_collect}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, TRANSIT_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        if frame_data_obj.collected_relative_times:
            cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, TRANSIT_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)

        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if self.debug_mode and frame_data_obj.debug_display_text:
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame

    def _check_timeout_and_reset(self): 
        current_time = time.monotonic() 
        if current_time - self.last_detection_timestamp > self.detection_timeout_s:
            if self.is_tracking_r_to_l_transit:
                if self.debug_mode: print(f"DEBUG: R-L transit ENDING by detection timeout (ball lost for {self.detection_timeout_s}s).")
                
                last_point_event_time = current_time 
                if self.use_video_file: 
                    last_point_event_time = self.frame_counter / self.actual_fps
                
                if self.current_r_to_l_transit_data: 
                    last_point_event_time = self.current_r_to_l_transit_data[-1]['t']
                elif self.trajectory: 
                     last_point_event_time = self.trajectory[-1][2]

                self._finalize_r_to_l_transit(last_point_event_time, "detection_timeout")
            
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0 

    def process_single_frame(self, frame):
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame) 
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_detected_this_frame = False

        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi) 
            
            if ball_pos_in_roi:
                ball_detected_this_frame = True
                # MODIFICATION: If ball detected during an active transit, reset lost frames counter
                if self.is_tracking_r_to_l_transit:
                    self.transit_ball_lost_frames_count = 0
                
                self._calculate_ball_speed() 

                cx_global = ball_pos_in_roi[0] + self.roi_start_x
                cy_global = ball_pos_in_roi[1] + self.roi_top_y
                current_event_timestamp = self.trajectory[-1][2] 
                
                self._update_r_to_l_transit_tracking(cx_global, cy_global, current_event_timestamp, self.current_ball_speed_kmh)
        
        # MODIFICATION: Handle ball not detected with tolerance
        if not ball_detected_this_frame:
            if self.is_tracking_r_to_l_transit:
                self.transit_ball_lost_frames_count += 1
                if self.debug_mode:
                    print(f"DEBUG: Ball not detected during transit. Lost count: {self.transit_ball_lost_frames_count}/{self.MAX_TRANSIT_BALL_LOST_FRAMES_TOLERANCE}")
                
                if self.transit_ball_lost_frames_count >= self.MAX_TRANSIT_BALL_LOST_FRAMES_TOLERANCE:
                    if self.debug_mode: 
                        print(f"DEBUG: Ball lost for {self.transit_ball_lost_frames_count} frames. Finalizing transit due to tolerance.")
                    
                    last_known_event_time = time.monotonic() 
                    if self.use_video_file:
                        last_known_event_time = self.frame_counter / self.actual_fps
                    if self.current_r_to_l_transit_data: 
                        last_known_event_time = self.current_r_to_l_transit_data[-1]['t']
                    elif self.trajectory: 
                        last_known_event_time = self.trajectory[-1][2]
                    
                    self._finalize_r_to_l_transit(last_known_event_time, f"ball_lost_for_{self.transit_ball_lost_frames_count}_frames")
                # else: Ball lost but within tolerance, do not finalize yet.
        
        self._check_timeout_and_reset() 

        debug_text = None
        if self.debug_mode:
            transit_status = "Y" if self.is_tracking_r_to_l_transit else "N"
            pts_count = len(self.current_r_to_l_transit_data) if self.is_tracking_r_to_l_transit else 0
            lost_cnt = self.transit_ball_lost_frames_count if self.is_tracking_r_to_l_transit else 0
            debug_text = f"Traj: {len(self.trajectory)}, Transit: {transit_status} ({pts_count}pts), Lost: {lost_cnt}"
        
        frame_data = FrameData(
            frame=frame, 
            roi_sub_frame=roi_sub_frame, 
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi,
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_transit_speeds=list(self.collected_transit_speeds),
            last_recorded_transit_speed_kmh=self.last_recorded_transit_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=debug_text,
            frame_counter=self.frame_counter
        )
        if self.trajectory:
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker (ROI R-L Transit Average Speed) ===")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm at ROI horizontal extents.")
        print(f"ROI (x1,y1)-(x2,y2): ({self.roi_start_x},{self.roi_top_y})-({self.roi_end_x},{self.roi_bottom_y})")
        print(f"Target transit speeds to collect: {self.max_transit_speeds_to_collect}")
        # MODIFICATION: Print new tolerance parameters
        print(f"Transit Movement Threshold: {self.X_MOVEMENT_PIXEL_THRESHOLD_TRANSIT}px, Lost Frame Tolerance: {self.MAX_TRANSIT_BALL_LOST_FRAMES_TOLERANCE} frames")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker v11.1 - ROI Transit Speed (MODIFIED for Max Records)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("Video ended or frame read error.")
                    else: print("Camera error or stream ended.")
                    if self.is_counting_active and self.collected_transit_speeds and not self.output_generated_for_session:
                        print("End of stream with pending data. Generating output.")
                        self._generate_outputs_async() 
                        self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # ESC
                    self.running = False
                    if self.is_counting_active and self.collected_transit_speeds and not self.output_generated_for_session:
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
            if self.is_counting_active and self.collected_transit_speeds and not self.output_generated_for_session:
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
    global DEFAULT_ROI_START_RATIO, DEFAULT_ROI_END_RATIO, DEFAULT_ROI_BOTTOM_RATIO
    
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker v11.1 (ROI R-L Avg Speed)')
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='Frame height.')
    parser.add_argument('--table_len', type=float, default=DEFAULT_TABLE_LENGTH_CM, help='Table length (cm) for nominal px/cm (fallback).')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='Ball detection timeout (s).')
    
    parser.add_argument('--count', type=int, default=MAX_TRANSIT_SPEEDS_TO_COLLECT, help='Number of R-L transit speeds to collect per session.')
    
    parser.add_argument('--near_width', type=float, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at near side (ROI bottom).')
    parser.add_argument('--far_width', type=float, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at far side (frame top or ROI top).')
    
    parser.add_argument('--roi_start', type=float, default=DEFAULT_ROI_START_RATIO, help='ROI start X ratio (0.0-1.0).')
    parser.add_argument('--roi_end', type=float, default=DEFAULT_ROI_END_RATIO, help='ROI end X ratio (0.0-1.0).')
    parser.add_argument('--roi_bottom', type=float, default=DEFAULT_ROI_BOTTOM_RATIO, help='ROI bottom Y ratio (0.0-1.0).')

    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='Enable debug printouts.')
    args = parser.parse_args()

    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False

    DEFAULT_ROI_START_RATIO = args.roi_start
    DEFAULT_ROI_END_RATIO = args.roi_end
    DEFAULT_ROI_BOTTOM_RATIO = args.roi_bottom


    tracker = PingPongSpeedTracker(
        video_source=video_source_arg,
        table_length_cm=args.table_len,
        detection_timeout_s=args.timeout,
        use_video_file=use_video_file_arg,
        target_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        debug_mode=args.debug,
        max_transit_speeds=args.count,
        near_width_cm=args.near_width,
        far_width_cm=args.far_width
    )
    tracker.run()

if __name__ == '__main__':
    main()