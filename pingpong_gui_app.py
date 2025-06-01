#!/usr/bin/env python3
# 乒乓球速度追蹤系統 - GUI版本 (修正版 + 設定面板)
# 修正停止計數時畫面變空白的問題，並簡化session管理
# 新增：參數設定面板與輸出路徑指定

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import csv
import threading
import queue
import concurrent.futures
from PIL import Image, ImageTk
import json

# —— 全局參數配置 (These serve as initial defaults for the GUI) ——
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 60
DEFAULT_FRAME_WIDTH = 1280 # Note: Frame width/height from camera might override this for camera source
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_TABLE_LENGTH_CM = 94.0 # Made float for GUI

DEFAULT_DETECTION_TIMEOUT = 0.3
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.55
MAX_TRAJECTORY_POINTS = 120

MAX_NET_SPEEDS_TO_COLLECT = 30
NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output' # This will be overridden by GUI setting

NEAR_SIDE_WIDTH_CM_DEFAULT = 29.0 # Made float
FAR_SIDE_WIDTH_CM_DEFAULT = 72.0  # Made float

MAX_PREV_FRAMES_FMO = 10
OPENING_KERNEL_SIZE_FMO = (12, 12)
CLOSING_KERNEL_SIZE_FMO = (25, 25)
THRESHOLD_VALUE_FMO = 9

MIN_BALL_AREA_PX = 10
MAX_BALL_AREA_PX = 10000
MIN_BALL_CIRCULARITY = 0.32

SPEED_SMOOTHING_FACTOR = 0.3
KMH_CONVERSION_FACTOR = 0.036

FPS_SMOOTHING_FACTOR = 0.4
MAX_FRAME_TIMES_FPS_CALC = 20

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

FRAME_QUEUE_SIZE = 30
EVENT_BUFFER_SIZE_CENTER_CROSS = 200

DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 10)
except AttributeError:
    cv2.setNumThreads(10)

# —— 核心數據結構 (與原版完全一致) ——
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
        self.processed = False

class FrameReader:
    """Reads frames from camera or video file in a separate thread."""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        # Forcing AVFoundation for macOS, an alternative for Windows could be cv2.CAP_DSHOW
        # Or remove it if not needed / causing issues on other OS
        try:
            if os.name == 'posix': # Simple check for macOS/Linux
                 self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
            else: # Windows or other
                 self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW) # Try DSHOW for Windows
            if not self.cap.isOpened(): # Fallback if specific backend fails
                print("Primary VideoCapture backend failed, trying default.")
                self.cap = cv2.VideoCapture(self.video_source)

        except Exception:
            print("VideoCapture backend specific attempt failed, trying default.")
            self.cap = cv2.VideoCapture(self.video_source)


        self._configure_capture(frame_width, frame_height)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.frame_width == 0 or self.frame_height == 0 and not self.use_video_file:
            print(f"Warning: Camera reported 0x0 frame size. Using default {frame_width}x{frame_height}")
            self.frame_width = frame_width
            self.frame_height = frame_height


        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000): # Camera FPS often unreliable
             self.actual_fps = self.target_fps # Use target_fps as a more reliable estimate for cameras

    def _configure_capture(self, frame_width, frame_height):
        if not self.use_video_file: # Only set for cameras
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
                    self.frame_queue.put((False, None)) # Signal end
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(1.0 / (self.target_fps * 2)) # Sleep if queue is full

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0) # Timeout to prevent blocking indefinitely
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0) # Wait for thread to finish
        if self.cap.isOpened():
            self.cap.release()

    def get_properties(self):
        return self.actual_fps, self.frame_width, self.frame_height

class Core_PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, 
                 use_video_file=False,
                 target_fps=DEFAULT_TARGET_FPS, 
                 # Default frame_width/height are for camera init, actual comes from FrameReader
                 frame_width_hint=DEFAULT_FRAME_WIDTH, 
                 frame_height_hint=DEFAULT_FRAME_HEIGHT,
                 table_length_cm=DEFAULT_TABLE_LENGTH_CM,
                 detection_timeout_s=DEFAULT_DETECTION_TIMEOUT, 
                 roi_start_ratio=DEFAULT_ROI_START_RATIO,
                 roi_end_ratio=DEFAULT_ROI_END_RATIO,
                 roi_bottom_ratio=DEFAULT_ROI_BOTTOM_RATIO,
                 max_net_speeds=MAX_NET_SPEEDS_TO_COLLECT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT,
                 min_ball_area=MIN_BALL_AREA_PX,
                 max_ball_area=MAX_BALL_AREA_PX,
                 min_ball_circularity=MIN_BALL_CIRCULARITY,
                 debug_mode=DEBUG_MODE_DEFAULT,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT
                 ):
        
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.target_fps = target_fps # Target FPS for camera, also used in some calcs if actual_fps is weird

        self.reader = FrameReader(video_source, self.target_fps, self.use_video_file, frame_width_hint, frame_height_hint)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps # Initial display FPS

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        # Nominal pixels_per_cm, perspective correction will refine this
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm if self.table_length_cm > 0 else 1

        # Use passed-in ROI ratios
        self.roi_start_x = int(self.frame_width * roi_start_ratio)
        self.roi_end_x = int(self.frame_width * roi_end_ratio)
        self.roi_top_y = 0 # Assuming ROI starts from the top of the frame
        self.roi_bottom_y = int(self.frame_height * roi_bottom_ratio)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y

        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time() # Use time.monotonic for intervals

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.monotonic() # Use monotonic for FPS calc
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
        
        # Perspective correction parameters
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        self.running = False # This seems to be for the FrameReader, tracker itself doesn't have a run loop
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self.ball_on_left_of_center = False 
        self.last_committed_crossing_time = 0 
        self.EFFECTIVE_CROSSING_COOLDOWN_S = 0.3 # s
        self.CENTER_ZONE_WIDTH_PIXELS = self.frame_width * 0.05 # 5% of frame width as center dead zone
        
        self.current_player_id = ""
        self.current_test_mode = ""
        self.output_folder_for_current_session = OUTPUT_DATA_FOLDER # Will be set by GUI
        self.crossing_callback = None

        # Ball detection parameters from arguments
        self.min_ball_area_px = min_ball_area
        self.max_ball_area_px = max_ball_area
        self.min_ball_circularity = min_ball_circularity
        
        self._precalculate_overlay()
        self._create_perspective_lookup_table()
        print(f"Tracker initialized. Frame: {self.frame_width}x{self.frame_height} @ Effective FPS: {self.actual_fps:.2f}. ROI: X({self.roi_start_x}-{self.roi_end_x}), Y({self.roi_top_y}-{self.roi_bottom_y})")


    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        # Ensure roi_height_px is positive before creating range
        if self.roi_height_px <=0:
            print("Warning: ROI height is zero or negative. Perspective lookup table will be empty.")
            # Provide a fallback default ratio if table is empty
            self.perspective_lookup_px_to_cm[0] = self.table_length_cm / self.frame_width if self.frame_width > 0 else 1
            return

        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10): # Step by 10px for LUT
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y) 
        if self.roi_bottom_y == 0:  # Avoid division by zero if ROI bottom is at top of frame
            relative_y = 0.5 # Assume mid-point if ROI height is zero (should not happen with valid ROI)
        else: 
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)

        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        
        roi_width_px = self.roi_end_x - self.roi_start_x
        if roi_width_px <= 0: # Avoid division by zero if ROI width is zero
            return self.pixels_per_cm_nominal # Fallback to nominal if ROI width invalid

        if current_width_cm > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px
        else: # Fallback if calculated width is zero (e.g. if far and near are zero)
            pixel_to_cm_ratio = self.pixels_per_cm_nominal
        return pixel_to_cm_ratio

    def _update_display_fps(self):
        if self.use_video_file:
            self.display_fps = self.actual_fps # For video files, FPS is fixed
            return
        now = time.monotonic()
        self.frame_timestamps_for_fps.append(now)
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 0:
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                # Apply smoothing
                self.display_fps = (1 - FPS_SMOOTHING_FACTOR) * self.display_fps + FPS_SMOOTHING_FACTOR * measured_fps
            # else: display_fps remains unchanged
    
    def _preprocess_frame(self, frame):
        # Ensure ROI coordinates are within frame bounds
        eff_roi_top_y = max(0, self.roi_top_y)
        eff_roi_bottom_y = min(self.frame_height, self.roi_bottom_y)
        eff_roi_start_x = max(0, self.roi_start_x)
        eff_roi_end_x = min(self.frame_width, self.roi_end_x)

        if eff_roi_bottom_y <= eff_roi_top_y or eff_roi_end_x <= eff_roi_start_x:
            # print("Warning: Invalid ROI dimensions after clamping. Using full frame for preprocessing.")
            # Fallback to a small default ROI or handle error appropriately
            # For now, let's create a dummy sub_frame if ROI is invalid to prevent crash
            # A better solution might be to flag an error or use full frame
            roi_sub_frame = frame[0:1, 0:1] # Dummy small ROI
        else:
            roi_sub_frame = frame[eff_roi_top_y:eff_roi_bottom_y, eff_roi_start_x:eff_roi_end_x]

        if roi_sub_frame.size == 0: # Check if sub_frame is empty
             # This can happen if ROI is outside frame; return a dummy gray_roi
            gray_roi_blurred = np.zeros((1,1), dtype=np.uint8)
        else:
            gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY)
            gray_roi_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        self.prev_frames_gray_roi.append(gray_roi_blurred)
        return roi_sub_frame, gray_roi_blurred # gray_roi_blurred is now always defined

    def _detect_fmo(self):
        if len(self.prev_frames_gray_roi) < 3: 
            return None
        
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        
        # Ensure frames have compatible shapes for absdiff
        if not (f1.shape == f2.shape == f3.shape and f1.size > 0):
            # print("FMO detection: Frame shape mismatch or empty frame in buffer.")
            return None # Cannot perform diff

        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        
        try: # Otsu can fail on uniform images
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error: # Fallback to fixed threshold if Otsu fails
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations only if kernel size is positive
        if OPENING_KERNEL_SIZE_FMO[0] > 0 and OPENING_KERNEL_SIZE_FMO[1] > 0:
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else: 
            opened_mask = thresh_mask # Skip opening if kernel is (0,0)
        
        if CLOSING_KERNEL_SIZE_FMO[0] > 0 and CLOSING_KERNEL_SIZE_FMO[1] > 0:
            closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        else:
            closed_mask = opened_mask # Skip closing

        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        # Use instance variables for ball properties
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        potential_balls = []
        for i in range(1, num_labels): # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_ball_area_px < area < self.max_ball_area_px:
                x_roi, y_roi = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w_roi, h_roi = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                
                circularity = 0
                contour_to_store = None # Initialize
                if max(w_roi, h_roi) > 0: # Ensure width or height is positive
                    # Create a mask for the current component to find its contour
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0] # Assume the largest contour is the component itself
                        contour_to_store = cnt
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: # Avoid division by zero
                            circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                # Check circularity against the instance variable
                if circularity >= self.min_ball_circularity:
                    potential_balls.append({'position_roi': (int(cx_roi), int(cy_roi)), 'area': area,
                                            'circularity': circularity, 'contour_roi': contour_to_store})
        
        if not potential_balls: 
            return None, None, None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: 
            return None, None, None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        # Convert ROI ball position to global frame position
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic() # Always use monotonic for time differences
        if self.use_video_file: # If video file, timestamp is based on frame number and FPS
            current_timestamp = self.frame_counter / self.actual_fps if self.actual_fps > 0 else self.frame_counter / self.target_fps
        
        self.last_detection_timestamp = time.monotonic() # Update last detection time
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}, T:{current_timestamp:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi'), (cx_global, cy_global), current_timestamp


    def _select_best_ball_candidate(self, candidates):
        if not candidates: 
            return None
        
        # If no trajectory yet, prefer highly circular candidates or largest area
        if not self.trajectory:
            # Filter by instance circularity threshold
            highly_circular = [b for b in candidates if b['circularity'] > self.min_ball_circularity]
            if highly_circular: 
                return max(highly_circular, key=lambda b: b['circularity']) # Choose most circular
            return max(candidates, key=lambda b: b['area']) # Or largest by area

        # If trajectory exists, use predictive scoring
        last_x_global, last_y_global, _ = self.trajectory[-1]
        
        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global, cy_global = cx_roi + self.roi_start_x, cy_roi + self.roi_top_y
            
            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            
            # Penalize candidates too far from the last known position
            if distance > self.frame_width * 0.2: # e.g., >20% of frame width
                ball_info['distance_from_last'] = float('inf') # Effectively disqualifies if too far

            # Calculate motion consistency score
            consistency_score = 0
            if len(self.trajectory) >= 2: # Need at least two previous points for direction vector
                prev_x_global, prev_y_global, _ = self.trajectory[-2]
                vec_hist_dx, vec_hist_dy = last_x_global - prev_x_global, last_y_global - prev_y_global
                vec_curr_dx, vec_curr_dy = cx_global - last_x_global, cy_global - last_y_global
                
                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist = math.sqrt(vec_hist_dx**2 + vec_hist_dy**2)
                mag_curr = math.sqrt(vec_curr_dx**2 + vec_curr_dy**2)
                
                if mag_hist > 0 and mag_curr > 0: 
                    cosine_similarity = dot_product / (mag_hist * mag_curr)
                    consistency_score = max(0, cosine_similarity) # Ensure non-negative
                # else: consistency_score remains 0
            ball_info['consistency'] = consistency_score
            
        # Scoring: combination of proximity, consistency, and circularity
        for ball_info in candidates:
            # Weights can be tuned
            score = (0.4 / (1.0 + ball_info['distance_from_last'])) + \
                    (0.4 * ball_info['consistency']) + \
                    (0.2 * ball_info['circularity']) # Use instance circularity
            ball_info['score'] = score
            
        return max(candidates, key=lambda b: b['score'])

    def _record_potential_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        """與 real_time_v14.py 完全一致的穿越記錄邏輯"""
        if not self.is_counting_active:
            self.last_ball_x_global = ball_x_global
            return

        # For this example, we assume 'right_to_left' or 'both' is active
        # if self.net_crossing_direction not in ['right_to_left', 'both']:
        #     self.last_ball_x_global = ball_x_global
        #     return

        # Cooldown based on the last *committed* crossing
        if current_timestamp - self.last_committed_crossing_time < self.EFFECTIVE_CROSSING_COOLDOWN_S:
            if self.debug_mode: 
                print(f"DEBUG REC: In cooldown. CT: {current_timestamp:.3f}, LastCommitT: {self.last_committed_crossing_time:.3f}")
            self.last_ball_x_global = ball_x_global
            return

        # Update ball's general position relative to center
        is_currently_left = ball_x_global < self.center_x_global - self.CENTER_ZONE_WIDTH_PIXELS
        is_currently_right = ball_x_global > self.center_x_global + self.CENTER_ZONE_WIDTH_PIXELS

        # --- Actual Crossing Detection (Right-to-Left, assuming this direction for simplicity) ---
        crossed_r_to_l_strictly = False
        if self.last_ball_x_global is not None and \
           self.last_ball_x_global >= self.center_x_global and \
           ball_x_global < self.center_x_global and \
           not self.ball_on_left_of_center: # Ball was previously not considered on the left
            crossed_r_to_l_strictly = True
            if self.debug_mode:
                print(f"DEBUG REC: Strict R-L Actual Crossing Detected. PrevX: {self.last_ball_x_global:.1f}, CurrX: {ball_x_global:.1f}. Speed: {self.current_ball_speed_kmh:.1f}")

        if crossed_r_to_l_strictly and self.current_ball_speed_kmh > 0.1: # Minimal speed to be considered a valid crossing
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            if self.debug_mode: 
                print(f"DEBUG REC: Added ACTUAL event to buffer. Buffer size: {len(self.event_buffer_center_cross)}")
        
        # Update ball_on_left_of_center *after* checking crossing based on previous state
        if is_currently_left:
            if not self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball now clearly on left (X={ball_x_global}).")
            self.ball_on_left_of_center = True
        elif is_currently_right: # If ball moves back to the right, reset the flag
            if self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball returned to right (X={ball_x_global}), resetting left flag.")
            self.ball_on_left_of_center = False
        
        self.last_ball_x_global = ball_x_global
        # Prediction logic (simplified or omitted for brevity in this snippet, but was in original)


    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_processing_time = time.monotonic()
        if self.use_video_file: 
            current_processing_time = self.frame_counter / self.actual_fps if self.actual_fps > 0 else self.frame_counter / self.target_fps

        processed_event_this_cycle = False
        
        # Sort events by timestamp to process chronologically
        temp_event_list = sorted(list(self.event_buffer_center_cross), key=lambda e: e.timestamp)
        new_event_buffer = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS) # To rebuild the buffer

        committed_event_ts = -1 # Timestamp of the event committed in this cycle

        # --- Stage 1: Try to commit an ACTUAL (non-predicted) event ---
        actual_event_to_commit = None
        for event_idx, event in enumerate(temp_event_list):
            if event.processed: continue # Already handled
            if not event.predicted: # Is an actual crossing event
                # Check cooldown against last *committed* crossing
                if event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                    actual_event_to_commit = event
                    break 
        
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
                
                self.last_committed_crossing_time = event.timestamp # Update cooldown reference
                self.ball_on_left_of_center = True # After a right-to-left crossing, ball is on the left

                if self.debug_mode: 
                    print(f"--- COMMITTED ACTUAL Event #{len(self.collected_net_speeds)}: Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. Cooldown from {event.timestamp:.3f} ---")
                
                event.processed = True
                processed_event_this_cycle = True
                committed_event_ts = event.timestamp
                
                if self.crossing_callback:
                    self.crossing_callback(event.speed_kmh, relative_time)
            else: # Max speeds collected, but mark event as processed to prevent re-evaluation
                event.processed = True 
        
        # Stage 2: Prediction logic (omitted for brevity, but was in original)

        # --- Stage 3: Clean up buffer ---
        # If an event was committed, invalidate nearby (in time) unprocessed events
        if committed_event_ts > 0:
            for event_in_list in temp_event_list:
                if not event_in_list.processed and abs(event_in_list.timestamp - committed_event_ts) < self.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0:
                    event_in_list.processed = True # Nullify events too close to a committed one
                    if self.debug_mode: 
                        print(f"DEBUG PROC: Nullified nearby event (Pred: {event_in_list.predicted}, T: {event_in_list.timestamp:.3f}) due to commit at {committed_event_ts:.3f}")
        
        # Rebuild buffer with events that are not processed and not too old
        for event_in_list in temp_event_list:
            # Keep if not processed AND (it's recent OR it's a future predicted event)
            if not event_in_list.processed and \
               ( (current_processing_time - event_in_list.timestamp < 2.0) or \
                 (event_in_list.predicted and event_in_list.timestamp > current_processing_time) ):
                new_event_buffer.append(event_in_list)
        self.event_buffer_center_cross = new_event_buffer


        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Collected {self.max_net_speeds_to_collect} net speeds. Generating output.")
            self._generate_outputs_async(self.output_folder_for_current_session) # Pass stored output folder
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION: 
                # This would typically involve calling a method that GUI can react to,
                # or if GUI directly manages is_counting_active, this line is just internal.
                # For now, assuming internal state change.
                self.is_counting_active = False 
                print("Auto-stopped counting after collecting max speeds.")


    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return

        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob

        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        delta_t = t2 - t1

        if delta_t > 0.0001: # Avoid division by zero or tiny dt
            speed_cm_per_sec = dist_cm / delta_t # Assuming t1, t2 are in seconds
            speed_kmh = speed_cm_per_sec * KMH_CONVERSION_FACTOR # 1 cm/s = 0.036 km/h

            # Apply smoothing
            if self.current_ball_speed_kmh > 0: # If there's a previous speed
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else: # First speed calculation or reset
                self.current_ball_speed_kmh = speed_kmh
        else: # If delta_t is too small, decay current speed slightly or hold it
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR) # Or some other handling


    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        # Convert global y to ROI y for perspective lookup, clamping to ROI bounds
        y1_roi = max(0, min(self.roi_height_px, y1_g - self.roi_top_y))
        y2_roi = max(0, min(self.roi_height_px, y2_g - self.roi_top_y))

        # Round to nearest 10 for lookup table key
        y1_roi_rounded = round(y1_roi / 10) * 10
        y2_roi_rounded = round(y2_roi / 10) * 10
        
        # Get pixel_to_cm ratio from lookup table, with fallback to direct calculation if key not found
        # or if ROI is not well-defined.
        default_ratio = self.pixels_per_cm_nominal 
        if self.perspective_lookup_px_to_cm: # Check if LUT is not empty
            default_ratio = next(iter(self.perspective_lookup_px_to_cm.values())) # Get a typical value from LUT

        ratio1 = self.perspective_lookup_px_to_cm.get(y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)
        real_distance_cm = pixel_distance * avg_px_to_cm_ratio
        return real_distance_cm

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0
            self.last_ball_x_global = None # Reset last known x position as well

    def toggle_counting(self): # This is now mostly internal state management
        try:
            # self.is_counting_active is managed by start_counting/stop_counting from GUI
            if self.is_counting_active: # Called when starting
                self.count_session_id += 1
                self.collected_net_speeds = []
                self.collected_relative_times = []
                self.timing_started_for_session = False
                self.first_ball_crossing_timestamp = None
                self.event_buffer_center_cross.clear()
                self.output_generated_for_session = False
                self.ball_on_left_of_center = False # Reset for new session
                self.last_committed_crossing_time = 0 # Reset cooldown
                self.last_ball_x_global = None 
                print(f"Counting ON (Session #{self.count_session_id}, Player: {self.current_player_id}, Mode: {self.current_test_mode}) - Target: {self.max_net_speeds_to_collect} speeds. Output to: {self.output_folder_for_current_session}")
            else: # Called when stopping
                print(f"Counting OFF (Session #{self.count_session_id}).")
                if self.collected_net_speeds and not self.output_generated_for_session:
                    print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                    self._generate_outputs_async(self.output_folder_for_current_session) # Use stored path
                self.output_generated_for_session = True # Ensure it's marked as generated
        except Exception as e:
            print(f"❌ Error in toggle_counting: {e}")
            self.is_counting_active = False # Ensure consistent state on error

    def start_counting(self, player_id, test_mode, output_folder_for_session):
        self.current_player_id = player_id
        self.current_test_mode = test_mode
        self.output_folder_for_current_session = output_folder_for_session # Store for this session
        if not self.is_counting_active:
            self.is_counting_active = True
            self.toggle_counting() # Initialize session variables

    def stop_counting(self):
        if self.is_counting_active:
            self.is_counting_active = False
            self.toggle_counting() # Finalize session (e.g., output generation)

    def _generate_outputs_async(self, output_folder_base): # Accepts base output folder
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
        
        # Create copies of data for the thread
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        player_id_copy = self.current_player_id
        test_mode_copy = self.current_test_mode
        
        self.file_writer_executor.submit(self._create_output_files, 
                                         speeds_copy, times_copy, 
                                         session_id_copy, player_id_copy, test_mode_copy,
                                         output_folder_base) # Pass it to the target method

    def _create_output_files(self, net_speeds, relative_times, session_id, 
                             player_id="", test_mode="", output_folder_base=OUTPUT_DATA_FOLDER): # Accepts base folder
        try:
            if not net_speeds: return

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use the provided output_folder_base
            output_dir_path = os.path.join(output_folder_base, timestamp_str) # Use os.path.join
            os.makedirs(output_dir_path, exist_ok=True)

            if player_id and test_mode:
                filename_base = f"speed_data_{player_id}_{test_mode}_{timestamp_str}"
            else:
                filename_base = f"speed_data_{timestamp_str}"

            avg_speed = sum(net_speeds) / len(net_speeds) if net_speeds else 0
            max_speed = max(net_speeds) if net_speeds else 0
            min_speed = min(net_speeds) if net_speeds else 0

            # Chart
            try:
                chart_filename = os.path.join(output_dir_path, f'{filename_base}.png')
                plt.figure(figsize=(12, 7))
                plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
                if net_speeds: # Only plot avg_speed if there's data
                    plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Average: {avg_speed:.1f} km/h')
                
                for t, s in zip(relative_times, net_speeds): 
                    plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                
                title_parts = ["Net Crossing Speed"]
                if player_id: title_parts.append(f"Player: {player_id}")
                if test_mode: title_parts.append(f"Mode: {test_mode}")
                title_parts.append(f"Session {session_id}")
                plt.title(" - ".join(title_parts), fontsize=16)
                
                plt.xlabel('Relative Time (s)', fontsize=12)
                plt.ylabel('Speed (km/h)', fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.legend()
                
                if relative_times:
                    x_min, x_max = min(relative_times), max(relative_times)
                    x_margin = (x_max - x_min) * 0.05 if len(relative_times) > 1 and x_max > x_min else 0.5
                    plt.xlim(x_min - x_margin, x_max + x_margin)
                if net_speeds:
                    y_range = max_speed - min_speed if max_speed > min_speed else 10.0 # ensure y_range is float for calc
                    y_min_plot = max(0, min_speed - y_range * 0.1)
                    y_max_plot = max_speed + y_range * 0.1
                    if y_max_plot <= y_min_plot : y_max_plot = y_min_plot + 10.0 # Ensure max > min
                    plt.ylim(y_min_plot, y_max_plot)

                info_text_parts = [f"Count: {len(net_speeds)}"]
                if net_speeds:
                    info_text_parts.extend([f"Max: {max_speed:.1f}", f"Min: {min_speed:.1f} km/h"])
                if player_id: info_text_parts.append(f"Player: {player_id}")
                if test_mode: info_text_parts.append(f"Mode: {test_mode}")
                plt.figtext(0.02, 0.02, " | ".join(info_text_parts), fontsize=9)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for figtext
                plt.savefig(chart_filename, dpi=150)
                plt.close() # Close the figure to free memory
                print(f"   📊 Chart: {filename_base}.png")
            except Exception as e:
                print(f"❌ Error creating chart: {e}")

            # TXT
            try:
                txt_filename = os.path.join(output_dir_path, f'{filename_base}.txt')
                with open(txt_filename, 'w') as f:
                    if player_id and test_mode:
                        f.write(f"Net Speeds - Player: {player_id} | Mode: {test_mode}\n")
                    f.write(f"Session {session_id} - {timestamp_str}\n")
                    f.write("---------------------------------------\n")
                    f.write("Rel. Time (s) | Speed (km/h)\n")
                    f.write("---------------------------------------\n")
                    for t, s in zip(relative_times, net_speeds): 
                        f.write(f"{t:>13.2f} | {s:>12.1f}\n")
                    f.write("---------------------------------------\n")
                    if player_id: f.write(f"Player ID: {player_id}\n")
                    if test_mode: f.write(f"Test Mode: {test_mode}\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"Total Points: {len(net_speeds)}\n")
                    if net_speeds:
                        f.write(f"Average Speed: {avg_speed:.1f} km/h\n")
                        f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
                        f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")
                print(f"   📄 Text:  {filename_base}.txt")
            except Exception as e:
                print(f"❌ Error creating text file: {e}")

            # CSV
            try:
                csv_filename = os.path.join(output_dir_path, f'{filename_base}.csv')
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['Point Number', 'Relative Time (s)', 'Speed (km/h)']
                    if player_id: header = ['Player ID'] + header
                    if test_mode: header = ['Test Mode'] + header
                    header = ['Session ID', 'File Timestamp'] + header
                    writer.writerow(header)

                    for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                        row = [i+1, f"{t:.2f}", f"{s:.1f}"]
                        if player_id: row = [player_id] + row
                        if test_mode: row = [test_mode] + row
                        row = [session_id, timestamp_str] + row
                        writer.writerow(row)
                    
                    writer.writerow([]) # Spacer
                    writer.writerow(['Statistic', 'Value'])
                    if player_id: writer.writerow(['Player ID', player_id])
                    if test_mode: writer.writerow(['Test Mode', test_mode])
                    writer.writerow(['Session ID', session_id])
                    writer.writerow(['Total Points', len(net_speeds)])
                    if net_speeds:
                        writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
                        writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
                        writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
                print(f"   📈 CSV:   {filename_base}.csv")
            except Exception as e:
                print(f"❌ Error creating CSV file: {e}")
            
            print(f"📁 Output files saved to {output_dir_path}")
            
        except Exception as e:
            print(f"❌ General error in _create_output_files: {e}")

    def process_single_frame(self, frame):
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        # motion_mask_roi can be None if FMO detection fails (e.g. shape mismatch)
        motion_mask_roi = self._detect_fmo() if gray_roi_for_fmo.size > 0 else None # Avoid FMO on dummy gray_roi
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_global_coords, ball_timestamp = None, None

        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi, ball_global_coords, ball_timestamp = self._detect_ball_in_roi(motion_mask_roi)
            if ball_pos_in_roi: # Ball detected
                self._calculate_ball_speed()
                if self.is_counting_active and ball_global_coords:
                    self._record_potential_crossing(ball_global_coords[0], ball_global_coords[1], ball_timestamp)
            else: # No ball detected in this frame
                self.last_ball_x_global = None # Reset if ball not found
        else: # No motion mask (e.g., not enough frames for FMO or ROI issue)
            self.last_ball_x_global = None

        self._check_timeout_and_reset() # Reset trajectory if ball lost for too long
        
        if self.is_counting_active:
            self._process_crossing_events() # Process any pending crossing events

        debug_text = None
        if self.debug_mode:
            on_left_text = "Y" if self.ball_on_left_of_center else "N"
            last_commit_str = f"{self.last_committed_crossing_time:.2f}" if self.last_committed_crossing_time > 0 else "N/A"
            debug_text = f"Traj:{len(self.trajectory)} EvtBuf:{len(self.event_buffer_center_cross)} OnLeft:{on_left_text} LastCommitT:{last_commit_str}"
        
        frame_data = FrameData(
            frame=frame, roi_sub_frame=roi_sub_frame, 
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi, 
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps, 
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds), # Send copy
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times), # Send copy
            debug_display_text=debug_text, 
            frame_counter=self.frame_counter
        )
        if self.trajectory: 
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data


    def cleanup(self):
        self.running = False # Signal to any loops if they check this
        if hasattr(self, 'reader') and self.reader:
            self.reader.stop()
        if hasattr(self, 'file_writer_executor') and self.file_writer_executor:
            self.file_writer_executor.shutdown(wait=True)
        print("Tracker resources cleaned up.")


# —— GUI類別 with Settings Panel ——
class PingPongGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏓 乒乓球速度追蹤系統 v14.1 GUI (with Settings)")
        self.root.geometry("1600x950") # Increased size for settings
        
        self.tracker = None
        self.is_running = False
        self.video_file_path = None
        
        self.current_player_id = ""
        self.current_test_mode = ""
        
        self.crossing_speed_history = deque(maxlen=50)
        self.crossing_time_history = deque(maxlen=50)

        # --- Settings Variables ---
        self.settings_vars = {
            "output_folder": tk.StringVar(value=OUTPUT_DATA_FOLDER),
            "target_fps": tk.IntVar(value=DEFAULT_TARGET_FPS),
            "table_length_cm": tk.DoubleVar(value=DEFAULT_TABLE_LENGTH_CM),
            "max_net_speeds": tk.IntVar(value=MAX_NET_SPEEDS_TO_COLLECT),
            "detection_timeout": tk.DoubleVar(value=DEFAULT_DETECTION_TIMEOUT),
            "min_ball_area": tk.IntVar(value=MIN_BALL_AREA_PX),
            "max_ball_area": tk.IntVar(value=MAX_BALL_AREA_PX),
            "min_ball_circularity": tk.DoubleVar(value=MIN_BALL_CIRCULARITY),
            "roi_start_ratio": tk.DoubleVar(value=DEFAULT_ROI_START_RATIO),
            "roi_end_ratio": tk.DoubleVar(value=DEFAULT_ROI_END_RATIO),
            "roi_bottom_ratio": tk.DoubleVar(value=DEFAULT_ROI_BOTTOM_RATIO),
            "near_width_cm": tk.DoubleVar(value=NEAR_SIDE_WIDTH_CM_DEFAULT),
            "far_width_cm": tk.DoubleVar(value=FAR_SIDE_WIDTH_CM_DEFAULT),
            "debug_mode": tk.BooleanVar(value=DEBUG_MODE_DEFAULT),
            # Frame width/height hints are for camera init, actual comes from FrameReader
            "frame_width_hint": tk.IntVar(value=DEFAULT_FRAME_WIDTH), 
            "frame_height_hint": tk.IntVar(value=DEFAULT_FRAME_HEIGHT),
        }
        
        self._create_main_ui()
        self._setup_update_loop()
        
    def _create_main_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2) # Video and controls
        
        # Right panel will now have a Notebook for Data and Settings
        right_notebook_frame = ttk.Frame(main_paned)
        main_paned.add(right_notebook_frame, weight=1) # Data, Plot, Settings

        self._create_video_panel(left_frame)
        self._create_control_panel(left_frame)
        
        # Create Notebook for right panel
        self.right_notebook = ttk.Notebook(right_notebook_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        data_tab = ttk.Frame(self.right_notebook)
        settings_tab = ttk.Frame(self.right_notebook)

        self.right_notebook.add(data_tab, text="即時數據 & 圖表")
        self.right_notebook.add(settings_tab, text="系統設定")

        self._create_data_panel(data_tab) # Existing data panel goes into the first tab
        self._create_settings_panel(settings_tab) # New settings panel

    def _create_video_panel(self, parent):
        video_frame = ttk.LabelFrame(parent, text="影片顯示", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.video_label = ttk.Label(video_frame, text="請選擇影片來源並開始", 
                                    font=("Arial", 16), foreground="gray", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
    def _create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(info_frame, text="玩家ID:").pack(side=tk.LEFT)
        self.player_entry = ttk.Entry(info_frame, width=12)
        self.player_entry.pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(info_frame, text="測試模式:").pack(side=tk.LEFT)
        self.mode_entry = ttk.Entry(info_frame, width=12)
        self.mode_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        source_frame = ttk.LabelFrame(control_frame, text="影片來源", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        source_control_frame = ttk.Frame(source_frame)
        source_control_frame.pack(fill=tk.X)
        self.video_source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(source_control_frame, text="攝影機", variable=self.video_source_var, 
                       value="camera", command=self._on_source_change).pack(side=tk.LEFT)
        ttk.Radiobutton(source_control_frame, text="影片檔案", variable=self.video_source_var, 
                       value="file", command=self._on_source_change).pack(side=tk.LEFT, padx=(10, 0))
        self.select_file_button = ttk.Button(source_control_frame, text="選擇檔案", 
                                           command=self._select_video_file, state=tk.DISABLED)
        self.select_file_button.pack(side=tk.LEFT, padx=(15, 0))
        self.file_label = ttk.Label(source_control_frame, text="", foreground="blue", width=30) # Added width
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        self.start_button = ttk.Button(button_frame, text="🎥 開始影片", command=self._start_video)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_button = ttk.Button(button_frame, text="⏹️ 停止影片", command=self._stop_video, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 15))
        self.count_button = ttk.Button(button_frame, text="📊 開始計數", command=self._start_counting, state=tk.DISABLED)
        self.count_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_count_button = ttk.Button(button_frame, text="⏹️ 停止計數", command=self._stop_counting, state=tk.DISABLED)
        self.stop_count_button.pack(side=tk.LEFT, padx=(0, 15))
        self.export_button = ttk.Button(button_frame, text="💾 手動匯出", command=self._export_data, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT)
        
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="狀態:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="準備就緒")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        self.settings_feedback_label = ttk.Label(status_frame, text="", foreground="purple")
        self.settings_feedback_label.pack(side=tk.LEFT, padx=(20,0))

    def _create_data_panel(self, parent_tab): # parent_tab is the frame of the notebook tab
        data_frame = ttk.LabelFrame(parent_tab, text="即時數據", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
        
        self.speed_var = tk.StringVar(value="0.0 km/h")
        self.fps_var = tk.StringVar(value="0 FPS")
        self.count_var = tk.StringVar(value=f"0 / {self.settings_vars['max_net_speeds'].get()}")
        self.last_speed_var = tk.StringVar(value="0.0 km/h")
        
        ttk.Label(data_frame, text="當前速度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(data_frame, textvariable=self.speed_var, font=("Arial", 12, "bold"), foreground="red").grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(data_frame, text="幀率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(data_frame, textvariable=self.fps_var).grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(data_frame, text="記錄數:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(data_frame, textvariable=self.count_var).grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(data_frame, text="最後穿越:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(data_frame, textvariable=self.last_speed_var, font=("Arial", 10, "bold"), foreground="green").grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        plot_frame = ttk.LabelFrame(parent_tab, text="穿越速度圖表", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_plot(plot_frame)

    def _create_settings_panel(self, parent_tab):
        settings_scroll_canvas = tk.Canvas(parent_tab)
        settings_scrollbar = ttk.Scrollbar(parent_tab, orient="vertical", command=settings_scroll_canvas.yview)
        settings_scrollable_frame = ttk.Frame(settings_scroll_canvas)

        settings_scrollable_frame.bind(
            "<Configure>",
            lambda e: settings_scroll_canvas.configure(
                scrollregion=settings_scroll_canvas.bbox("all")
            )
        )
        settings_scroll_canvas.create_window((0, 0), window=settings_scrollable_frame, anchor="nw")
        settings_scroll_canvas.configure(yscrollcommand=settings_scrollbar.set)

        settings_scroll_canvas.pack(side="left", fill="both", expand=True)
        settings_scrollbar.pack(side="right", fill="y")

        # --- Output Settings ---
        out_frame = ttk.LabelFrame(settings_scrollable_frame, text="輸出設定", padding="10")
        out_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(out_frame, text="輸出資料夾:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        out_entry = ttk.Entry(out_frame, textvariable=self.settings_vars["output_folder"], width=40)
        out_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        out_btn = ttk.Button(out_frame, text="瀏覽...", command=self._browse_output_folder)
        out_btn.grid(row=0, column=2, padx=5, pady=2)
        out_frame.columnconfigure(1, weight=1)

        # --- General Settings ---
        gen_frame = ttk.LabelFrame(settings_scrollable_frame, text="通用設定", padding="10")
        gen_frame.pack(fill=tk.X, padx=5, pady=5)
        g_row = 0
        ttk.Label(gen_frame, text="目標FPS (相機):").grid(row=g_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.settings_vars["target_fps"], width=10).grid(row=g_row, column=1, sticky=tk.W, pady=2)
        g_row+=1
        ttk.Label(gen_frame, text="桌面長度 (cm):").grid(row=g_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.settings_vars["table_length_cm"], width=10).grid(row=g_row, column=1, sticky=tk.W, pady=2)
        g_row+=1
        ttk.Label(gen_frame, text="最大記錄次數:").grid(row=g_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.settings_vars["max_net_speeds"], width=10).grid(row=g_row, column=1, sticky=tk.W, pady=2)
        g_row+=1
        ttk.Label(gen_frame, text="預設畫面寬 (相機):").grid(row=g_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.settings_vars["frame_width_hint"], width=10).grid(row=g_row, column=1, sticky=tk.W, pady=2)
        g_row+=1
        ttk.Label(gen_frame, text="預設畫面高 (相機):").grid(row=g_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(gen_frame, textvariable=self.settings_vars["frame_height_hint"], width=10).grid(row=g_row, column=1, sticky=tk.W, pady=2)


        # --- Detection Settings ---
        det_frame = ttk.LabelFrame(settings_scrollable_frame, text="偵測參數", padding="10")
        det_frame.pack(fill=tk.X, padx=5, pady=5)
        d_row = 0
        ttk.Label(det_frame, text="偵測超時 (s):").grid(row=d_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(det_frame, textvariable=self.settings_vars["detection_timeout"], width=10).grid(row=d_row, column=1, sticky=tk.W, pady=2)
        d_row+=1
        ttk.Label(det_frame, text="最小球面積 (px):").grid(row=d_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(det_frame, textvariable=self.settings_vars["min_ball_area"], width=10).grid(row=d_row, column=1, sticky=tk.W, pady=2)
        d_row+=1
        ttk.Label(det_frame, text="最大球面積 (px):").grid(row=d_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(det_frame, textvariable=self.settings_vars["max_ball_area"], width=10).grid(row=d_row, column=1, sticky=tk.W, pady=2)
        d_row+=1
        ttk.Label(det_frame, text="最小球圓度 (0-1):").grid(row=d_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(det_frame, textvariable=self.settings_vars["min_ball_circularity"], width=10).grid(row=d_row, column=1, sticky=tk.W, pady=2)

        # --- ROI Settings ---
        roi_frame = ttk.LabelFrame(settings_scrollable_frame, text="ROI 比例 (0-1)", padding="10")
        roi_frame.pack(fill=tk.X, padx=5, pady=5)
        r_row=0
        ttk.Label(roi_frame, text="起始X比例:").grid(row=r_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(roi_frame, textvariable=self.settings_vars["roi_start_ratio"], width=10).grid(row=r_row, column=1, sticky=tk.W, pady=2)
        r_row+=1
        ttk.Label(roi_frame, text="結束X比例:").grid(row=r_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(roi_frame, textvariable=self.settings_vars["roi_end_ratio"], width=10).grid(row=r_row, column=1, sticky=tk.W, pady=2)
        r_row+=1
        ttk.Label(roi_frame, text="底部Y比例:").grid(row=r_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(roi_frame, textvariable=self.settings_vars["roi_bottom_ratio"], width=10).grid(row=r_row, column=1, sticky=tk.W, pady=2)

        # --- Perspective Settings ---
        persp_frame = ttk.LabelFrame(settings_scrollable_frame, text="透視校正寬度 (cm)", padding="10")
        persp_frame.pack(fill=tk.X, padx=5, pady=5)
        p_row=0
        ttk.Label(persp_frame, text="近端寬度:").grid(row=p_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(persp_frame, textvariable=self.settings_vars["near_width_cm"], width=10).grid(row=p_row, column=1, sticky=tk.W, pady=2)
        p_row+=1
        ttk.Label(persp_frame, text="遠端寬度:").grid(row=p_row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(persp_frame, textvariable=self.settings_vars["far_width_cm"], width=10).grid(row=p_row, column=1, sticky=tk.W, pady=2)

        # --- Debug Settings ---
        debug_frame = ttk.LabelFrame(settings_scrollable_frame, text="除錯", padding="10")
        debug_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(debug_frame, text="啟用除錯模式", variable=self.settings_vars["debug_mode"]).pack(anchor=tk.W)

        # --- Action Buttons ---
        action_frame = ttk.Frame(settings_scrollable_frame, padding="10")
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(action_frame, text="套用設定", command=self._apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="還原預設值", command=self._reset_settings_to_defaults).pack(side=tk.LEFT, padx=5)
        
    def _apply_settings(self):
        try:
            # Validate and update internal variables.
            # Basic validation: ensure numbers are numbers. More complex validation can be added.
            for key, var in self.settings_vars.items():
                if isinstance(var, (tk.IntVar, tk.DoubleVar)):
                    try:
                        var.get() # This will raise TclError if not a valid number
                    except tk.TclError:
                        messagebox.showerror("設定錯誤", f"參數 '{key}' 的值無效。請輸入數字。")
                        return
            
            # Update count_var display if max_net_speeds changed
            current_collected = len(self.tracker.collected_net_speeds) if self.tracker else 0
            self.count_var.set(f"{current_collected} / {self.settings_vars['max_net_speeds'].get()}")

            self.settings_feedback_label.config(text="設定已更新。部分設定需重啟影片生效。")
            self.root.after(3000, lambda: self.settings_feedback_label.config(text="")) # Clear after 3s
            print("⚙️ Settings updated in GUI. Restart video for tracker to use new settings.")
        except Exception as e:
            messagebox.showerror("套用設定錯誤", f"無法套用設定: {e}")

    def _reset_settings_to_defaults(self):
        if messagebox.askyesno("確認", "確定要將所有設定還原為預設值嗎？"):
            self.settings_vars["output_folder"].set(OUTPUT_DATA_FOLDER) # Global default
            self.settings_vars["target_fps"].set(DEFAULT_TARGET_FPS)
            self.settings_vars["table_length_cm"].set(DEFAULT_TABLE_LENGTH_CM)
            self.settings_vars["max_net_speeds"].set(MAX_NET_SPEEDS_TO_COLLECT)
            self.settings_vars["detection_timeout"].set(DEFAULT_DETECTION_TIMEOUT)
            self.settings_vars["min_ball_area"].set(MIN_BALL_AREA_PX)
            self.settings_vars["max_ball_area"].set(MAX_BALL_AREA_PX)
            self.settings_vars["min_ball_circularity"].set(MIN_BALL_CIRCULARITY)
            self.settings_vars["roi_start_ratio"].set(DEFAULT_ROI_START_RATIO)
            self.settings_vars["roi_end_ratio"].set(DEFAULT_ROI_END_RATIO)
            self.settings_vars["roi_bottom_ratio"].set(DEFAULT_ROI_BOTTOM_RATIO)
            self.settings_vars["near_width_cm"].set(NEAR_SIDE_WIDTH_CM_DEFAULT)
            self.settings_vars["far_width_cm"].set(FAR_SIDE_WIDTH_CM_DEFAULT)
            self.settings_vars["debug_mode"].set(DEBUG_MODE_DEFAULT)
            self.settings_vars["frame_width_hint"].set(DEFAULT_FRAME_WIDTH)
            self.settings_vars["frame_height_hint"].set(DEFAULT_FRAME_HEIGHT)

            self.settings_feedback_label.config(text="設定已還原為預設值。")
            self.root.after(3000, lambda: self.settings_feedback_label.config(text=""))
            print("⚙️ Settings reset to defaults in GUI.")


    def _browse_output_folder(self):
        directory = filedialog.askdirectory(initialdir=self.settings_vars["output_folder"].get())
        if directory:
            self.settings_vars["output_folder"].set(directory)
            print(f"Output folder set to: {directory}")

    def _create_plot(self, parent): # Unchanged
        self.fig = Figure(figsize=(6, 4), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Net Crossing Speed")
        self.ax.set_xlabel("Relative Time (s)")
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        
    def _setup_update_loop(self): # Unchanged
        self._update_display()
        
    def _update_display(self): # Mostly unchanged
        if self.is_running and self.tracker and self.tracker.reader:
            ret, frame = self.tracker.reader.read()
            if ret and frame is not None:
                frame_data = self.tracker.process_single_frame(frame)
                self._update_video_display(frame, frame_data)
                self._update_data_displays(frame_data)
            elif not ret and self.tracker.reader.running == False: # Check if reader explicitly stopped
                print("Video source ended or error.")
                self._stop_video() # Gracefully stop
        
        self.root.after(max(1, int(1000 / self.settings_vars["target_fps"].get() / 3)), self._update_display) # Dynamic refresh rate

    def _update_video_display(self, frame, frame_data): # Unchanged
        display_frame = self._draw_visualizations(frame, frame_data)
        display_height = self.video_label.winfo_height() - 10 # Use actual label height
        display_width = self.video_label.winfo_width() - 10

        if display_height <= 10 or display_width <= 10: # Fallback if panel not rendered yet
            display_height = 480
            aspect_ratio = display_frame.shape[1] / display_frame.shape[0] if display_frame.shape[0] > 0 else 1.0
            display_width = int(display_height * aspect_ratio)
        else: # Maintain aspect ratio
            frame_aspect_ratio = display_frame.shape[1] / display_frame.shape[0] if display_frame.shape[0] > 0 else 1.0
            label_aspect_ratio = display_width / display_height
            if frame_aspect_ratio > label_aspect_ratio: # Frame is wider, fit to width
                target_width = display_width
                target_height = int(target_width / frame_aspect_ratio)
            else: # Frame is taller, fit to height
                target_height = display_height
                target_width = int(target_height * frame_aspect_ratio)
            display_width = target_width
            display_height = target_height
            
        if display_width <=0 or display_height <=0 : # safety for resize
            return

        frame_resized = cv2.resize(display_frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        
    def _draw_visualizations(self, frame, frame_data): # Unchanged, but uses tracker's MAX_NET_SPEEDS...
        vis_frame = frame.copy()
        is_full_draw = frame_data.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0
        
        if is_full_draw and self.tracker: # Ensure tracker exists
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.tracker.static_overlay, 0.7, 0)
            if frame_data.trajectory_points_global and len(frame_data.trajectory_points_global) >= 2:
                pts = np.array(frame_data.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)
                
        if frame_data.ball_position_in_roi and frame_data.roi_sub_frame is not None and self.tracker:
            cx_roi, cy_roi = frame_data.ball_position_in_roi
            # Ensure roi_sub_frame is valid for drawing
            if frame_data.roi_sub_frame.shape[0] > 0 and frame_data.roi_sub_frame.shape[1] > 0 :
                cv2.circle(frame_data.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
                if frame_data.ball_contour_in_roi is not None:
                    cv2.drawContours(frame_data.roi_sub_frame, [frame_data.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            
            cx_global_vis = cx_roi + self.tracker.roi_start_x
            cy_global_vis = cy_roi + self.tracker.roi_top_y
            cv2.circle(vis_frame, (cx_global_vis, cy_global_vis), 8, BALL_COLOR_BGR, -1)
            
        cv2.putText(vis_frame, f"Speed: {frame_data.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        count_status_text = "ON" if frame_data.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        if frame_data.last_recorded_net_speed_kmh > 0: 
            cv2.putText(vis_frame, f"Last Net: {frame_data.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        # Use max_net_speeds from settings for display consistency
        max_speeds_to_show = self.settings_vars["max_net_speeds"].get() if self.tracker else MAX_NET_SPEEDS_TO_COLLECT
        cv2.putText(vis_frame, f"Recorded: {len(frame_data.collected_net_speeds)}/{max_speeds_to_show}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        if frame_data.collected_relative_times: 
            cv2.putText(vis_frame, f"Last Time: {frame_data.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
            
        if self.current_player_id or self.current_test_mode:
            player_info = f"Player: {self.current_player_id or 'N/A'} | Mode: {self.current_test_mode or 'N/A'}"
            cv2.putText(vis_frame, player_info, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), FONT_THICKNESS_VIS)
            
        if frame_data.debug_display_text: 
            cv2.putText(vis_frame, frame_data.debug_display_text, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame
        
    def _update_data_displays(self, frame_data): # Unchanged
        self.speed_var.set(f"{frame_data.current_ball_speed_kmh:.1f} km/h")
        self.fps_var.set(f"{frame_data.display_fps:.1f} FPS")
        # Use max_net_speeds from settings for display consistency
        max_speeds_to_show = self.settings_vars["max_net_speeds"].get() if self.tracker else MAX_NET_SPEEDS_TO_COLLECT
        self.count_var.set(f"{len(frame_data.collected_net_speeds)} / {max_speeds_to_show}")
        self.last_speed_var.set(f"{frame_data.last_recorded_net_speed_kmh:.1f} km/h")
        
    def _on_crossing_event(self, speed_kmh, relative_time): # Unchanged
        self.crossing_speed_history.append(speed_kmh)
        self.crossing_time_history.append(relative_time)
        self._update_crossing_plot()
        
    def _update_crossing_plot(self): # Unchanged
        self.ax.clear() # Clear previous plot
        if self.crossing_speed_history: # Only plot if there's data
            self.ax.plot(list(self.crossing_time_history), list(self.crossing_speed_history), 'ro-', linewidth=2, markersize=6)
            
            max_s = max(self.crossing_speed_history) if self.crossing_speed_history else 10
            min_s = min(self.crossing_speed_history) if self.crossing_speed_history else 0
            self.ax.set_ylim(max(0, min_s - (max_s-min_s)*0.1 if (max_s-min_s)>0 else min_s-5 ), max_s * 1.1 if max_s > 0 else 10)

            latest_speed = self.crossing_speed_history[-1]
            latest_time = self.crossing_time_history[-1]
            self.ax.annotate(f'{latest_speed:.1f}', (latest_time, latest_speed), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=10, color='red', weight='bold')
        
        self.ax.set_title("Net Crossing Speed")
        self.ax.set_xlabel("Relative Time (s)")
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def _on_source_change(self): # Unchanged
        if self.video_source_var.get() == "file":
            self.select_file_button.config(state=tk.NORMAL)
        else:
            self.select_file_button.config(state=tk.DISABLED)
            self.video_file_path = None # Clear path if switching to camera
            self.file_label.config(text="")
    
    def _select_video_file(self): # Unchanged
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="選擇影片檔案", filetypes=filetypes)
        if filename:
            self.video_file_path = filename
            display_name = os.path.basename(filename)
            if len(display_name) > 25: display_name = "..." + display_name[-22:] # Truncate
            self.file_label.config(text=display_name)
    
    def _start_video(self): # MODIFIED to use settings_vars
        try:
            self.current_player_id = self.player_entry.get().strip()
            self.current_test_mode = self.mode_entry.get().strip()
            
            if self.video_source_var.get() == "file":
                if not self.video_file_path:
                    messagebox.showerror("錯誤", "請先選擇影片檔案")
                    return
                video_source = self.video_file_path
                use_video_file = True
            else:
                video_source = DEFAULT_CAMERA_INDEX # Can be changed to an Entry in future
                use_video_file = False
            
            # Create tracker with values from GUI settings
            self.tracker = Core_PingPongSpeedTracker(
                video_source=video_source,
                use_video_file=use_video_file,
                target_fps=self.settings_vars["target_fps"].get(),
                frame_width_hint=self.settings_vars["frame_width_hint"].get(),
                frame_height_hint=self.settings_vars["frame_height_hint"].get(),
                table_length_cm=self.settings_vars["table_length_cm"].get(),
                detection_timeout_s=self.settings_vars["detection_timeout"].get(),
                roi_start_ratio=self.settings_vars["roi_start_ratio"].get(),
                roi_end_ratio=self.settings_vars["roi_end_ratio"].get(),
                roi_bottom_ratio=self.settings_vars["roi_bottom_ratio"].get(),
                max_net_speeds=self.settings_vars["max_net_speeds"].get(),
                near_width_cm=self.settings_vars["near_width_cm"].get(),
                far_width_cm=self.settings_vars["far_width_cm"].get(),
                min_ball_area=self.settings_vars["min_ball_area"].get(),
                max_ball_area=self.settings_vars["max_ball_area"].get(),
                min_ball_circularity=self.settings_vars["min_ball_circularity"].get(),
                debug_mode=self.settings_vars["debug_mode"].get()
            )
            
            self.tracker.current_player_id = self.current_player_id # Set these directly for now
            self.tracker.current_test_mode = self.current_test_mode
            self.tracker.crossing_callback = self._on_crossing_event
            
            self.tracker.reader.start()
            self.is_running = True
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.count_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.NORMAL) # Enable export once video starts
            self.player_entry.config(state=tk.DISABLED) # Lock player info during session
            self.mode_entry.config(state=tk.DISABLED)
            # Disable settings tabs/widgets while running (optional, but safer)
            # For simplicity, we'll just rely on "Apply Settings" and restart.
            
            self.status_var.set("影片運行中")
            self.status_label.config(foreground="green")
            self.settings_feedback_label.config(text="") # Clear settings message
            
            print(f"✅ Video started. Tracker using: {self.tracker.frame_width}x{self.tracker.frame_height} @ effective {self.tracker.actual_fps:.1f} FPS")
            
        except Exception as e:
            messagebox.showerror("啟動影片錯誤", f"無法啟動影片: {e}")
            print(f"❌ Video start error: {e}")
            if self.tracker: # Partial cleanup if tracker was created
                self.tracker.cleanup()
                self.tracker = None
            self._reset_ui_to_stopped_state() # Ensure UI is consistent
            
    def _stop_video(self): # Mostly unchanged, added export_button disable
        try:
            if self.tracker:
                if self.tracker.is_counting_active:
                    self.tracker.stop_counting() # This will also trigger export if data exists
                self.tracker.cleanup()
                self.tracker = None
            self.is_running = False
            self._reset_ui_to_stopped_state()
            
            print("✅ Video stopped.")
            self.video_label.configure(image=None, text="請選擇影片來源並開始") # Reset video display
            self.video_label.image = None

            self.crossing_speed_history.clear()
            self.crossing_time_history.clear()
            self._update_crossing_plot() # Redraw empty plot
            
        except Exception as e:
            print(f"❌ Error stopping video: {e}")
            self._reset_ui_to_stopped_state() # Ensure UI consistency on error

    def _reset_ui_to_stopped_state(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.count_button.config(state=tk.DISABLED)
        self.stop_count_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED) # Disable export when no video/tracker
        self.player_entry.config(state=tk.NORMAL)
        self.mode_entry.config(state=tk.NORMAL)
        self.status_var.set("已停止")
        self.status_label.config(foreground="red")

    def _start_counting(self): # MODIFIED to pass output folder
        try:
            if not self.tracker or not self.is_running:
                messagebox.showerror("錯誤", "請先啟動影片！")
                return
            
            self.current_player_id = self.player_entry.get().strip() # Update before starting count
            self.current_test_mode = self.mode_entry.get().strip()
            
            output_folder = self.settings_vars["output_folder"].get()
            if not os.path.isdir(output_folder):
                try:
                    os.makedirs(output_folder, exist_ok=True)
                    print(f"Created output directory: {output_folder}")
                except Exception as e:
                    messagebox.showerror("輸出路徑錯誤", f"無法創建輸出資料夾 '{output_folder}': {e}")
                    return

            self.tracker.start_counting(self.current_player_id, self.current_test_mode, output_folder)
            self.count_button.config(state=tk.DISABLED)
            self.stop_count_button.config(state=tk.NORMAL)
            
            self.crossing_speed_history.clear() # Clear plot for new count session
            self.crossing_time_history.clear()
            self._update_crossing_plot()
            
            print(f"🟢 Counting started by GUI. Player: {self.current_player_id}, Mode: {self.current_test_mode}, Output: {output_folder}")
        except Exception as e:
            print(f"❌ Error starting counting: {e}")
            messagebox.showerror("計數錯誤", f"開始計數時發生錯誤: {e}")
            self.count_button.config(state=tk.NORMAL) # Reset button states
            self.stop_count_button.config(state=tk.DISABLED)
    
    def _stop_counting(self): # Unchanged
        try:
            if self.tracker and self.tracker.is_counting_active:
                self.tracker.stop_counting() # This will trigger data export if needed
            # Button states are managed by tracker's is_counting_active flag usually,
            # but ensure they are correct after manual stop.
            self.count_button.config(state=tk.NORMAL if self.is_running else tk.DISABLED)
            self.stop_count_button.config(state=tk.DISABLED)
            print(f"🔴 Counting stopped by GUI.")
        except Exception as e:
            print(f"❌ Error stopping counting: {e}")
            messagebox.showerror("計數錯誤", f"停止計數時發生錯誤: {e}")
            # Try to restore sensible button states
            if self.is_running and self.tracker:
                 self.count_button.config(state=tk.NORMAL if not self.tracker.is_counting_active else tk.DISABLED)
                 self.stop_count_button.config(state=tk.NORMAL if self.tracker.is_counting_active else tk.DISABLED)

    def _export_data(self): # MODIFIED to pass output folder from settings
        try:
            if not self.tracker or not self.tracker.collected_net_speeds:
                messagebox.showwarning("警告", "沒有已記錄的數據可以匯出。請先開始計數並記錄數據。")
                return
            
            output_folder = self.settings_vars["output_folder"].get()
            if not os.path.isdir(output_folder):
                try:
                    os.makedirs(output_folder, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("輸出路徑錯誤", f"無法創建輸出資料夾 '{output_folder}': {e}")
                    return
            
            # Ensure tracker has up-to-date player/mode info if changed after counting started (though UI locks them)
            self.tracker.current_player_id = self.player_entry.get().strip()
            self.tracker.current_test_mode = self.mode_entry.get().strip()
            
            self.tracker._generate_outputs_async(output_folder) # Pass current output folder
            messagebox.showinfo("成功", f"數據已開始異步匯出至 '{output_folder}'.\n請檢查主控台輸出以確認完成。")
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            messagebox.showerror("匯出錯誤", f"匯出數據時發生錯誤: {e}")
    
    def run(self): # Unchanged
        self.root.protocol("WM_DELETE_WINDOW", self._quit_app)
        self.root.mainloop()
    
    def _quit_app(self): # Unchanged
        try:
            print("Attempting to quit application...")
            if self.is_running:
                self._stop_video() # Try to gracefully stop video and tracker
            elif self.tracker: # If tracker exists but not running (e.g. after stop_video)
                self.tracker.cleanup()
        except Exception as e:
            print(f"❌ Error during pre-quit cleanup: {e}")
        finally:
            try:
                self.root.quit()
                self.root.destroy()
                print("Application quit.")
            except Exception as e:
                print(f"❌ Error destroying root window: {e}")

def main():
    app = PingPongGUI()
    app.run()

if __name__ == '__main__':
    main()