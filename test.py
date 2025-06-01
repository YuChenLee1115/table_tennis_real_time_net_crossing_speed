#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11 (OPTIMIZED FOR M2 PRO)
# Enhanced with: Memory pre-allocation, Vectorization, Multi-threading, Data structure optimization, Async visualization, Batch I/O

import cv2
import numpy as np
import time
import datetime
from collections import deque, namedtuple
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
import array
from functools import wraps

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

# Center Line Detection Related
MAX_NET_SPEEDS_TO_COLLECT = 30
NET_CROSSING_DIRECTION_DEFAULT = 'right_to_left'
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
EVENT_BUFFER_SIZE_CENTER_CROSS = 200

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization for Apple Silicon ——
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # Let OpenCV choose optimal thread count
import os
os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '1'

# —— Optimized Data Structures ——
TrajectoryPoint = namedtuple('TrajectoryPoint', ['x', 'y', 't'])

class OptimizedTrajectory:
    """Memory-efficient trajectory storage using arrays"""
    def __init__(self, maxlen=MAX_TRAJECTORY_POINTS):
        self.maxlen = maxlen
        self.x_coords = array.array('f')  # float array
        self.y_coords = array.array('f')
        self.timestamps = array.array('d')  # double array
        
    def append(self, x, y, t):
        if len(self.x_coords) >= self.maxlen:
            # Batch remove 1/4 of points for efficiency
            quarter = self.maxlen // 4
            del self.x_coords[:quarter]
            del self.y_coords[:quarter]
            del self.timestamps[:quarter]
            
        self.x_coords.append(float(x))
        self.y_coords.append(float(y))
        self.timestamps.append(float(t))
    
    def __len__(self):
        return len(self.x_coords)
    
    def __getitem__(self, index):
        if index < 0:
            index = len(self.x_coords) + index
        return TrajectoryPoint(self.x_coords[index], self.y_coords[index], self.timestamps[index])
    
    def __iter__(self):
        for i in range(len(self.x_coords)):
            yield TrajectoryPoint(self.x_coords[i], self.y_coords[i], self.timestamps[i])
    
    def clear(self):
        self.x_coords = array.array('f')
        self.y_coords = array.array('f')
        self.timestamps = array.array('d')
    
    def get_recent_points(self, n):
        """Get last n points as list for visualization"""
        start_idx = max(0, len(self.x_coords) - n)
        return [(int(self.x_coords[i]), int(self.y_coords[i])) 
                for i in range(start_idx, len(self.x_coords))]

class ObjectPool:
    """Object pool for reducing memory allocation overhead"""
    def __init__(self, create_func, max_size=10):
        self.create_func = create_func
        self.pool = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def get(self):
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            return self.create_func()
    
    def put(self, obj):
        with self.lock:
            self.pool.append(obj)

class BatchFileWriter:
    """Batch file operations to reduce I/O overhead"""
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        self.pending_data = []
        self.lock = threading.Lock()
        
    def add_data(self, filename, data):
        with self.lock:
            self.pending_data.append((filename, data))
            if len(self.pending_data) >= self.batch_size:
                self._flush_batch()
                
    def _flush_batch(self):
        # Group by filename for efficient writing
        file_groups = {}
        for filename, data in self.pending_data:
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append(data)
        
        # Batch write to each file
        for filename, data_list in file_groups.items():
            try:
                with open(filename, 'a') as f:
                    for data in data_list:
                        f.write(data)
            except Exception as e:
                print(f"Batch write error for {filename}: {e}")
        
        self.pending_data.clear()
    
    def flush(self):
        with self.lock:
            if self.pending_data:
                self._flush_batch()

class AsyncVisualizer:
    """Asynchronous visualization to avoid blocking main processing"""
    def __init__(self, max_workers=2):
        self.vis_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.last_vis_frame = None
        self.current_vis_future = None
        
    def update_visualization_async(self, frame, frame_data, draw_func):
        # Return last frame if current visualization is still processing
        if self.current_vis_future and not self.current_vis_future.done():
            return self.last_vis_frame if self.last_vis_frame is not None else frame
        
        # Get result of previous visualization if available
        if self.current_vis_future and self.current_vis_future.done():
            try:
                self.last_vis_frame = self.current_vis_future.result(timeout=0.001)
            except (concurrent.futures.TimeoutError, Exception):
                pass
        
        # Start new visualization task
        self.current_vis_future = self.vis_executor.submit(draw_func, frame.copy(), frame_data)
        
        return self.last_vis_frame if self.last_vis_frame is not None else frame
    
    def shutdown(self):
        if self.current_vis_future:
            try:
                self.current_vis_future.result(timeout=0.1)
            except:
                pass
        self.vis_executor.shutdown(wait=False)

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

class PipelineProcessor:
    """Multi-threaded pipeline processing for better CPU utilization"""
    def __init__(self, num_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.processing_queue = queue.Queue(maxsize=20)
        
    def process_frame_pipeline(self, tracker, frame):
        """Process frame in pipeline stages"""
        # Stage 1: Preprocessing (can be done in parallel)
        future_preprocess = self.executor.submit(tracker._preprocess_frame, frame)
        
        # Stage 2: FMO detection (depends on preprocessing)
        future_fmo = self.executor.submit(self._detect_fmo_when_ready, tracker, future_preprocess)
        
        # Stage 3: Ball detection (depends on FMO)
        future_ball = self.executor.submit(self._detect_ball_when_ready, tracker, future_fmo)
        
        return future_ball
    
    def _detect_fmo_when_ready(self, tracker, preprocess_future):
        try:
            roi_sub_frame, gray_roi_blurred = preprocess_future.result(timeout=0.1)
            return tracker._detect_fmo_vectorized(), roi_sub_frame
        except concurrent.futures.TimeoutError:
            return None, None
    
    def _detect_ball_when_ready(self, tracker, fmo_future):
        try:
            motion_mask_roi, roi_sub_frame = fmo_future.result(timeout=0.1)
            if motion_mask_roi is not None:
                return tracker._detect_ball_in_roi(motion_mask_roi), roi_sub_frame
            return (None, None, None, None), roi_sub_frame
        except concurrent.futures.TimeoutError:
            return (None, None, None, None), None
    
    def shutdown(self):
        self.executor.shutdown(wait=False)

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

        # Use optimized trajectory
        self.trajectory = OptimizedTrajectory(MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        # —— Memory Pre-allocation ——
        self._init_memory_buffers()
        self._init_morphology_kernels()

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
        
        # —— Optimized components ——
        self.batch_writer = BatchFileWriter()
        self.async_visualizer = AsyncVisualizer()
        self.pipeline_processor = PipelineProcessor()
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # State variables for crossing detection
        self.ball_on_left_of_center = False 
        self.last_committed_crossing_time = 0 
        self.EFFECTIVE_CROSSING_COOLDOWN_S = 0.3
        self.CENTER_ZONE_WIDTH_PIXELS = self.frame_width * 0.05
        
        # User input variables for file naming
        self.current_player_id = ""
        self.current_test_mode = ""

        self._precalculate_overlay()
        self._create_perspective_lookup_table()
        
        # Pre-compute frequently used values
        self.center_zone_left = self.center_x_global - self.CENTER_ZONE_WIDTH_PIXELS
        self.center_zone_right = self.center_x_global + self.CENTER_ZONE_WIDTH_PIXELS
        self.distance_threshold_sq = (self.frame_width * 0.2) ** 2

    def _sanitize_filename_input(self, input_str):
        """Sanitize user input for safe filename usage"""
        import re
        # Remove or replace invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', input_str.strip())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length
        return sanitized[:50] if sanitized else "unknown"

    def _get_user_input_for_session(self):
        """Get player ID and test mode from user input"""
        print("\n" + "="*50)
        print("📝 SESSION SETUP - Please provide the following information:")
        print("="*50)
        
        # Get Player ID
        while True:
            player_id = input("🏓 Enter Player ID (e.g., 'John_Doe', 'Player1'): ").strip()
            if player_id:
                self.current_player_id = self._sanitize_filename_input(player_id)
                print(f"✅ Player ID set to: {self.current_player_id}")
                break
            else:
                print("❌ Player ID cannot be empty. Please try again.")
        
        # Get Test Mode
        print("\n📋 Common test modes: 'forehand', 'backhand', 'serve', 'rally', 'practice', 'match'")
        while True:
            test_mode = input("🎯 Enter Test Mode: ").strip()
            if test_mode:
                self.current_test_mode = self._sanitize_filename_input(test_mode)
                print(f"✅ Test Mode set to: {self.current_test_mode}")
                break
            else:
                print("❌ Test Mode cannot be empty. Please try again.")
        
        print("\n" + "="*50)
        print(f"🚀 Session ready! Files will be saved as:")
        print(f"   speed_data_{self.current_player_id}_{self.current_test_mode}_[timestamp].*")
        print("="*50 + "\n")
        return True

    def _init_memory_buffers(self):
        """Pre-allocate work buffers to avoid repeated memory allocation"""
        self.work_buffer_gray = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.work_buffer_diff1 = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.work_buffer_diff2 = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.work_buffer_motion = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.work_buffer_thresh = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        
        # Buffer for storing previous frames
        self.prev_frames_buffer = np.empty((MAX_PREV_FRAMES_FMO, self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.prev_frames_count = 0
        self.prev_frames_index = 0

    def _init_morphology_kernels(self):
        """Pre-compute morphology kernels"""
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # ROI Box lines
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        # Center line
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: Toggle Count (Setup Player/Mode) | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10):
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y)
        if self.roi_bottom_y == 0: 
            relative_y = 0.5
        else: 
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        if current_width_cm > 0 and self.roi_width_px > 0:
            pixel_to_cm_ratio = current_width_cm / self.roi_width_px
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
        cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY, dst=self.work_buffer_gray)
        cv2.GaussianBlur(self.work_buffer_gray, (5, 5), 0, dst=self.work_buffer_gray)
        
        # Store in circular buffer
        self.prev_frames_buffer[self.prev_frames_index] = self.work_buffer_gray.copy()
        self.prev_frames_index = (self.prev_frames_index + 1) % MAX_PREV_FRAMES_FMO
        self.prev_frames_count = min(self.prev_frames_count + 1, MAX_PREV_FRAMES_FMO)
        
        return roi_sub_frame, self.work_buffer_gray

    def _detect_fmo_vectorized(self):
        """Vectorized FMO detection using NumPy operations"""
        if self.prev_frames_count < 3:
            return None
        
        # Get indices for last 3 frames
        idx_current = (self.prev_frames_index - 1) % MAX_PREV_FRAMES_FMO
        idx_prev1 = (self.prev_frames_index - 2) % MAX_PREV_FRAMES_FMO
        idx_prev2 = (self.prev_frames_index - 3) % MAX_PREV_FRAMES_FMO
        
        # Vectorized difference calculation
        f1 = self.prev_frames_buffer[idx_prev2]
        f2 = self.prev_frames_buffer[idx_prev1]  
        f3 = self.prev_frames_buffer[idx_current]
        
        # Use pre-allocated buffers for differences
        cv2.absdiff(f1, f2, dst=self.work_buffer_diff1)
        cv2.absdiff(f2, f3, dst=self.work_buffer_diff2)
        
        # Vectorized bitwise AND
        cv2.bitwise_and(self.work_buffer_diff1, self.work_buffer_diff2, dst=self.work_buffer_motion)
        
        try:
            cv2.threshold(self.work_buffer_motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=self.work_buffer_thresh)
        except cv2.error:
            cv2.threshold(self.work_buffer_motion, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY, dst=self.work_buffer_thresh)
        
        if OPENING_KERNEL_SIZE_FMO[0] > 0:
            cv2.morphologyEx(self.work_buffer_thresh, cv2.MORPH_OPEN, self.opening_kernel, dst=self.work_buffer_thresh)
        
        cv2.morphologyEx(self.work_buffer_thresh, cv2.MORPH_CLOSE, self.closing_kernel, dst=self.work_buffer_thresh)
        
        return self.work_buffer_thresh

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        potential_balls = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
                x_roi, y_roi = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                w_roi, h_roi = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx_roi, cy_roi = centroids[i]
                circularity = 0
                contour_to_store = None
                
                if max(w_roi, h_roi) > 0:
                    component_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]
                        contour_to_store = cnt
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0: 
                            circularity = 4 * math.pi * area / (perimeter * perimeter)
                            
                potential_balls.append({
                    'position_roi': (int(cx_roi), int(cy_roi)), 
                    'area': area,
                    'circularity': circularity, 
                    'contour_roi': contour_to_store
                })
        
        if not potential_balls: 
            return None, None, None, None

        best_ball_info = self._select_best_ball_candidate_optimized(potential_balls)
        if not best_ball_info: 
            return None, None, None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file: 
            current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()
        self.trajectory.append(cx_global, cy_global, current_timestamp)
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}, T:{current_timestamp:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi'), (cx_global, cy_global), current_timestamp

    def _select_best_ball_candidate_optimized(self, candidates):
        """Optimized candidate selection with vectorized distance calculation"""
        if not candidates: 
            return None
            
        if len(self.trajectory) == 0:
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular: 
                return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])

        last_point = self.trajectory[-1]
        last_x_global, last_y_global = last_point.x, last_point.y
        
        # Vectorized distance calculation
        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global, cy_global = cx_roi + self.roi_start_x, cy_roi + self.roi_top_y
            
            # Use pre-computed threshold for efficiency
            distance_sq = (cx_global - last_x_global) ** 2 + (cy_global - last_y_global) ** 2
            if distance_sq <= self.distance_threshold_sq:
                ball_info['distance_from_last'] = math.sqrt(distance_sq)
            else:
                ball_info['distance_from_last'] = float('inf')
            
            # Consistency score calculation
            consistency_score = 0
            if len(self.trajectory) >= 2:
                prev_point = self.trajectory[-2]
                prev_x_global, prev_y_global = prev_point.x, prev_point.y
                
                vec_hist_dx, vec_hist_dy = last_x_global - prev_x_global, last_y_global - prev_y_global
                vec_curr_dx, vec_curr_dy = cx_global - last_x_global, cy_global - last_y_global
                
                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist_sq = vec_hist_dx**2 + vec_hist_dy**2
                mag_curr_sq = vec_curr_dx**2 + vec_curr_dy**2
                
                if mag_hist_sq > 0 and mag_curr_sq > 0:
                    cosine_similarity = dot_product / (math.sqrt(mag_hist_sq * mag_curr_sq))
                    consistency_score = max(0, cosine_similarity)
                    
            ball_info['consistency'] = consistency_score
        
        # Vectorized scoring
        for ball_info in candidates:
            ball_info['score'] = (0.4 / (1.0 + ball_info['distance_from_last'])) + \
                                 (0.4 * ball_info['consistency']) + \
                                 (0.2 * ball_info['circularity'])
        
        return max(candidates, key=lambda b: b['score'])

    def toggle_counting(self):
        self.is_counting_active = not self.is_counting_active
        if self.is_counting_active:
            # Get user input for session before starting
            print("\n⏸️  Video processing paused for session setup...")
            if not self._get_user_input_for_session():
                print("❌ Session setup cancelled. Counting remains OFF.")
                self.is_counting_active = False
                return
            
            self.count_session_id += 1
            self.collected_net_speeds = []
            self.collected_relative_times = []
            self.timing_started_for_session = False
            self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear()
            self.output_generated_for_session = False
            self.ball_on_left_of_center = False
            self.last_committed_crossing_time = 0
            self.last_ball_x_global = None 
            print(f"🟢 Counting ON (Session #{self.count_session_id}) - Target: {self.max_net_speeds_to_collect} speeds.")
            print(f"📊 Player: {self.current_player_id}, Mode: {self.current_test_mode}")
        else:
            print(f"🔴 Counting OFF (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"💾 Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_outputs_async()
            self.output_generated_for_session = True

    def _record_potential_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        if not self.is_counting_active:
            self.last_ball_x_global = ball_x_global
            return

        if self.net_crossing_direction not in ['right_to_left', 'both']:
            self.last_ball_x_global = ball_x_global
            return

        # Cooldown check
        if current_timestamp - self.last_committed_crossing_time < self.EFFECTIVE_CROSSING_COOLDOWN_S:
            if self.debug_mode: 
                print(f"DEBUG REC: In cooldown. CT: {current_timestamp:.3f}, LastCommitT: {self.last_committed_crossing_time:.3f}")
            self.last_ball_x_global = ball_x_global
            return

        # Update ball position relative to center using pre-computed zones
        if ball_x_global < self.center_zone_left:
            if not self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball now clearly on left (X={ball_x_global}).")
            self.ball_on_left_of_center = True
        elif ball_x_global > self.center_zone_right:
            if self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball returned to right (X={ball_x_global}), resetting left flag.")
            self.ball_on_left_of_center = False

        # Crossing detection
        crossed_r_to_l_strictly = False
        if (self.last_ball_x_global is not None and 
            self.last_ball_x_global >= self.center_x_global and 
            ball_x_global < self.center_x_global and 
            not self.ball_on_left_of_center):
            
            crossed_r_to_l_strictly = True
            if self.debug_mode:
                print(f"DEBUG REC: Strict R-L Actual Crossing Detected. PrevX: {self.last_ball_x_global:.1f}, CurrX: {ball_x_global:.1f}. Speed: {self.current_ball_speed_kmh:.1f}")

        if crossed_r_to_l_strictly and self.current_ball_speed_kmh > 0.1:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            if self.debug_mode: 
                print(f"DEBUG REC: Added ACTUAL event to buffer. Buffer size: {len(self.event_buffer_center_cross)}")

        # Prediction logic (simplified for performance)
        if (not crossed_r_to_l_strictly and not self.ball_on_left_of_center and 
            len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 0.1 and 
            ball_x_global >= self.center_x_global):
            
            self._add_prediction_if_needed(ball_x_global, current_timestamp)
        
        self.last_ball_x_global = ball_x_global

    def _add_prediction_if_needed(self, ball_x_global, current_timestamp):
        """Optimized prediction logic"""
        pt1 = self.trajectory[-2]
        pt2 = self.trajectory[-1]
        
        delta_t_hist = pt2.t - pt1.t
        if delta_t_hist <= 0:
            return
            
        vx_pixels_per_time_unit = (pt2.x - pt1.x) / delta_t_hist
        
        # Pre-compute minimum velocity threshold
        fps_for_calc = self.display_fps if self.display_fps > 1 else self.target_fps
        min_vx_for_prediction = -(self.frame_width * 0.02) * (delta_t_hist / (1.0 / fps_for_calc))

        if vx_pixels_per_time_unit < min_vx_for_prediction:
            for lookahead_frames in [1, 2]:
                time_to_predict = lookahead_frames / fps_for_calc
                predicted_x_at_crossing_time = ball_x_global + vx_pixels_per_time_unit * time_to_predict
                predicted_timestamp = current_timestamp + time_to_predict

                if predicted_x_at_crossing_time < self.center_x_global:
                    # Check for existing similar predictions (optimized)
                    time_threshold = 1.0 / fps_for_calc
                    can_add_prediction = not any(
                        ev.predicted and abs(ev.timestamp - predicted_timestamp) < time_threshold
                        for ev in self.event_buffer_center_cross
                    )
                    
                    if can_add_prediction:
                        if self.debug_mode: 
                            print(f"DEBUG REC: Added PREDICTED event. X_pred: {predicted_x_at_crossing_time:.1f} at T_pred: {predicted_timestamp:.3f}")
                        event = EventRecord(predicted_x_at_crossing_time, predicted_timestamp, self.current_ball_speed_kmh, predicted=True)
                        self.event_buffer_center_cross.append(event)
                    break

    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_processing_time = time.monotonic()
        if self.use_video_file: 
            current_processing_time = self.frame_counter / self.actual_fps

        # Convert deque to list for efficient processing
        temp_event_list = sorted(list(self.event_buffer_center_cross), key=lambda e: e.timestamp)
        new_event_buffer = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)

        committed_event_ts = -1

        # Process actual events first
        actual_event_to_commit = None
        for event in temp_event_list:
            if event.processed or event.predicted:
                continue
            if event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                actual_event_to_commit = event
                break

        if actual_event_to_commit:
            self._commit_event(actual_event_to_commit)
            committed_event_ts = actual_event_to_commit.timestamp
        else:
            # Process predicted events
            predicted_event_to_commit = None
            for event in temp_event_list:
                if event.processed or not event.predicted:
                    continue
                if (current_processing_time >= event.timestamp and 
                    event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S):
                    predicted_event_to_commit = event
                    break
            
            if predicted_event_to_commit:
                self._commit_event(predicted_event_to_commit)
                committed_event_ts = predicted_event_to_commit.timestamp

        # Clean up events near committed timestamp
        if committed_event_ts > 0:
            cleanup_threshold = self.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0
            for event_in_list in temp_event_list:
                if not event_in_list.processed and abs(event_in_list.timestamp - committed_event_ts) < cleanup_threshold:
                    event_in_list.processed = True

        # Rebuild buffer with unprocessed, recent events
        for event_in_list in temp_event_list:
            if not event_in_list.processed and (current_processing_time - event_in_list.timestamp < 2.0):
                new_event_buffer.append(event_in_list)
        self.event_buffer_center_cross = new_event_buffer

        # Check completion
        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"🎯 Collected {self.max_net_speeds_to_collect} net speeds for {self.current_player_id} ({self.current_test_mode}). Generating output...")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION: 
                self.is_counting_active = False

    def _commit_event(self, event):
        """Helper method to commit an event"""
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

            event_type = "PREDICTED" if event.predicted else "ACTUAL"
            if self.debug_mode: 
                print(f"--- COMMITTED {event_type} Event #{len(self.collected_net_speeds)}: Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. New cooldown starts from {event.timestamp:.3f} ---")
            event.processed = True

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return
            
        p1_glob = self.trajectory[-2]
        p2_glob = self.trajectory[-1]
        
        dist_cm = self._calculate_real_distance_cm_global(p1_glob.x, p1_glob.y, p2_glob.x, p2_glob.y)
        delta_t = p2_glob.t - p1_glob.t
        
        if delta_t > 0.0001:
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = ((1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + 
                                              SPEED_SMOOTHING_FACTOR * speed_kmh)
            else: 
                self.current_ball_speed_kmh = speed_kmh
        else: 
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
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

    def _generate_outputs_async(self):
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        """Optimized file creation with batch operations and custom naming"""
        if not net_speeds: 
            return
            
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
        os.makedirs(output_dir_path, exist_ok=True)

        # Create custom filename base
        filename_base = f"speed_data_{self.current_player_id}_{self.current_test_mode}_{timestamp_str}"

        avg_speed = sum(net_speeds) / len(net_speeds)
        max_speed = max(net_speeds)
        min_speed = min(net_speeds)

        # Create chart with custom filename
        chart_filename = f'{output_dir_path}/{filename_base}.png'
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        
        for t, s in zip(relative_times, net_speeds): 
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # Updated chart title with player and mode info
        plt.title(f'Net Crossing Speeds - {self.current_player_id} ({self.current_test_mode})\nSession {session_id} - {timestamp_str}', fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        
        if relative_times:
            x_margin = (max(relative_times) - min(relative_times)) * 0.05 if len(relative_times) > 1 and max(relative_times) > min(relative_times) else 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
        if net_speeds:
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(max(0, min_speed - y_range*0.1), max_speed + y_range*0.1)
        
        plt.figtext(0.02, 0.02, f"Player: {self.current_player_id} | Mode: {self.current_test_mode} | Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_filename, dpi=150)
        plt.close()

        # Batch write text and CSV files with custom filenames
        txt_filename = f'{output_dir_path}/{filename_base}.txt'
        csv_filename = f'{output_dir_path}/{filename_base}.csv'
        
        # Prepare text content with player and mode info
        txt_content = []
        txt_content.append(f"Net Speeds - Player: {self.current_player_id} | Mode: {self.current_test_mode}\n")
        txt_content.append(f"Session {session_id} - {timestamp_str}\n")
        txt_content.append("="*60 + "\n")
        for i, (t, s) in enumerate(zip(relative_times, net_speeds)): 
            txt_content.append(f"{t:.2f}s: {s:.1f} km/h\n")
        txt_content.append("="*60 + "\n")
        txt_content.append(f"Player ID: {self.current_player_id}\n")
        txt_content.append(f"Test Mode: {self.current_test_mode}\n")
        txt_content.append(f"Session ID: {session_id}\n")
        txt_content.append(f"Total Points: {len(net_speeds)}\n")
        txt_content.append(f"Average Speed: {avg_speed:.1f} km/h\n")
        txt_content.append(f"Maximum Speed: {max_speed:.1f} km/h\n")
        txt_content.append(f"Minimum Speed: {min_speed:.1f} km/h\n")
        
        # Write files in batch
        with open(txt_filename, 'w') as f:
            f.writelines(txt_content)

        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Enhanced CSV header with player and mode info
            writer.writerow(['Player ID', 'Test Mode', 'Session ID', 'Timestamp File', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): 
                writer.writerow([self.current_player_id, self.current_test_mode, session_id, timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Player ID', self.current_player_id])
            writer.writerow(['Test Mode', self.current_test_mode])
            writer.writerow(['Session ID', session_id])
            writer.writerow(['Total Points', len(net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        
        print(f"📁 Output files saved to {output_dir_path}:")
        print(f"   📊 Chart: {filename_base}.png")
        print(f"   📄 Text:  {filename_base}.txt")
        print(f"   📈 CSV:   {filename_base}.csv")
        print(f"🏓 Player: {self.current_player_id} | Mode: {self.current_test_mode}")

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        """Optimized visualization drawing"""
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
            cx_global_vis = cx_roi + self.roi_start_x
            cy_global_vis = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global_vis, cy_global_vis), 8, BALL_COLOR_BGR, -1)
        
        # Batch text rendering for better performance
        text_items = [
            (f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), SPEED_TEXT_COLOR_BGR),
            (f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), FPS_TEXT_COLOR_BGR),
            (f"Counting: {'ON' if frame_data_obj.is_counting_active else 'OFF'}", (10, 110), 
             (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)),
            (f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, 150), NET_SPEED_TEXT_COLOR_BGR),
            (self.instruction_text, (10, self.frame_height - 20), (255, 255, 255))
        ]
        
        # Add player info if available
        if self.current_player_id:
            text_items.insert(-1, (f"Player: {self.current_player_id} | Mode: {self.current_test_mode}", (10, 190), (255, 255, 0)))
            
        if frame_data_obj.last_recorded_net_speed_kmh > 0:
            text_items.insert(-2, (f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 230), NET_SPEED_TEXT_COLOR_BGR))
        
        if frame_data_obj.collected_relative_times:
            text_items.insert(-2, (f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 270), NET_SPEED_TEXT_COLOR_BGR))
        
        if self.debug_mode and frame_data_obj.debug_display_text:
            text_items.insert(-1, (frame_data_obj.debug_display_text, (10, 310), (200, 200, 0)))
        
        # Batch render all text
        for text, pos, color in text_items:
            font_scale = 0.7 if pos == (10, self.frame_height - 20) else FONT_SCALE_VIS
            thickness = 1 if pos == (10, self.frame_height - 20) else FONT_THICKNESS_VIS
            cv2.putText(vis_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        motion_mask_roi = self._detect_fmo_vectorized()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_global_coords, ball_timestamp = None, None

        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi, ball_global_coords, ball_timestamp = self._detect_ball_in_roi(motion_mask_roi)
            if ball_pos_in_roi:
                self._calculate_ball_speed()
                if self.is_counting_active:
                    self._record_potential_crossing(ball_global_coords[0], ball_global_coords[1], ball_timestamp)
            else:
                self.last_ball_x_global = None
        else:
            self.last_ball_x_global = None

        self._check_timeout_and_reset()
        
        if self.is_counting_active:
            self._process_crossing_events()

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
        
        if len(self.trajectory) > 0: 
            frame_data.trajectory_points_global = self.trajectory.get_recent_points(50)  # Limit points for performance
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker (v11 OPTIMIZED FOR M2 PRO) ===")
        print("🚀 Enhanced with Player ID and Test Mode tracking")
        print("📝 Press SPACE to start counting - you'll be prompted for:")
        print("   • Player ID (e.g., 'John_Doe', 'Player1')")  
        print("   • Test Mode (e.g., 'forehand', 'backhand', 'serve')")
        print("")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction} (Focus on Right-to-Left)")
        print(f"Target speeds to collect: {self.max_net_speeds_to_collect}")
        print(f"Effective Crossing Cooldown: {self.EFFECTIVE_CROSSING_COOLDOWN_S}s")
        if self.debug_mode: 
            print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker v11 (OPTIMIZED)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: 
                        print("Video ended or frame read error.")
                    else: 
                        print("Camera error or stream ended.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("End of stream with pending data. Generating output.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                
                # Use async visualization
                display_frame = self.async_visualizer.update_visualization_async(
                    frame_data_obj.frame, frame_data_obj, self._draw_visualizations
                )
                
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
            
            # Cleanup optimized components
            self.batch_writer.flush()
            self.async_visualizer.shutdown()
            self.pipeline_processor.shutdown()
            self.file_writer_executor.shutdown(wait=True)
            print("File writer stopped.")
            
            cv2.destroyAllWindows()
            print("System shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker v11 (OPTIMIZED FOR M2 PRO)')
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
        target_fps=args.fps, frame_width=args.width, frame_height=args.height,
        debug_mode=args.debug, net_crossing_direction=args.direction,
        max_net_speeds=args.count, near_width_cm=args.near_width, far_width_cm=args.far_width
    )
    tracker.run()

if __name__ == '__main__':
    main()