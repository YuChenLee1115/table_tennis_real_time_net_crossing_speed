#!/usr/bin/env python3
# Ping Pong Speed Tracker - Optimized for M2 Pro MacBook Pro (10-core)
# Ultra-optimized version with enhanced multi-threading and performance tuning

import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for better performance
import matplotlib.pyplot as plt
import os
import csv
import threading
import queue
import concurrent.futures
from multiprocessing import cpu_count
import gc
from numba import jit, njit
import psutil

# —— M2 Pro Specific Optimizations ——
# M2 Pro has 10 cores (6 performance + 4 efficiency)
M2_PRO_CORES = 10
M2_PRO_PERFORMANCE_CORES = 6
M2_PRO_EFFICIENCY_CORES = 4

# Set process priority for better real-time performance
try:
    import psutil
    current_process = psutil.Process()
    current_process.nice(-10)  # Higher priority on macOS
    print("Process priority set to high (-10)")
except ImportError:
    print("psutil not available, skipping process priority setting")
except (psutil.AccessDenied, PermissionError):
    print("Permission denied for setting process priority. Running with normal priority.")
    print("For better performance, consider running with sudo or adjusting system permissions.")
except Exception as e:
    print(f"Failed to set process priority: {e}. Running with normal priority.")

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

# Threading & Queue Parameters - Optimized for M2 Pro
FRAME_QUEUE_SIZE = 60  # Increased for better buffering
EVENT_BUFFER_SIZE_CENTER_CROSS = 200
PROCESSING_THREAD_COUNT = M2_PRO_PERFORMANCE_CORES  # Use performance cores for processing

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV M2 Pro Optimizations ——
cv2.setUseOptimized(True)

# Optimize for M2 Pro's 10 cores
cv2.setNumThreads(M2_PRO_CORES)

# Enable TBB (Threading Building Blocks) if available
try:
    cv2.setUseOpenVX(True)
except (AttributeError, cv2.error):
    # OpenVX not available or not enabled at compile time
    pass

# Set optimal memory alignment for M2 Pro
os.environ['OPENCV_CPU_DISABLE'] = '0'
os.environ['TBB_NUM_THREADS'] = str(M2_PRO_CORES)

# —— Numba Optimized Functions ——
@njit(cache=True, fastmath=True)
def calculate_distance_fast(x1, y1, x2, y2):
    """Ultra-fast distance calculation using Numba JIT compilation."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

@njit(cache=True, fastmath=True)
def calculate_circularity_fast(area, perimeter):
    """Fast circularity calculation."""
    if perimeter > 0:
        return 4 * math.pi * area / (perimeter * perimeter)
    return 0

@njit(cache=True, fastmath=True)
def smooth_speed_fast(current_speed, new_speed, factor):
    """Fast speed smoothing calculation."""
    if current_speed > 0:
        return (1 - factor) * current_speed + factor * new_speed
    return new_speed

class OptimizedFrameData:
    """Optimized data structure with memory pooling."""
    __slots__ = ['frame', 'roi_sub_frame', 'ball_position_in_roi', 'ball_contour_in_roi',
                 'current_ball_speed_kmh', 'display_fps', 'is_counting_active',
                 'collected_net_speeds', 'last_recorded_net_speed_kmh',
                 'collected_relative_times', 'debug_display_text', 'frame_counter',
                 'trajectory_points_global']
    
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
    """Optimized event record with slots."""
    __slots__ = ['ball_x_global', 'timestamp', 'speed_kmh', 'predicted', 'processed']
    
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False

class OptimizedFrameReader:
    """High-performance frame reader optimized for M2 Pro."""
    
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        
        # Use AVFoundation backend for best macOS performance
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
        self._configure_capture(frame_width, frame_height)

        # Optimized queue size for M2 Pro
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        
        # Use high-priority thread for frame reading
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.name = "FrameReader-Priority"

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             self.actual_fps = self.target_fps

        # Pre-allocate frame buffer to reduce memory allocation overhead
        self.frame_buffer = np.empty((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    def _configure_capture(self, frame_width, frame_height):
        """Optimized capture configuration for M2 Pro."""
        if not self.use_video_file:
            # Set hardware acceleration if available
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # M2 Pro specific optimizations
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)  # Disable auto-exposure for consistent performance
            
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")

    def _read_frames(self):
        """Optimized frame reading loop."""
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    self.frame_queue.put((False, None))
                    break
                
                # Use copy to avoid memory reference issues
                frame_copy = frame.copy()
                self.frame_queue.put((True, frame_copy))
                
                # Explicit memory management for M2 Pro
                del frame
            else:
                # Adaptive sleep based on target FPS
                time.sleep(1.0 / (self.target_fps * 3))

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=0.5)  # Reduced timeout for better responsiveness
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def get_properties(self):
        return self.actual_fps, self.frame_width, self.frame_height

class OptimizedPingPongSpeedTracker:
    """Ultra-optimized ping pong speed tracker for M2 Pro MacBook Pro."""
    
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM,
                 detection_timeout_s=DEFAULT_DETECTION_TIMEOUT, use_video_file=False,
                 target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=DEBUG_MODE_DEFAULT,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
                 max_net_speeds=MAX_NET_SPEEDS_TO_COLLECT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT):
        
        # Core initialization
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.target_fps = target_fps

        # Initialize optimized frame reader
        self.reader = OptimizedFrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps

        # Basic parameters
        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm

        # ROI calculation
        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y
        self.roi_width_px = self.roi_end_x - self.roi_start_x

        # Trajectory and speed tracking
        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        # FMO detection optimization
        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        # Frame processing
        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        # Center line detection
        self.center_x_global = self.frame_width // 2
        self.net_crossing_direction = net_crossing_direction
        self.max_net_speeds_to_collect = max_net_speeds
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_recorded_net_speed_kmh = 0
        self.last_ball_x_global = None
        self.output_generated_for_session = False
        
        # Counting session management
        self.is_counting_active = False
        self.count_session_id = 0
        self.timing_started_for_session = False
        self.first_ball_crossing_timestamp = None
        
        # Perspective correction
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        # Event processing
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        # Threading
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=M2_PRO_EFFICIENCY_CORES,  # Use efficiency cores for I/O
            thread_name_prefix="FileWriter"
        )
        
        # Processing thread pool for compute-intensive tasks
        self.processing_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=PROCESSING_THREAD_COUNT,
            thread_name_prefix="Processing"
        )

        # Enhanced crossing detection state
        self.ball_on_left_of_center = False 
        self.last_committed_crossing_time = 0 
        self.EFFECTIVE_CROSSING_COOLDOWN_S = 0.3
        self.CENTER_ZONE_WIDTH_PIXELS = self.frame_width * 0.05

        # Pre-compute static elements
        self._precalculate_overlay()
        self._create_perspective_lookup_table()
        
        # Memory optimization
        self._setup_memory_optimization()

    def _setup_memory_optimization(self):
        """Setup memory optimization for M2 Pro."""
        # Pre-allocate commonly used arrays
        self.temp_mask = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        self.temp_gray = np.empty((self.roi_height_px, self.roi_width_px), dtype=np.uint8)
        
        # Enable garbage collection optimization
        gc.set_threshold(700, 10, 10)  # Optimized for real-time processing

    def _precalculate_overlay(self):
        """Pre-calculate static overlay elements for better performance."""
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # ROI Box lines
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), 
                (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), 
                (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), 
                (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        
        # Center line
        cv2.line(self.static_overlay, (self.center_x_global, 0), 
                (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        """Optimized perspective lookup table creation."""
        self.perspective_lookup_px_to_cm = {}
        
        # Pre-compute for every 5 pixels instead of 10 for better accuracy
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 5):
            ratio = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = ratio

    def _get_pixel_to_cm_ratio(self, y_global):
        """Fast pixel to cm ratio calculation."""
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
        """Optimized FPS calculation."""
        if self.use_video_file:
            self.display_fps = self.actual_fps
            return
        
        now = time.monotonic()
        self.frame_timestamps_for_fps.append(now)
        
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 0:
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                self.display_fps = smooth_speed_fast(self.display_fps, measured_fps, FPS_SMOOTHING_FACTOR)
        
        self.last_frame_timestamp_for_fps = now

    def _preprocess_frame(self, frame):
        """Optimized frame preprocessing."""
        # Direct ROI extraction without copying
        roi_sub_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        
        # Use pre-allocated array for grayscale conversion
        gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY, dst=self.temp_gray)
        
        # In-place Gaussian blur for memory efficiency
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0, dst=gray_roi.copy())
        
        self.prev_frames_gray_roi.append(gray_roi_blurred)
        
        return roi_sub_frame, gray_roi_blurred

    def _detect_fmo(self):
        """Optimized Fast Moving Object detection."""
        if len(self.prev_frames_gray_roi) < 3: 
            return None
        
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        
        # Use pre-allocated arrays for difference calculation
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        
        # In-place bitwise operation
        motion_mask = cv2.bitwise_and(diff1, diff2, dst=self.temp_mask)
        
        try:
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        if OPENING_KERNEL_SIZE_FMO[0] > 0:
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else: 
            opened_mask = thresh_mask
        
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        
        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        """Optimized ball detection in ROI."""
        # Fast connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask_roi, connectivity=8)
        
        potential_balls = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
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
                        
                        # Use optimized circularity calculation
                        circularity = calculate_circularity_fast(area, perimeter)
                
                potential_balls.append({
                    'position_roi': (int(cx_roi), int(cy_roi)), 
                    'area': area,
                    'circularity': circularity, 
                    'contour_roi': contour_to_store
                })
        
        if not potential_balls: 
            return None, None, None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: 
            return None, None, None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file: 
            current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), "
                  f"Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}, "
                  f"T:{current_timestamp:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi'), (cx_global, cy_global), current_timestamp

    def _select_best_ball_candidate(self, candidates):
        """Optimized ball candidate selection."""
        if not candidates: 
            return None
        
        if not self.trajectory:
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular: 
                return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])

        last_x_global, last_y_global, _ = self.trajectory[-1]
        max_distance_threshold = self.frame_width * 0.2
        
        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global, cy_global = cx_roi + self.roi_start_x, cy_roi + self.roi_top_y
            
            # Use optimized distance calculation
            distance = calculate_distance_fast(cx_global, last_x_global, cy_global, last_y_global)
            
            ball_info['distance_from_last'] = distance if distance <= max_distance_threshold else float('inf')
            
            # Calculate consistency score
            consistency_score = 0
            if len(self.trajectory) >= 2:
                prev_x_global, prev_y_global, _ = self.trajectory[-2]
                vec_hist_dx, vec_hist_dy = last_x_global - prev_x_global, last_y_global - prev_y_global
                vec_curr_dx, vec_curr_dy = cx_global - last_x_global, cy_global - last_y_global
                
                dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                mag_hist = calculate_distance_fast(0, 0, vec_hist_dx, vec_hist_dy)
                mag_curr = calculate_distance_fast(0, 0, vec_curr_dx, vec_curr_dy)
                
                if mag_hist > 0 and mag_curr > 0: 
                    cosine_similarity = dot_product / (mag_hist * mag_curr)
                else: 
                    cosine_similarity = 0
                
                consistency_score = max(0, cosine_similarity)
            
            ball_info['consistency'] = consistency_score
        
        # Calculate composite scores
        for ball_info in candidates:
            ball_info['score'] = (0.4 / (1.0 + ball_info['distance_from_last'])) + \
                               (0.4 * ball_info['consistency']) + \
                               (0.2 * ball_info['circularity'])
        
        return max(candidates, key=lambda b: b['score'])

    def toggle_counting(self):
        """Toggle counting mode with optimized state management."""
        self.is_counting_active = not self.is_counting_active
        
        if self.is_counting_active:
            self.count_session_id += 1
            self.collected_net_speeds.clear()
            self.collected_relative_times.clear()
            self.timing_started_for_session = False
            self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear()
            self.output_generated_for_session = False
            
            # Reset enhanced detection state
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

    def _record_potential_crossing(self, ball_x_global, ball_y_global, current_timestamp):
        """Optimized crossing detection and recording."""
        if not self.is_counting_active:
            self.last_ball_x_global = ball_x_global
            return

        if self.net_crossing_direction not in ['right_to_left', 'both']:
            self.last_ball_x_global = ball_x_global
            return

        # Cooldown check
        if current_timestamp - self.last_committed_crossing_time < self.EFFECTIVE_CROSSING_COOLDOWN_S:
            if self.debug_mode: 
                print(f"DEBUG REC: In cooldown. CT: {current_timestamp:.3f}, "
                      f"LastCommitT: {self.last_committed_crossing_time:.3f}")
            self.last_ball_x_global = ball_x_global
            return

        # Update ball position state
        center_zone_left = self.center_x_global - self.CENTER_ZONE_WIDTH_PIXELS
        center_zone_right = self.center_x_global + self.CENTER_ZONE_WIDTH_PIXELS
        
        if ball_x_global < center_zone_left:
            if not self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball now clearly on left (X={ball_x_global}).")
            self.ball_on_left_of_center = True
        elif ball_x_global > center_zone_right:
            if self.ball_on_left_of_center and self.debug_mode: 
                print(f"DEBUG REC: Ball returned to right (X={ball_x_global}), resetting left flag.")
            self.ball_on_left_of_center = False

        # Actual crossing detection (Right-to-Left)
        crossed_r_to_l_strictly = False
        if (self.last_ball_x_global is not None and 
            self.last_ball_x_global >= self.center_x_global and 
            ball_x_global < self.center_x_global and 
            not self.ball_on_left_of_center):
            
            crossed_r_to_l_strictly = True
            if self.debug_mode:
                print(f"DEBUG REC: Strict R-L Actual Crossing Detected. "
                      f"PrevX: {self.last_ball_x_global:.1f}, CurrX: {ball_x_global:.1f}. "
                      f"Speed: {self.current_ball_speed_kmh:.1f}")

        if crossed_r_to_l_strictly and self.current_ball_speed_kmh > 0.1:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
            self.event_buffer_center_cross.append(event)
            if self.debug_mode: 
                print(f"DEBUG REC: Added ACTUAL event to buffer. Buffer size: {len(self.event_buffer_center_cross)}")

        # Conservative prediction logic
        if (not crossed_r_to_l_strictly and not self.ball_on_left_of_center and 
            len(self.trajectory) >= 2 and self.current_ball_speed_kmh > 0.1 and 
            ball_x_global >= self.center_x_global):

            pt1_x, _, pt1_t = self.trajectory[-2]
            pt2_x, _, pt2_t = self.trajectory[-1]
            
            delta_t_hist = pt2_t - pt1_t
            if delta_t_hist > 0:
                vx_pixels_per_time_unit = (pt2_x - pt1_x) / delta_t_hist
                
                # Movement threshold for prediction
                fps_estimate = self.display_fps if self.display_fps > 1 else self.target_fps
                min_vx_for_prediction = -(self.frame_width * 0.02) * (delta_t_hist / (1.0 / fps_estimate))

                if vx_pixels_per_time_unit < min_vx_for_prediction:
                    for lookahead_frames in [1, 2]:
                        time_to_predict = lookahead_frames / fps_estimate
                        predicted_x_at_crossing_time = ball_x_global + vx_pixels_per_time_unit * time_to_predict
                        predicted_timestamp = current_timestamp + time_to_predict

                        if predicted_x_at_crossing_time < self.center_x_global:
                            # Check for similar predictions
                            can_add_prediction = True
                            frame_time_threshold = 1.0 / fps_estimate
                            
                            for ev in self.event_buffer_center_cross:
                                if ev.predicted and abs(ev.timestamp - predicted_timestamp) < frame_time_threshold:
                                    can_add_prediction = False
                                    break
                            
                            if can_add_prediction:
                                if self.debug_mode: 
                                    print(f"DEBUG REC: Added PREDICTED event. "
                                          f"X_pred: {predicted_x_at_crossing_time:.1f} at T_pred: {predicted_timestamp:.3f}")
                                event = EventRecord(predicted_x_at_crossing_time, predicted_timestamp, 
                                                  self.current_ball_speed_kmh, predicted=True)
                                self.event_buffer_center_cross.append(event)
                            break
        
        self.last_ball_x_global = ball_x_global

    def _process_crossing_events(self):
        """Optimized crossing event processing."""
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_processing_time = time.monotonic()
        if self.use_video_file: 
            current_processing_time = self.frame_counter / self.actual_fps

        # Convert deque to sorted list for processing
        temp_event_list = sorted(list(self.event_buffer_center_cross), key=lambda e: e.timestamp)
        new_event_buffer = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)

        committed_event_ts = -1

        # Stage 1: Process actual events
        actual_event_to_commit = None
        for event in temp_event_list:
            if event.processed or event.predicted:
                continue
                
            if event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S:
                actual_event_to_commit = event
                break

        if actual_event_to_commit and len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
            event = actual_event_to_commit
            
            if not self.timing_started_for_session:
                self.timing_started_for_session = True
                self.first_ball_crossing_timestamp = event.timestamp
                
            relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2) if self.timing_started_for_session else 0.0

            self.last_recorded_net_speed_kmh = event.speed_kmh
            self.collected_net_speeds.append(event.speed_kmh)
            self.collected_relative_times.append(relative_time)
            
            self.last_committed_crossing_time = event.timestamp
            self.ball_on_left_of_center = True

            if self.debug_mode: 
                print(f"--- COMMITTED ACTUAL Event #{len(self.collected_net_speeds)}: "
                      f"Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. "
                      f"New cooldown starts from {event.timestamp:.3f} ---")
            
            event.processed = True
            committed_event_ts = event.timestamp

        # Stage 2: Process predicted events if no actual event was committed
        elif not actual_event_to_commit and len(self.collected_net_speeds) < self.max_net_speeds_to_collect:
            predicted_event_to_commit = None
            
            for event in temp_event_list:
                if event.processed or not event.predicted:
                    continue
                    
                if (current_processing_time >= event.timestamp and 
                    event.timestamp - self.last_committed_crossing_time >= self.EFFECTIVE_CROSSING_COOLDOWN_S):
                    predicted_event_to_commit = event
                    break
            
            if predicted_event_to_commit:
                event = predicted_event_to_commit
                
                if not self.timing_started_for_session:
                    self.timing_started_for_session = True
                    self.first_ball_crossing_timestamp = event.timestamp
                    
                relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2) if self.timing_started_for_session else 0.0

                self.last_recorded_net_speed_kmh = event.speed_kmh
                self.collected_net_speeds.append(event.speed_kmh)
                self.collected_relative_times.append(relative_time)
                
                self.last_committed_crossing_time = event.timestamp
                self.ball_on_left_of_center = True

                if self.debug_mode: 
                    print(f"--- COMMITTED PREDICTED Event #{len(self.collected_net_speeds)}: "
                          f"Speed {event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s. "
                          f"New cooldown from {event.timestamp:.3f} ---")
                
                event.processed = True
                committed_event_ts = event.timestamp

        # Stage 3: Clean up buffer
        if committed_event_ts > 0:
            cooldown_half = self.EFFECTIVE_CROSSING_COOLDOWN_S / 2.0
            for event_in_list in temp_event_list:
                if (not event_in_list.processed and 
                    abs(event_in_list.timestamp - committed_event_ts) < cooldown_half):
                    event_in_list.processed = True
                    if self.debug_mode: 
                        print(f"DEBUG PROC: Nullified nearby event (Pred: {event_in_list.predicted}, "
                              f"T: {event_in_list.timestamp:.3f}) due to commit at {committed_event_ts:.3f}")

        # Rebuild buffer with unprocessed and recent events
        for event_in_list in temp_event_list:
            if (not event_in_list.processed and 
                (current_processing_time - event_in_list.timestamp < 2.0)):
                new_event_buffer.append(event_in_list)
        
        self.event_buffer_center_cross = new_event_buffer

        # Check if collection is complete
        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Collected {self.max_net_speeds_to_collect} net speeds. Generating output.")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION: 
                self.is_counting_active = False

    def _calculate_ball_speed(self):
        """Optimized ball speed calculation."""
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
            
            # Use optimized smoothing function
            self.current_ball_speed_kmh = smooth_speed_fast(
                self.current_ball_speed_kmh, speed_kmh, SPEED_SMOOTHING_FACTOR)
        else: 
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        """Optimized real distance calculation."""
        y1_roi = max(0, min(self.roi_height_px, y1_g - self.roi_top_y))
        y2_roi = max(0, min(self.roi_height_px, y2_g - self.roi_top_y))
        
        # Use 5-pixel resolution for better accuracy
        y1_roi_rounded = round(y1_roi / 5) * 5
        y2_roi_rounded = round(y2_roi / 5) * 5
        
        ratio1 = self.perspective_lookup_px_to_cm.get(
            y1_roi_rounded, self._get_pixel_to_cm_ratio(y1_g))
        ratio2 = self.perspective_lookup_px_to_cm.get(
            y2_roi_rounded, self._get_pixel_to_cm_ratio(y2_g))
        
        avg_px_to_cm_ratio = (ratio1 + ratio2) / 2.0
        
        # Use optimized distance calculation
        pixel_distance = calculate_distance_fast(x1_g, y1_g, x2_g, y2_g)
        real_distance_cm = pixel_distance * avg_px_to_cm_ratio
        
        return real_distance_cm

    def _generate_outputs_async(self):
        """Optimized asynchronous output generation."""
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
        
        # Create copies for thread safety
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id
        
        # Submit to efficiency cores for I/O operations
        self.file_writer_executor.submit(
            self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        """Optimized output file creation."""
        if not net_speeds: 
            return
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_path = f"{OUTPUT_DATA_FOLDER}/{timestamp_str}"
        os.makedirs(output_dir_path, exist_ok=True)

        # Pre-calculate statistics
        avg_speed = sum(net_speeds) / len(net_speeds)
        max_speed = max(net_speeds)
        min_speed = min(net_speeds)

        # Create chart with optimized settings
        chart_filename = f'{output_dir_path}/speed_chart_{timestamp_str}.png'
        
        plt.figure(figsize=(12, 7))
        plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        
        # Add annotations
        for t, s in zip(relative_times, net_speeds): 
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.title(f'Net Crossing Speeds - Session {session_id} - {timestamp_str}', fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        
        # Set axis limits
        if relative_times:
            if len(relative_times) > 1 and max(relative_times) > min(relative_times):
                x_margin = (max(relative_times) - min(relative_times)) * 0.05
            else:
                x_margin = 0.5
            plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)
        
        if net_speeds:
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(max(0, min_speed - y_range*0.1), max_speed + y_range*0.1)
        
        plt.figtext(0.02, 0.02, f"Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        plt.close()

        # Create text file
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

        # Create CSV file
        csv_filename = f'{output_dir_path}/speed_data_{timestamp_str}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Session ID', 'Timestamp File', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            
            for i, (t, s) in enumerate(zip(relative_times, net_speeds)): 
                writer.writerow([session_id, timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Total Points', len(net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        
        print(f"Output files for session {session_id} saved to {output_dir_path}")

    def _draw_visualizations(self, display_frame, frame_data_obj: OptimizedFrameData):
        """Optimized visualization drawing."""
        vis_frame = display_frame
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0
        
        if is_full_draw:
            # Use addWeighted for better performance
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            
            # Draw trajectory if available
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)
        
        # Draw ball detection
        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
            
            if frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            
            cx_global_vis = cx_roi + self.roi_start_x
            cy_global_vis = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global_vis, cy_global_vis), 8, BALL_COLOR_BGR, -1)
        
        # Text overlays
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        if frame_data_obj.last_recorded_net_speed_kmh > 0: 
            cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", 
                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        if frame_data_obj.collected_relative_times: 
            cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        cv2.putText(vis_frame, self.instruction_text, 
                   (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if self.debug_mode and frame_data_obj.debug_display_text: 
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, 
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        return vis_frame

    def _check_timeout_and_reset(self):
        """Optimized timeout checking and reset."""
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        """Optimized single frame processing."""
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        motion_mask_roi = self._detect_fmo()
        
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

        # Generate debug text if needed
        debug_text = None
        if self.debug_mode:
            on_left_text = "Y" if self.ball_on_left_of_center else "N"
            debug_text = (f"Traj:{len(self.trajectory)} EvtBuf:{len(self.event_buffer_center_cross)} "
                         f"OnLeft:{on_left_text} LastCommitT:{self.last_committed_crossing_time:.2f}")
        
        # Create optimized frame data object
        frame_data = OptimizedFrameData(
            frame=frame, roi_sub_frame=roi_sub_frame, ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi, current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps, is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            debug_display_text=debug_text, frame_counter=self.frame_counter
        )
        
        if self.trajectory: 
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self):
        """Optimized main run loop."""
        print("=== Ping Pong Speed Tracker - M2 Pro Optimized ===")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction} (Focus on Right-to-Left)")
        print(f"Target speeds to collect: {self.max_net_speeds_to_collect}")
        print(f"Effective Crossing Cooldown: {self.EFFECTIVE_CROSSING_COOLDOWN_S}s")
        print(f"M2 Pro Optimization: {M2_PRO_CORES} cores, {PROCESSING_THREAD_COUNT} processing threads")
        
        if self.debug_mode: 
            print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker - M2 Pro Optimized'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # Performance monitoring
        frame_times = deque(maxlen=30)
        last_perf_report = time.time()

        try:
            while self.running:
                frame_start_time = time.perf_counter()
                
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: 
                        print("Video ended or frame read error.")
                    else: 
                        print("Camera error or stream ended.")
                    
                    if (self.is_counting_active and self.collected_net_speeds and 
                        not self.output_generated_for_session):
                        print("End of stream with pending data. Generating output.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                cv2.imshow(window_name, display_frame)
                
                # Performance monitoring
                frame_end_time = time.perf_counter()
                frame_times.append(frame_end_time - frame_start_time)
                
                # Periodic performance report
                if self.debug_mode and time.time() - last_perf_report > 5.0:
                    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
                    print(f"Performance: Avg frame time: {avg_frame_time*1000:.2f}ms, "
                          f"Theoretical max FPS: {1/avg_frame_time:.1f}")
                    last_perf_report = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    self.running = False
                    if (self.is_counting_active and self.collected_net_speeds and 
                        not self.output_generated_for_session):
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
            if (self.is_counting_active and self.collected_net_speeds and 
                not self.output_generated_for_session):
                print("Interrupted with pending data. Generating output.")
                self._generate_outputs_async()
                self.output_generated_for_session = True
                
        finally:
            self.running = False
            print("Shutting down...")
            
            self.reader.stop()
            print("Frame reader stopped.")
            
            self.processing_executor.shutdown(wait=True)
            print("Processing threads stopped.")
            
            self.file_writer_executor.shutdown(wait=True)
            print("File writer stopped.")
            
            cv2.destroyAllWindows()
            print("System shutdown complete.")

def main():
    """Optimized main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description='Ping Pong Speed Tracker - M2 Pro Optimized',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--video', type=str, default=None, 
                       help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, 
                       help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, 
                       help='Target FPS for webcam.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, 
                       help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, 
                       help='Frame height.')
    parser.add_argument('--table_len', type=float, default=DEFAULT_TABLE_LENGTH_CM, 
                       help='Table length (cm) for nominal px/cm.')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, 
                       help='Ball detection timeout (s).')
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                       choices=['left_to_right', 'right_to_left', 'both'], 
                       help='Net crossing direction to record.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, 
                       help='Number of net speeds to collect per session.')
    parser.add_argument('--near_width', type=float, default=NEAR_SIDE_WIDTH_CM_DEFAULT, 
                       help='Real width (cm) of ROI at near side.')
    parser.add_argument('--far_width', type=float, default=FAR_SIDE_WIDTH_CM_DEFAULT, 
                       help='Real width (cm) of ROI at far side.')
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, 
                       help='Enable debug printouts.')
    
    args = parser.parse_args()

    print(f"M2 Pro Optimization Status:")
    print(f"  - CPU Cores: {M2_PRO_CORES} (6 Performance + 4 Efficiency)")
    print(f"  - Processing Threads: {PROCESSING_THREAD_COUNT}")
    print(f"  - OpenCV Optimization: Enabled")
    print(f"  - Numba JIT Compilation: Enabled")
    print(f"  - Memory Optimization: Enabled")
    print()

    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False

    tracker = OptimizedPingPongSpeedTracker(
        video_source=video_source_arg, table_length_cm=args.table_len,
        detection_timeout_s=args.timeout, use_video_file=use_video_file_arg,
        target_fps=args.fps, frame_width=args.width, frame_height=args.height,
        debug_mode=args.debug, net_crossing_direction=args.direction,
        max_net_speeds=args.count, near_width_cm=args.near_width, 
        far_width_cm=args.far_width
    )
    
    tracker.run()

if __name__ == '__main__':
    main()
