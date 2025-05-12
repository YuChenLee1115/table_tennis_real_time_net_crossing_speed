#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11.2 (Restored Annotations, Improved Timeout Handling)
# Lightweight, optimized, multi-threaded (acquisition & I/O), macOS compatible
# Added interpolation for crossing detection and multi-point speed calculation.
# Waits indefinitely for webcam frames instead of shutting down on temporary timeout.
# Restored point annotations on the output chart.

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
import time # Explicitly import time for perf_counter

# —— Global Parameter Configuration ——
# Basic Settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 120 # Target FPS for webcam setup
DEFAULT_FRAME_WIDTH = 1280 # Reduced default width for performance
DEFAULT_FRAME_HEIGHT = 720 # Reduced default height for performance
DEFAULT_TABLE_LENGTH_CM = 142

# Detection Parameters
DEFAULT_DETECTION_TIMEOUT = 0.2 # Timeout for resetting trajectory if ball not seen
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.8
MAX_TRAJECTORY_POINTS = 120

# Center Line Detection
CENTER_LINE_WIDTH_PIXELS = 55 # Width of the central detection zone
CENTER_DETECTION_COOLDOWN_S = 0.01 # Cooldown between consecutive net crossing detections
MAX_NET_SPEEDS_TO_COLLECT = 27
NET_CROSSING_DIRECTION_DEFAULT = 'left_to_right' # 'left_to_right', 'right_to_left', 'both'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output'

# Perspective Correction
NEAR_SIDE_WIDTH_CM_DEFAULT = 29
FAR_SIDE_WIDTH_CM_DEFAULT = 72

# FMO (Fast Moving Object) Parameters
MAX_PREV_FRAMES_FMO = 10
OPENING_KERNEL_SIZE_FMO = (10, 10) # Consider reducing if performance is an issue
CLOSING_KERNEL_SIZE_FMO = (25, 25) # Consider reducing if performance is an issue
THRESHOLD_VALUE_FMO = 8

# Ball Detection Parameters
MIN_BALL_AREA_PX = 5
MAX_BALL_AREA_PX = 10000
MIN_BALL_CIRCULARITY = 0.4
# Speed Calculation
SPEED_SMOOTHING_FACTOR = 0.3
KMH_CONVERSION_FACTOR = 0.036
SPEED_CALC_POINTS = 3 # Number of recent points to use for speed calculation (min 2)

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
FONT_SCALE_VIS = 0.8 # Adjusted font scale for potentially smaller frames
FONT_THICKNESS_VIS = 2
VISUALIZATION_DRAW_INTERVAL = 3 # Draw full visuals every N frames (Increased default)

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 5 # Reduced queue size, might help reduce latency perception
EVENT_BUFFER_SIZE_CENTER_CROSS = 50 # Reduced buffer size
PREDICTION_LOOKAHEAD_FRAMES = 15 # For optional prediction logic

# Debug
DEBUG_MODE_DEFAULT = False

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    num_threads = os.cpu_count()
    if num_threads and num_threads > 1:
        cv2.setNumThreads(num_threads) # Use available cores
    else:
        cv2.setNumThreads(4) # Fallback
except AttributeError:
    cv2.setNumThreads(4) # Default if os.cpu_count fails

class FrameData:
    """Data structure for passing frame-related information."""
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0, profiling_info=""):
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
        self.profiling_info = profiling_info # For debug timing

class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp # Can be interpolated time
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False # Flag to avoid processing multiple times

class FrameReader:
    """Reads frames from camera or video file in a separate thread."""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps # Target FPS for webcam setting
        self.use_video_file = use_video_file
        self.cap = cv2.VideoCapture(self.video_source)
        self._configure_capture(frame_width, frame_height)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        # Get actual properties AFTER configuration attempts
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Requested Res: {frame_width}x{frame_height}, Actual Res: {self.frame_width}x{self.frame_height}")
        print(f"Requested FPS: {target_fps}, Actual FPS from camera: {self.actual_fps:.2f}")

        # If webcam reports unreliable FPS (0 or very high), use target as a fallback for calculations
        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             print(f"Warning: Unreliable FPS ({self.actual_fps}) reported by webcam. Using target FPS ({self.target_fps}) for some calculations.")
             self.display_fps_source = self.target_fps # Base value for display FPS calc
        else:
             self.display_fps_source = self.actual_fps # Use reported FPS from video/reliable webcam

    def _configure_capture(self, frame_width, frame_height):
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")
        if not self.use_video_file:
            # Attempt to set desired properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Some cameras might need specific fourcc codes for high FPS, e.g., 'MJPG'
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def _read_frames(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    # If read fails, could be end of file or camera issue.
                    # Signal the problem but let the main loop decide based on self.running state.
                    print("FrameReader: read() returned False.")
                    # Keep trying to read unless stop() is called (self.running becomes False)
                    # If it's a persistent camera issue, read() will keep returning False.
                    # Put a signal indicating failure? Or just let read() in main loop timeout?
                    # Let's signal failure so main loop knows it's not just empty queue.
                    try:
                        # Put a failure marker, but don't block if queue is full (shouldn't be if read failed)
                        self.frame_queue.put_nowait((False, None))
                    except queue.Full:
                        pass # If queue is full, main loop will find out anyway
                    # Add a small sleep to prevent tight loop on persistent error
                    time.sleep(0.1)
                    # We don't set self.running=False here anymore, stop() method does that.
                else:
                    # Successfully read a frame
                    self.frame_queue.put((True, frame))
            else:
                # Queue is full, sleep briefly to yield CPU
                time.sleep(0.001) # Shorter sleep

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()

    def read(self):
        """Reads from the queue with a timeout."""
        try:
            # Calculate timeout based on expected FPS, with a minimum
            timeout_duration = max(0.001, 1.0 / self.display_fps_source if self.display_fps_source > 0 else 0.1)
            return self.frame_queue.get(timeout=timeout_duration)
        except queue.Empty:
            # This means no frame arrived within the timeout, but doesn't necessarily mean error yet.
            return None, None # Return None, None to indicate timeout

    def stop(self):
        """Stops the reading thread and releases the camera."""
        self.running = False # Signal the thread to stop reading
        if self.thread.is_alive():
            # Clear queue to unblock thread if it's waiting on put()
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.thread.join(timeout=1.0) # Wait briefly for thread to exit
        if self.cap.isOpened():
            self.cap.release()
        print("FrameReader stopped.")

    @property
    def is_running(self):
        """Check if the reading thread is supposed to be running."""
        return self.running

    def get_properties(self):
        return self.display_fps_source, self.frame_width, self.frame_height

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
        # self.target_fps = target_fps # Target FPS stored in FrameReader

        self.reader = FrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        # Use the potentially adjusted FPS source from the reader
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps # Initial display FPS, will be updated

        # --- Rest of __init__ ---
        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        # Nominal pixels_per_cm, less critical with perspective lookup
        # self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0 # ROI starts from top of the frame
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y
        self.roi_width_px = self.roi_end_x - self.roi_start_x

        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.monotonic() # Use monotonic clock for timeout

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

        self.frame_counter = 0
        # self.last_frame_timestamp_for_fps = time.monotonic() # Renamed/replaced by deque logic
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2
        self.center_line_start_x = self.center_x_global - CENTER_LINE_WIDTH_PIXELS // 2
        self.center_line_end_x = self.center_x_global + CENTER_LINE_WIDTH_PIXELS // 2

        self.net_crossing_direction = net_crossing_direction
        self.max_net_speeds_to_collect = max_net_speeds
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_net_crossing_detection_time = 0 # Timestamp of the last recorded crossing event
        self.last_recorded_net_speed_kmh = 0
        self.last_ball_x_global = None
        self.last_ball_timestamp = None # NEW: Timestamp of the last detected ball position

        self.output_generated_for_session = False
        self.is_counting_active = False
        self.count_session_id = 0
        self.timing_started_for_session = False
        self.first_ball_crossing_timestamp = None

        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm

        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)

        self.main_loop_running = False # Renamed from self.running to avoid conflict
        # Increased workers slightly, consider adjusting based on CPU/task nature
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        self._precalculate_overlay()
        self._create_perspective_lookup_table()
        self.last_frame_display = None # Store last good frame for display during waits

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # Draw ROI Box
        cv2.rectangle(self.static_overlay, (self.roi_start_x, self.roi_top_y),
                      (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        # Draw Center Line
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 1)
        # Draw Center Detection Zone (thicker representation)
        center_zone_rect = np.zeros_like(self.static_overlay)
        cv2.rectangle(center_zone_rect, (self.center_line_start_x, 0),
                      (self.center_line_end_x, self.frame_height), CENTER_LINE_COLOR_BGR, -1) # Filled rectangle
        # Blend the center zone with the main overlay
        self.static_overlay = cv2.addWeighted(self.static_overlay, 1.0, center_zone_rect, 0.2, 0) # Adjust alpha for visibility

        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"
        # Add text for waiting state
        self.waiting_text_org = (int(self.frame_width * 0.3), int(self.frame_height * 0.5))
        self.waiting_text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.waiting_text_scale = 1.0
        self.waiting_text_color = (0, 0, 255) # Red
        self.waiting_text_thickness = 2


    def _create_perspective_lookup_table(self):
        """Pre-calculates cm/pixel ratio for different y-coordinates in the ROI."""
        self.perspective_lookup_px_to_cm = {}
        # Step by 5 pixels for finer granularity
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 5):
            # y_global corresponds to the center of this 5px band within the ROI
            y_global_center = self.roi_top_y + y_in_roi_rounded + 2.5
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_global_center)

    def _get_pixel_to_cm_ratio(self, y_global):
        """Calculates estimated cm per pixel at a given global y-coordinate."""
        # Ensure y_global is within reasonable bounds for calculation
        y_eff = np.clip(y_global, 0, self.frame_height)

        # Relative position: 0 at frame top (far), 1 at roi_bottom_y (near perspective reference)
        if self.roi_bottom_y <= 0: # Avoid division by zero
             relative_y = 0.5
        else:
             # Use roi_bottom_y as the reference point for 'near'
             relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)

        # Linear interpolation of width between far and near sides
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y

        # Calculate cm per pixel based on the interpolated width across the ROI width in pixels
        if current_width_cm > 0 and self.roi_width_px > 0:
             # This ratio represents cm per HORIZONTAL pixel at this depth (y)
             cm_per_pixel_horizontal = current_width_cm / self.roi_width_px
        else:
             # Fallback: Use nominal table length / frame width (less accurate)
             cm_per_pixel_horizontal = self.table_length_cm / self.frame_width if self.frame_width > 0 else 0.1

        # *** Assumption: Assuming the cm/pixel ratio is roughly isotropic (same for x and y movement at this depth) ***
        # This is a simplification. True perspective correction is more complex.
        # We return the horizontal ratio, assuming it applies reasonably well to diagonal movement too.
        return cm_per_pixel_horizontal

    def _update_display_fps(self):
        # Always use monotonic clock for measuring frame processing rate
        now = time.monotonic()
        self.frame_timestamps_for_fps.append(now)

        # Calculate FPS based on the time elapsed over the stored timestamps
        if len(self.frame_timestamps_for_fps) >= 2:
            elapsed_time = self.frame_timestamps_for_fps[-1] - self.frame_timestamps_for_fps[0]
            if elapsed_time > 1e-9: # Avoid division by zero
                # Calculate FPS over the interval covered by the deque
                measured_fps = (len(self.frame_timestamps_for_fps) - 1) / elapsed_time
                # Apply smoothing
                # Use self.actual_fps (from reader) as initial baseline if display_fps is 0
                current_base = self.display_fps if self.display_fps > 0 else self.actual_fps
                self.display_fps = (1 - FPS_SMOOTHING_FACTOR) * current_base + FPS_SMOOTHING_FACTOR * measured_fps
            # else: handle case of zero elapsed time if necessary, maybe decay FPS?

    def _preprocess_frame(self, frame):
        # ROI slicing (this creates a view, modifications affect original frame)
        roi_sub_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        gray_roi = cv2.cvtColor(roi_sub_frame, cv2.COLOR_BGR2GRAY)

        # Gaussian blur reduces noise before differencing
        gray_roi_blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Store the blurred frame for FMO calculation
        self.prev_frames_gray_roi.append(gray_roi_blurred)

        # Return the original ROI view (for drawing) and the processed gray ROI
        return roi_sub_frame, gray_roi_blurred

    def _detect_fmo(self):
        if len(self.prev_frames_gray_roi) < 3:
            return None # Need at least 3 frames for this FMO method

        # Get the last 3 frames
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]

        # Calculate differences
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        motion_mask = cv2.bitwise_and(diff1, diff2)

        # Thresholding (OTSU is good for contrast, but fallback needed)
        try:
            _, thresh_mask = cv2.threshold(motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error: # Handle cases where OTSU might fail (e.g., uniform image)
            _, thresh_mask = cv2.threshold(motion_mask, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up the mask
        if self.opening_kernel.shape[0] > 0: # Avoid error if kernel size is (0,0)
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else:
            opened_mask = thresh_mask

        if self.closing_kernel.shape[0] > 0:
             closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        else:
             closed_mask = opened_mask

        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        """Detects ball candidates in the ROI using connected components."""
        contours, _ = cv2.findContours(motion_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        potential_balls = []
        current_timestamp = time.monotonic() # Use monotonic time for detection timestamp

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            if MIN_BALL_AREA_PX < area < MAX_BALL_AREA_PX:
                # Calculate moments to find centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx_roi = int(M["m10"] / M["m00"])
                    cy_roi = int(M["m01"] / M["m00"])
                else:
                    # Fallback if moment is zero (shouldn't happen for valid contours)
                    (cx_roi, cy_roi), _ = cv2.minEnclosingCircle(cnt)
                    cx_roi, cy_roi = int(cx_roi), int(cy_roi)

                # Calculate circularity using perimeter
                perimeter = cv2.arcLength(cnt, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)

                # Store candidate info
                potential_balls.append({
                    'position_roi': (cx_roi, cy_roi),
                    'area': area,
                    'circularity': circularity,
                    'contour_roi': cnt # Store the contour itself
                })

        if not potential_balls: return None, None

        # Select the best candidate based on criteria
        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        # Convert ROI coordinates to global frame coordinates
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y

        # Update last detection system time (used for timeout)
        self.last_detection_timestamp = time.monotonic()

        # --- Call crossing check ---
        # Pass the current ball position and its corresponding timestamp
        if self.is_counting_active:
            # check_center_crossing now handles interpolation using last_ball_timestamp
            self.check_center_crossing(cx_global, current_timestamp) # Pass monotonic time


        # Add to trajectory (global coords, monotonic time)
        self.trajectory.append((cx_global, cy_global, current_timestamp))

        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}")

        # Return position and contour relative to ROI for drawing
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')

    def _select_best_ball_candidate(self, candidates):
        """Selects the most likely ball from candidates based on circularity, proximity, and consistency."""
        if not candidates: return None

        # Filter by minimum circularity first
        plausible_balls = [b for b in candidates if b['circularity'] >= MIN_BALL_CIRCULARITY]
        if not plausible_balls:
            # If none meet circularity, maybe relax slightly or return largest?
            # For now, let's stick to the circularity requirement.
             # Or fallback: return max(candidates, key=lambda b: b['area'])
            return None # No sufficiently circular candidates found

        # If only one plausible candidate, return it
        if len(plausible_balls) == 1:
            return plausible_balls[0]

        # If trajectory exists, score based on proximity and motion consistency
        if self.trajectory:
            last_x_global, last_y_global, _ = self.trajectory[-1]

            for ball_info in plausible_balls:
                cx_roi, cy_roi = ball_info['position_roi']
                cx_global = cx_roi + self.roi_start_x
                cy_global = cy_roi + self.roi_top_y

                # Distance from last known position
                distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
                ball_info['distance_from_last'] = distance

                # Motion consistency score (using cosine similarity with previous vector)
                consistency_score = 0
                if len(self.trajectory) >= 2:
                    prev_x_global, prev_y_global, _ = self.trajectory[-2]
                    vec_hist_dx = last_x_global - prev_x_global
                    vec_hist_dy = last_y_global - prev_y_global
                    vec_curr_dx = cx_global - last_x_global
                    vec_curr_dy = cy_global - last_y_global

                    mag_hist_sq = vec_hist_dx**2 + vec_hist_dy**2
                    mag_curr_sq = vec_curr_dx**2 + vec_curr_dy**2

                    if mag_hist_sq > 1e-6 and mag_curr_sq > 1e-6: # Avoid division by zero / sqrt(0)
                        dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
                        cosine_similarity = dot_product / (math.sqrt(mag_hist_sq) * math.sqrt(mag_curr_sq))
                        consistency_score = max(0, cosine_similarity) # Penalize direction changes > 90deg
                ball_info['consistency'] = consistency_score

                # Combined Score (adjust weights as needed)
                # Lower distance is better -> use inverse or similar
                # Higher consistency is better
                # Higher circularity is better (already filtered, but can still weigh)
                score = (0.5 * ball_info['consistency']) + \
                        (0.3 * ball_info['circularity']) + \
                        (0.2 / (1.0 + distance)) # Weight proximity less if consistency/circ are high
                ball_info['score'] = score

            # Return the candidate with the highest score
            return max(plausible_balls, key=lambda b: b['score'])
        else:
            # No trajectory history, return the most circular among the plausible ones
            return max(plausible_balls, key=lambda b: b['circularity'])


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
            # Clear previous trajectory and speed when starting new count? Optional.
            # self.trajectory.clear()
            # self.current_ball_speed_kmh = 0
            self.last_ball_x_global = None # Reset last position for crossing check
            self.last_ball_timestamp = None # Reset last timestamp
            print(f"Counting ON (Session #{self.count_session_id}) - Target: {self.max_net_speeds_to_collect} speeds.")
        else:
            print(f"Counting OFF (Session #{self.count_session_id}).")
            # Generate output if counting is turned off and data exists
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_outputs_async()
            self.output_generated_for_session = True # Mark as generated/handled for this session


    def check_center_crossing(self, ball_x_global, current_timestamp):
        """Checks for center line crossing using interpolation between the last and current point."""
        if self.last_ball_x_global is None or self.last_ball_timestamp is None:
            # Store current position and time as the "last" for the next frame
            self.last_ball_x_global = ball_x_global
            self.last_ball_timestamp = current_timestamp
            return

        # --- Interpolation Check ---
        recorded = self._check_and_record_crossing_interpolated(
            self.last_ball_x_global,
            ball_x_global,
            self.last_ball_timestamp, # Timestamp of the previous ball detection
            current_timestamp,       # Timestamp of the current ball detection
            self.current_ball_speed_kmh # Use the latest calculated speed
        )

        # --- Update Last Known State ---
        # Always update the last known position and timestamp for the next frame's check
        self.last_ball_x_global = ball_x_global
        self.last_ball_timestamp = current_timestamp

        # --- Optional: Prediction Logic (Can be called here if needed) ---
        # If you still want predictions for events far in the future, you could call
        # a separate prediction function here, perhaps only if `recorded` is False.
        # self._predict_crossing(...)


    def _check_and_record_crossing_interpolated(self, last_x, current_x, last_t, current_t, current_speed_kmh):
        """
        Checks if the line segment (last_pos -> current_pos) crosses the center zone.
        If yes, interpolates the time and records the event using current speed.
        Returns True if an event was recorded, False otherwise.
        """
        # Basic validation: need valid points and time difference
        if last_x is None or current_t <= last_t:
            return False

        # Check cooldown based on the *last actual recorded event time*
        time_since_last_event = current_t - self.last_net_crossing_detection_time
        if time_since_last_event < CENTER_DETECTION_COOLDOWN_S:
            return False # Still cooling down from the last recorded crossing

        crossed_center = False
        crossing_direction = None
        intersection_line_x = None # Which line (start or end) was crossed
        fraction = 0.5 # Default fraction

        # Scenario 1: Crossing the end line (typically L -> R)
        # Check if the segment potentially crosses the right boundary of the zone
        if last_x < self.center_line_end_x and current_x >= self.center_line_end_x:
             # Ensure the segment has non-zero x-movement to avoid division by zero
             dx = current_x - last_x
             if abs(dx) > 1e-6:
                  crossed_center = True
                  crossing_direction = 'left_to_right'
                  intersection_line_x = self.center_line_end_x
                  # Calculate fraction of segment before crossing the END line
                  fraction = (self.center_line_end_x - last_x) / dx

        # Scenario 2: Crossing the start line (typically R -> L)
        # Check if the segment potentially crosses the left boundary of the zone
        elif last_x > self.center_line_start_x and current_x <= self.center_line_start_x:
             dx = current_x - last_x
             if abs(dx) > 1e-6:
                  crossed_center = True
                  crossing_direction = 'right_to_left'
                  intersection_line_x = self.center_line_start_x
                  # Calculate fraction of segment before crossing the START line
                  fraction = (self.center_line_start_x - last_x) / dx

        # If a crossing matching the desired direction occurred
        if crossed_center and (self.net_crossing_direction == 'both' or self.net_crossing_direction == crossing_direction):

            # Ensure speed is valid
            if current_speed_kmh <= 0:
                if self.debug_mode: print(f"Crossing {crossing_direction} detected but speed ({current_speed_kmh:.1f}) is zero/negative. Ignoring.")
                return False # Don't record events with zero or negative speed

            # Interpolate the time of crossing
            # Clamp fraction to [0, 1] just in case, although it should be within this range
            fraction = max(0.0, min(1.0, fraction))
            interpolated_time = last_t + fraction * (current_t - last_t)

            # --- Duplicate Check ---
            # Check if this interpolated event is too close to an existing event in the buffer
            is_duplicate = False
            # Use a slightly larger window for duplicate check than cooldown? Maybe 1.5x cooldown.
            duplicate_check_window = CENTER_DETECTION_COOLDOWN_S * 1.5
            for ev in self.event_buffer_center_cross:
                 # Check timestamp and also ensure speed is somewhat similar? (Optional)
                 if abs(ev.timestamp - interpolated_time) < duplicate_check_window:
                     is_duplicate = True
                     if self.debug_mode: print(f"Duplicate crossing event detected near {interpolated_time:.3f}. Ignoring.")
                     break

            if not is_duplicate:
                # Record the event with interpolated time and current speed
                event = EventRecord(intersection_line_x, # Record crossing at the line edge
                                    interpolated_time,
                                    current_speed_kmh,
                                    predicted=False) # This is an actual (interpolated) event
                self.event_buffer_center_cross.append(event)

                # IMPORTANT: Update the last *recorded event* timestamp - used for cooldown
                self.last_net_crossing_detection_time = interpolated_time

                if self.debug_mode:
                    print(f"Interpolated Crossing Recorded: {crossing_direction} at t={interpolated_time:.3f}s, Speed={current_speed_kmh:.1f} km/h")
                return True # Event recorded

        return False # No valid crossing recorded in this check


    def _process_crossing_events(self):
        """Processes events from the buffer, adds them to the collected list if valid."""
        if not self.is_counting_active or self.output_generated_for_session:
            return

        # Process events from the buffer one by one
        processed_count = 0
        events_to_commit = []

        # Iterate through a copy or manage indices carefully if modifying deque during iteration
        temp_buffer = list(self.event_buffer_center_cross) # Work on a temporary list

        indices_to_remove = []

        for i, event in enumerate(temp_buffer):
            if event.processed: # Already handled in a previous cycle? Should not happen if removed properly.
                # This might happen if processing is slow and events build up faster than processed
                # Mark for removal anyway
                indices_to_remove.append(i)
                continue

            # Check if we've reached the target count
            if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                # Once target is reached, we should stop adding new events.
                # We can break here, leaving unprocessed events in the buffer.
                # They will be cleared next time counting is toggled ON.
                break # Stop processing if limit reached

            # Mark event as processed *before* adding to commit list
            # This prevents it being added again if loop iterates strangely
            event.processed = True
            indices_to_remove.append(i) # Mark for removal from original deque later

            # Commit this event
            events_to_commit.append(event)
            processed_count += 1

        # --- Remove processed events from the actual deque ---
        # Rebuild the deque to ensure atomicity and avoid issues with modifying during iteration
        if indices_to_remove:
            new_buffer_list = []
            current_indices_set = set(range(len(temp_buffer)))
            processed_indices_set = set(indices_to_remove)
            indices_to_keep = sorted(list(current_indices_set - processed_indices_set))

            for idx in indices_to_keep:
                 new_buffer_list.append(temp_buffer[idx])

            self.event_buffer_center_cross = deque(new_buffer_list, maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)


        # --- Add committed events to the session's results ---
        if events_to_commit:
            # Sort events by timestamp before adding to results (important for relative time)
            events_to_commit.sort(key=lambda e: e.timestamp)

            for event in events_to_commit:
                 # Double check count again in case multiple events processed at once
                if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                    break

                # Calculate relative time
                if not self.timing_started_for_session:
                    self.timing_started_for_session = True
                    # Use the timestamp of the *first committed event* in this batch as the start
                    self.first_ball_crossing_timestamp = event.timestamp
                    relative_time = 0.0
                else:
                    # Ensure first_ball_crossing_timestamp is not None
                    if self.first_ball_crossing_timestamp is not None:
                        relative_time = round(event.timestamp - self.first_ball_crossing_timestamp, 2)
                    else:
                         # Fallback if first timestamp wasn't set (shouldn't happen)
                         relative_time = 0.0
                         if self.debug_mode: print("Warning: First crossing timestamp not set for relative time calc.")


                self.last_recorded_net_speed_kmh = event.speed_kmh
                self.collected_net_speeds.append(event.speed_kmh)
                self.collected_relative_times.append(relative_time)

                # Update last crossing detection time - already done in _check_and_record_crossing_interpolated
                # self.last_net_crossing_detection_time = event.timestamp

                status_msg = "Pred" if event.predicted else "Actual" # Should mostly be Actual now
                print(f"Net Speed #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg})")

        # Check if target count is reached *after* processing events for this frame
        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Target {self.max_net_speeds_to_collect} speeds collected. Generating output.")
            self._generate_outputs_async()
            self.output_generated_for_session = True
            if AUTO_STOP_AFTER_COLLECTION:
                print("Auto-stopping count.")
                self.is_counting_active = False # Optionally stop counting


    def _calculate_ball_speed(self):
        """Calculates ball speed using recent trajectory points, applying perspective correction."""
        # Use at least 2 points, up to SPEED_CALC_POINTS
        num_traj_points = len(self.trajectory)
        points_to_use = min(num_traj_points, SPEED_CALC_POINTS)

        if points_to_use < 2:
            # Not enough points, potentially decay speed slightly or set to zero
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR * 0.5) # Slower decay
            if self.current_ball_speed_kmh < 0.1 : self.current_ball_speed_kmh = 0
            return

        # Select the first and last point from the most recent 'points_to_use'
        segment_points = list(self.trajectory)[-points_to_use:]
        pt_start_glob = segment_points[0]
        pt_end_glob = segment_points[-1]

        x1_glob, y1_glob, t1 = pt_start_glob
        x2_glob, y2_glob, t2 = pt_end_glob

        delta_t = t2 - t1

        if delta_t > 1e-9: # Ensure time has passed
            # Calculate real-world distance using perspective correction
            dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)

            # Speed in cm/s (assuming time is in seconds from monotonic())
            speed_cm_per_sec = dist_cm / delta_t
            # Convert cm/s to km/h
            speed_kmh = speed_cm_per_sec * KMH_CONVERSION_FACTOR

            # Apply smoothing filter
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else:
                # If previous speed was zero, initialize directly
                self.current_ball_speed_kmh = speed_kmh

            if self.debug_mode and self.frame_counter % 10 == 0: # Print speed debug less often
                 print(f"Speed Calc ({points_to_use} pts): {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else:
            # No time difference, decay speed
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR * 0.5)
            if self.current_ball_speed_kmh < 0.1 : self.current_ball_speed_kmh = 0


    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        """Calculates estimated real-world distance in cm between two global points using perspective lookup."""
        # Find the corresponding y-coordinates within the ROI's perspective model
        # Use the y-coordinate relative to the top of the frame for perspective lookup
        # Find the nearest pre-calculated y-band in the lookup table
        y1_lookup = round((y1_g - self.roi_top_y) / 5) * 5 # Find closest 5px band in lookup
        y2_lookup = round((y2_g - self.roi_top_y) / 5) * 5

        # Clamp lookup keys to be within the valid range of the pre-calculated table
        y1_lookup = max(0, min(self.roi_height_px, y1_lookup))
        y2_lookup = max(0, min(self.roi_height_px, y2_lookup))


        # Get cm/pixel ratios from the lookup table, fallback to direct calc if needed (should be rare with clamping)
        cm_per_px_1 = self.perspective_lookup_px_to_cm.get(y1_lookup, self._get_pixel_to_cm_ratio(y1_g))
        cm_per_px_2 = self.perspective_lookup_px_to_cm.get(y2_lookup, self._get_pixel_to_cm_ratio(y2_g))

        # Use the average ratio for the segment
        avg_cm_per_pixel = (cm_per_px_1 + cm_per_px_2) / 2.0

        # Calculate pixel distance (hypotenuse)
        pixel_distance = math.hypot(x2_g - x1_g, y2_g - y1_g)

        # Convert pixel distance to real-world cm distance
        real_distance_cm = pixel_distance * avg_cm_per_pixel
        return real_distance_cm


    def _generate_outputs_async(self):
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return

        # Create copies for the thread to work on safely
        speeds_copy = list(self.collected_net_speeds)
        times_copy = list(self.collected_relative_times)
        session_id_copy = self.count_session_id

        print(f"Submitting output generation task for session {session_id_copy}...")
        self.file_writer_executor.submit(self._create_output_files, speeds_copy, times_copy, session_id_copy)

    def _create_output_files(self, net_speeds, relative_times, session_id):
        """Generates chart, TXT, and CSV files (runs in background thread)."""
        try:
            if not net_speeds:
                print(f"[FileWriter] No speeds provided for session {session_id}.")
                return

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include session ID in the folder name for clarity
            output_dir_path = os.path.join(OUTPUT_DATA_FOLDER, f"{timestamp_str}")
            os.makedirs(output_dir_path, exist_ok=True)

            avg_speed = sum(net_speeds) / len(net_speeds)
            max_speed = max(net_speeds)
            min_speed = min(net_speeds)

            # --- Generate Chart ---
            chart_filename = os.path.join(output_dir_path, f'speed_chart_{timestamp_str}.png')
            plt.figure(figsize=(12, 7)) # Create a new figure for this thread
            plt.plot(relative_times, net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
            plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')

            # --- Restore point annotations ---
            for t, s in zip(relative_times, net_speeds):
                plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            # --- End of restored section ---

            plt.title(f'Net Crossing Speeds - Session {session_id} ({timestamp_str})', fontsize=16)
            plt.xlabel('Relative Time (s)', fontsize=12)
            plt.ylabel('Speed (km/h)', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            # Adjust plot limits for better visualization
            if relative_times:
                time_range = max(relative_times) - min(relative_times) if len(relative_times) > 1 else 1.0
                x_margin = max(time_range * 0.05, 0.5) # Ensure some margin
                plt.xlim(min(relative_times) - x_margin, max(relative_times) + x_margin)

                speed_range = max_speed - min_speed if max_speed > min_speed else 10.0
                y_margin = max(speed_range * 0.1, 5.0) # Ensure some margin
                plt.ylim(max(0, min_speed - y_margin), max_speed + y_margin) # Ensure y starts at 0 or below min speed

            plt.figtext(0.02, 0.02, f"Count: {len(net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(chart_filename, dpi=150)
            plt.close() # IMPORTANT: Close the figure to release memory
            print(f"[FileWriter] Chart saved to {chart_filename}")

            # --- Generate TXT ---
            txt_filename = os.path.join(output_dir_path, f'speed_data_{timestamp_str}.txt')
            with open(txt_filename, 'w') as f:
                f.write(f"Net Speeds - Session {session_id} - {timestamp_str}\n")
                f.write("---------------------------------------\n")
                f.write("Point | Rel Time (s) | Speed (km/h)\n")
                f.write("---------------------------------------\n")
                for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                    f.write(f"{i+1:<5} | {t:<12.2f} | {s:.1f}\n")
                f.write("---------------------------------------\n")
                f.write(f"Total Points: {len(net_speeds)}\n")
                f.write(f"Average Speed: {avg_speed:.1f} km/h\n")
                f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
                f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")
            print(f"[FileWriter] TXT saved to {txt_filename}")

            # --- Generate CSV ---
            csv_filename = os.path.join(output_dir_path, f'speed_data_{timestamp_str}.csv')
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Data Header
                writer.writerow(['Session ID', 'Timestamp', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
                # Data Rows
                for i, (t, s) in enumerate(zip(relative_times, net_speeds)):
                    writer.writerow([session_id, timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
                # Summary Section
                writer.writerow([]) # Empty row separator
                writer.writerow(['Statistic', 'Value'])
                writer.writerow(['Total Points', len(net_speeds)])
                writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
                writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
                writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
            print(f"[FileWriter] CSV saved to {csv_filename}")

            print(f"[FileWriter] Output files for session {session_id} saved successfully to {output_dir_path}")

        except Exception as e:
            print(f"[FileWriter] Error generating output files for session {session_id}: {e}")
            import traceback
            traceback.print_exc()


    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData, waiting_for_frame=False):
        """Draws tracking information onto the display frame. Handles waiting state."""
        # If waiting, use the last known good frame, otherwise use the current one
        vis_frame = display_frame if not waiting_for_frame else self.last_frame_display
        if vis_frame is None: # Handle case where no frame has been received yet
             vis_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
             cv2.putText(vis_frame, "Initializing...", self.waiting_text_org, self.waiting_text_font,
                         self.waiting_text_scale, self.waiting_text_color, self.waiting_text_thickness)
             return vis_frame # Return blank frame with initializing text

        profiling_texts = [] # Collect profiling info if debug mode
        t_draw_start = time.perf_counter()

        # Apply static overlay with transparency
        # Make a copy if we are potentially modifying the stored last_frame_display
        if waiting_for_frame:
            vis_frame = vis_frame.copy()
        vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.4, 0)

        # If waiting, display waiting message and return
        if waiting_for_frame:
            cv2.putText(vis_frame, "Waiting for camera frame...", self.waiting_text_org, self.waiting_text_font,
                        self.waiting_text_scale, self.waiting_text_color, self.waiting_text_thickness)
            return vis_frame

        # --- If not waiting, draw normal visualizations ---
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0

        # Draw Trajectory (Full Draw)
        if is_full_draw and frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
            t_traj_start = time.perf_counter()
            pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)
            if self.debug_mode: profiling_texts.append(f"TrajDraw:{(time.perf_counter() - t_traj_start)*1000:.1f}ms")

        # Draw Ball Detection (Always if detected)
        if frame_data_obj.ball_position_in_roi:
            t_ball_draw_start = time.perf_counter()
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1)
            if is_full_draw and frame_data_obj.ball_contour_in_roi is not None:
                 offset_contour = frame_data_obj.ball_contour_in_roi + np.array([self.roi_start_x, self.roi_top_y])
                 cv2.drawContours(vis_frame, [offset_contour], 0, CONTOUR_COLOR_BGR, 1)
            if self.debug_mode: profiling_texts.append(f"BallDraw:{(time.perf_counter() - t_ball_draw_start)*1000:.1f}ms")

        # Draw Text Overlays (Always)
        t_text_start = time.perf_counter()
        y_pos = 30
        y_step = 35
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        y_pos += y_step
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        y_pos += y_step
        count_status_text = "ON" if frame_data_obj.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Counting: {count_status_text} (Sess:{self.count_session_id})", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        y_pos += y_step
        cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}/{self.max_net_speeds_to_collect}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        y_pos += y_step
        if frame_data_obj.last_recorded_net_speed_kmh > 0:
            cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
            y_pos += y_step
        if frame_data_obj.collected_relative_times:
            cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
            y_pos += y_step
        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Debug Info Text
        if self.debug_mode:
            debug_info = f"Traj:{len(self.trajectory)} EvtBuf:{len(self.event_buffer_center_cross)}"
            if frame_data_obj.debug_display_text:
                 debug_info += " " + frame_data_obj.debug_display_text
            cv2.putText(vis_frame, debug_info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            y_pos += y_step
            profiling_summary = frame_data_obj.profiling_info
            profiling_texts.append(f"TextDraw:{(time.perf_counter() - t_text_start)*1000:.1f}ms")
            total_draw_time = time.perf_counter() - t_draw_start
            profiling_texts.append(f"TotDraw: {total_draw_time*1000:.1f}ms")
            profiling_summary += " | " + " ".join(profiling_texts)
            cv2.putText(vis_frame, profiling_summary, (10, self.frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 230), 1)

        # Store the successfully drawn frame as the last good one
        self.last_frame_display = vis_frame.copy() # Store a copy
        return vis_frame


    def _check_timeout_and_reset(self):
        """Resets trajectory if no ball detected for timeout duration."""
        # This timeout only affects the trajectory, not system shutdown.
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            if self.trajectory: # Only clear if it's not already empty
                self.trajectory.clear()
                self.current_ball_speed_kmh = 0 # Reset speed as well
                self.last_ball_x_global = None # Reset last position for crossing check
                self.last_ball_timestamp = None
                if self.debug_mode: print("Ball detection timeout: Trajectory reset.")
            # No need to reset crossed_center flag, it's handled by cooldown/event buffer


    def process_single_frame(self, frame):
        """Processes a single frame for ball detection, tracking, and speed calculation."""
        t_start_process = time.perf_counter()
        profiling_results = []

        self.frame_counter += 1
        self._update_display_fps() # Update measured FPS

        # 1. Preprocessing (ROI, Grayscale, Blur)
        t_start_pre = time.perf_counter()
        # roi_sub_frame is a view, gray_roi_for_fmo is new data
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame)
        if self.debug_mode: profiling_results.append(f"Pre:{(time.perf_counter() - t_start_pre)*1000:.1f}ms")

        # 2. Fast Motion Object Detection (FMO)
        t_start_fmo = time.perf_counter()
        motion_mask_roi = self._detect_fmo()
        if self.debug_mode: profiling_results.append(f"FMO:{(time.perf_counter() - t_start_fmo)*1000:.1f}ms")

        # 3. Ball Detection and Tracking
        t_start_detect = time.perf_counter()
        ball_pos_in_roi, ball_contour_in_roi = None, None
        detection_successful = False # Flag to indicate if a ball was found in this frame
        if motion_mask_roi is not None:
            # This step updates trajectory, last_detection_timestamp,
            # and potentially calls check_center_crossing (which updates event buffer)
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi)
            if ball_pos_in_roi is not None:
                 detection_successful = True # Set flag if ball detected
        if self.debug_mode: profiling_results.append(f"Detect:{(time.perf_counter() - t_start_detect)*1000:.1f}ms")

        # 4. Speed Calculation (uses updated trajectory)
        t_start_speed = time.perf_counter()
        # Only calculate speed if a ball was detected/trajectory updated in this frame
        if detection_successful:
             self._calculate_ball_speed()
        if self.debug_mode: profiling_results.append(f"Speed:{(time.perf_counter() - t_start_speed)*1000:.1f}ms")

        # 5. Timeout Check (Reset trajectory if ball lost)
        t_start_timeout = time.perf_counter()
        # Pass detection result to timeout check - reset only if no ball *currently* detected
        # Or keep original logic: reset if no ball detected for X seconds? Keep original.
        self._check_timeout_and_reset()
        if self.debug_mode: profiling_results.append(f"Timeout:{(time.perf_counter() - t_start_timeout)*1000:.1f}ms")

        # 6. Process buffered crossing events (if counting)
        t_start_events = time.perf_counter()
        if self.is_counting_active:
            self._process_crossing_events()
        if self.debug_mode: profiling_results.append(f"Events:{(time.perf_counter() - t_start_events)*1000:.1f}ms")

        # 7. Prepare FrameData for visualization
        # Pass copies of mutable lists that might change concurrently (though less likely now)
        frame_data = FrameData(
            frame=frame, # Pass the original frame (or view) for drawing
            roi_sub_frame=roi_sub_frame, # Pass the ROI view
            ball_position_in_roi=ball_pos_in_roi,
            ball_contour_in_roi=ball_contour_in_roi,
            current_ball_speed_kmh=self.current_ball_speed_kmh,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds), # Copy
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times), # Copy
            debug_display_text=None, # Specific debug text can be added here if needed
            frame_counter=self.frame_counter,
            profiling_info = " ".join(profiling_results) if self.debug_mode else ""
        )
        # Get global trajectory points for drawing
        if self.trajectory:
            # Convert trajectory points (float) to int for drawing
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]

        if self.debug_mode:
            total_process_time = time.perf_counter() - t_start_process
            frame_data.profiling_info += f" | TotProc:{total_process_time*1000:.1f}ms"
            # Optional: Print if processing takes longer than frame interval
            # frame_interval = 1.0 / self.display_fps if self.display_fps > 0 else 0
            # if frame_interval > 0 and total_process_time > frame_interval:
            #     print(f"WARNING: Frame {self.frame_counter} processing time ({total_process_time*1000:.1f}ms) > frame interval ({frame_interval*1000:.1f}ms)")

        return frame_data


    def run(self):
        print("=== Ping Pong Speed Tracker v11.2 ===")
        print(f"Input Source: {'Video File' if self.use_video_file else 'Webcam'}")
        print(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.actual_fps:.1f} FPS (Reported/Target)")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction}")
        print(f"Target speeds to collect: {self.max_net_speeds_to_collect}")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.main_loop_running = True
        self.reader.start() # Start the frame reading thread

        window_name = 'Ping Pong Speed Tracker v11.2'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Or WINDOW_NORMAL

        frame_data_obj = None # To hold the last valid processed data
        waiting_for_frame = False # Flag to indicate if we are waiting

        try:
            while self.main_loop_running:
                t_loop_start = time.perf_counter()

                # 1. Read Frame from Queue
                ret, frame = self.reader.read()

                # --- Handle Read Result ---
                if ret is True and frame is not None:
                    # Successfully got a frame
                    waiting_for_frame = False # No longer waiting
                    # 2. Process Frame
                    frame_data_obj = self.process_single_frame(frame)
                    # 3. Draw Visualizations (Normal)
                    display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj, waiting_for_frame=False)

                elif ret is False:
                    # Read failed in the reader thread (e.g., camera disconnected, end of video)
                    print("Reader indicated read failure. Stopping main loop.")
                    self.main_loop_running = False # Stop the main loop
                    waiting_for_frame = False # Not waiting anymore
                    # Generate final output if needed
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("Read failure with pending data. Generating output.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    continue # Skip rest of the loop

                elif ret is None:
                    # Timeout happened in reader.read() - no frame arrived in time
                    if not self.reader.is_running:
                         # Reader is stopped, likely end of video or user quit indirectly
                         print("Reader stopped while waiting for frame. Stopping main loop.")
                         self.main_loop_running = False
                         waiting_for_frame = False
                         continue
                    else:
                         # Reader is still running, just haven't received a frame yet (temporary issue?)
                         waiting_for_frame = True # Set waiting flag
                         print("Waiting for camera frame...") # Inform user
                         # 3. Draw Visualizations (Waiting State) - using last good frame
                         # Pass frame_data_obj=None or similar? Draw needs last good frame.
                         display_frame = self._draw_visualizations(None, None, waiting_for_frame=True)

                else:
                    # Should not happen, but handle unexpected case
                    print(f"Unexpected return from reader.read(): ret={ret}. Skipping frame.")
                    waiting_for_frame = False
                    continue

                # 4. Display Frame (either normal or waiting frame)
                if display_frame is not None:
                     cv2.imshow(window_name, display_frame)
                else:
                     # Handle case where display_frame is None (e.g., first frame init failed)
                     # Display a black screen or placeholder
                     placeholder = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                     cv2.putText(placeholder, "Error displaying frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                     cv2.imshow(window_name, placeholder)


                # 5. Handle User Input
                key = cv2.waitKey(1) & 0xFF # waitKey(1) is crucial for frame display
                if key == ord('q') or key == 27: # ESC key
                    print("Quit key pressed.")
                    self.main_loop_running = False # Signal loop to stop
                    # Generate output if quitting with data
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                         print("Quitting with pending data. Generating output...")
                         self._generate_outputs_async()
                         self.output_generated_for_session = True
                    # No need to break here, loop condition will handle it
                elif key == ord(' '): # Space bar
                    self.toggle_counting()
                elif key == ord('d'): # 'd' key
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

                # Optional: Print loop time for performance diagnosis
                if self.debug_mode and self.frame_counter % 30 == 0: # Print less often
                    loop_time = time.perf_counter() - t_loop_start
                    print(f"Main loop time: {loop_time*1000:.1f}ms (Waiting: {waiting_for_frame})")


        except KeyboardInterrupt:
            print("Process interrupted by user (Ctrl+C).")
            self.main_loop_running = False # Signal loop to stop
            # Generate output if interrupted with data
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                 print("Interrupted with pending data. Generating output.")
                 self._generate_outputs_async()
                 self.output_generated_for_session = True
        finally:
            print("Shutting down...")
            # Ensure reader thread is stopped cleanly
            if hasattr(self, 'reader'):
                 self.reader.stop() # This now sets reader.running = False

            # Shutdown the file writer executor, wait for tasks to complete
            if hasattr(self, 'file_writer_executor'):
                 print("Waiting for file writer tasks to complete...")
                 # Allow slightly more time?
                 self.file_writer_executor.shutdown(wait=True, cancel_futures=False)
                 print("File writer stopped.")

            cv2.destroyAllWindows()
            print("System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker v11.2')
    parser.add_argument('--video', type=str, default=None, help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='Webcam index.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='Target FPS for webcam.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='Frame width.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='Frame height.')
    parser.add_argument('--table_len', type=float, default=DEFAULT_TABLE_LENGTH_CM, help='Table length (cm) for perspective fallback.') # Allow float
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='Ball detection timeout (s) for trajectory reset.') # Clarified help text

    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                        choices=['left_to_right', 'right_to_left', 'both'], help='Net crossing direction to record.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='Number of net speeds to collect per session.')

    parser.add_argument('--near_width', type=float, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at near side.') # Allow float
    parser.add_argument('--far_width', type=float, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='Real width (cm) of ROI at far side.') # Allow float

    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='Enable debug printouts and profiling.')

    # Example of how to make other parameters configurable if needed:
    # parser.add_argument('--vis_interval', type=int, default=VISUALIZATION_DRAW_INTERVAL, help='Full visualization draw interval (frames).')
    # parser.add_argument('--cooldown', type=float, default=CENTER_DETECTION_COOLDOWN_S, help='Net detection cooldown (s).')


    args = parser.parse_args()

    # Update globals if configured via args (Example)
    # VISUALIZATION_DRAW_INTERVAL = args.vis_interval
    # CENTER_DETECTION_COOLDOWN_S = args.cooldown

    video_source_arg = args.video if args.video else args.camera_idx
    use_video_file_arg = True if args.video else False

    try:
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
            # Pass other args if configured:
            # cooldown_s=args.cooldown,
            # vis_interval=args.vis_interval,
        )
        tracker.run()
    except IOError as e:
        print(f"ERROR initializing tracker: {e}")
        print("Please check camera index/video path and permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()