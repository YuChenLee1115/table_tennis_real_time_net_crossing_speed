#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11.1 高級深度校正版
# Lightweight, optimized, multi-threaded (acquisition & I/O), macOS compatible
# 增強版：添加高精度四點透視校正技術與非線性深度校正

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
DEFAULT_TABLE_LENGTH_CM = 152  # 更新到圖中顯示的值 (152cm)

# Detection Parameters
DEFAULT_DETECTION_TIMEOUT = 0.05
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.8
MAX_TRAJECTORY_POINTS = 80

# Center Line Detection
CENTER_LINE_WIDTH_PIXELS = 20
CENTER_DETECTION_COOLDOWN_S = 0.15
MAX_NET_SPEEDS_TO_COLLECT = 27
NET_CROSSING_DIRECTION_DEFAULT = 'left_to_right' # 'left_to_right', 'right_to_left', 'both'
AUTO_STOP_AFTER_COLLECTION = False
OUTPUT_DATA_FOLDER = 'real_time_output'

# Perspective Correction
NEAR_SIDE_WIDTH_CM_DEFAULT = 29  # 圖中顯示的近端寬度
FAR_SIDE_WIDTH_CM_DEFAULT = 72   # 圖中顯示的遠端寬度

# 深度校正參數 - 新增
DEPTH_CORRECTION_ENABLE = True   # 是否啟用深度校正
DEPTH_CORRECTION_FACTOR = 3    # 深度校正因子（用於調整遠端測量值）
DEPTH_CORRECTION_CURVE = 4.0     # 深度校正曲線（控制校正量的非線性程度）
BALL_SIZE_CORRECTION_ENABLE = True  # 是否使用球體大小作為深度參考

# FMO (Fast Moving Object) Parameters
MAX_PREV_FRAMES_FMO = 5
OPENING_KERNEL_SIZE_FMO = (10, 10)
CLOSING_KERNEL_SIZE_FMO = (15, 15)
THRESHOLD_VALUE_FMO = 10

# Ball Detection Parameters
MIN_BALL_AREA_PX = 10
MAX_BALL_AREA_PX = 7000
MIN_BALL_CIRCULARITY = 0.5
REFERENCE_BALL_AREA = 400  # 參考球體大小（像素面積）- 新增

# Speed Calculation
SPEED_SMOOTHING_FACTOR = 0.5
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
GRID_COLOR_BGR = (0, 255, 255)  # 透視網格顏色

# Threading & Queue Parameters
FRAME_QUEUE_SIZE = 5 # For FrameReader
EVENT_BUFFER_SIZE_CENTER_CROSS = 20
PREDICTION_LOOKAHEAD_FRAMES = 5

# Debug
DEBUG_MODE_DEFAULT = False
DRAW_GRID_INTERVAL = 30  # 每30幀繪製一次透視網格
SHOW_DEPTH_CORRECTION = True  # 顯示深度校正資訊

# —— OpenCV Optimization ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 4)
except AttributeError:
    cv2.setNumThreads(4)


class AdvancedPerspectiveCorrector:
    """用於桌球場景的進階透視校正，包含非線性深度校正"""
    
    def __init__(self, frame_width, frame_height, 
                 table_length_cm, near_width_cm, far_width_cm,
                 roi_start_x, roi_end_x, roi_top_y, roi_bottom_y,
                 depth_correction_factor=DEPTH_CORRECTION_FACTOR,
                 depth_correction_curve=DEPTH_CORRECTION_CURVE,
                 enable_depth_correction=DEPTH_CORRECTION_ENABLE,
                 enable_ball_size_correction=BALL_SIZE_CORRECTION_ENABLE):
        """
        初始化進階透視校正器
        
        Args:
            frame_width, frame_height: 影像尺寸
            table_length_cm: 桌球桌長度（公分）
            near_width_cm: 近端寬度（公分）
            far_width_cm: 遠端寬度（公分）
            roi_*: ROI區域座標
            depth_correction_factor: 深度校正係數
            depth_correction_curve: 深度校正曲線指數
            enable_depth_correction: 是否啟用深度校正
            enable_ball_size_correction: 是否啟用球體大小校正
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.table_length_cm = table_length_cm
        self.near_width_cm = near_width_cm
        self.far_width_cm = far_width_cm
        
        # ROI 區域
        self.roi_start_x = roi_start_x
        self.roi_end_x = roi_end_x
        self.roi_top_y = roi_top_y
        self.roi_bottom_y = roi_bottom_y
        
        # 深度校正參數
        self.depth_correction_factor = depth_correction_factor
        self.depth_correction_curve = depth_correction_curve
        self.enable_depth_correction = enable_depth_correction
        self.enable_ball_size_correction = enable_ball_size_correction
        
        # 最後一次校正信息（用於調試）
        self.last_correction_info = {}
        
        # 設置透視變換
        self.setup_homography()
    
    def setup_homography(self):
        """設置透視變換矩陣，基於四角對應法"""
        # 原始影像上的四個點（左上、右上、左下、右下）
        self.src_points = np.array([
            [self.roi_start_x, self.roi_top_y],             # 左上
            [self.roi_end_x, self.roi_top_y],               # 右上
            [self.roi_start_x, self.roi_bottom_y],          # 左下
            [self.roi_end_x, self.roi_bottom_y]             # 右下
        ], dtype=np.float32)
        
        # 實際空間中的對應點（公分）
        # 考慮到遠近寬度差異，遠端（上方）比近端（下方）窄
        far_half_width = self.far_width_cm / 2
        near_half_width = self.near_width_cm / 2
        
        self.dst_points_cm = np.array([
            [-far_half_width, 0],                           # 左上（遠端）
            [far_half_width, 0],                            # 右上（遠端）
            [-near_half_width, self.table_length_cm],       # 左下（近端）
            [near_half_width, self.table_length_cm]         # 右下（近端）
        ], dtype=np.float32)
        
        # 計算透視變換矩陣（從像素到公分）
        self.homography_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points_cm)
        
        # 計算反向透視變換矩陣（從公分到像素）
        self.inv_homography_matrix = cv2.getPerspectiveTransform(self.dst_points_cm, self.src_points)
        
        # 建立查找表以加速計算
        self.create_lookup_tables()
    
    def create_lookup_tables(self):
        """建立像素座標到實際座標的查找表，提高效率"""
        # 為提高效率，我們每隔幾個像素建立一次對照
        step = 10  # 每10個像素採樣一次
        self.position_lookup = {}
        
        # 針對ROI區域內的點建立查找表
        for y in range(self.roi_top_y, self.roi_bottom_y + 1, step):
            for x in range(self.roi_start_x, self.roi_end_x + 1, step):
                # 計算該點在實際空間中的座標
                real_x, real_y = self.pixel_to_real_position(x, y)
                # 儲存到查找表
                self.position_lookup[(x, y)] = (real_x, real_y)
    
    def pixel_to_real_position(self, px, py):
        """將像素座標轉換為實際空間座標（公分）"""
        # 組織為單點格式以便透視變換
        point = np.array([[[px, py]]], dtype=np.float32)
        
        # 應用透視變換
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        # 返回實際座標 (x, y)，以公分為單位
        return transformed[0][0][0], transformed[0][0][1]
    
    def real_to_pixel_position(self, real_x, real_y):
        """將實際空間座標（公分）轉換為像素座標"""
        # 組織為單點格式以便反向透視變換
        point = np.array([[[real_x, real_y]]], dtype=np.float32)
        
        # 應用反向透視變換
        transformed = cv2.perspectiveTransform(point, self.inv_homography_matrix)
        
        # 返回像素座標 (px, py)
        return transformed[0][0][0], transformed[0][0][1]
    
    def calculate_depth_correction_factor(self, y_position, ball_area=None):
        """
        計算基於深度的校正因子
        
        Args:
            y_position: 在畫面中的y座標（用於估計深度）
            ball_area: 球體面積（可選，用於進一步提高精度）
            
        Returns:
            float: 速度校正因子
        """
        # 如果未啟用深度校正，返回1.0（無校正）
        if not self.enable_depth_correction:
            return 1.0
            
        # 計算相對深度（0=遠端/頂部，1=近端/底部）
        relative_depth = 0
        if self.roi_bottom_y > self.roi_top_y:  # 避免除以零
            relative_depth = (y_position - self.roi_top_y) / (self.roi_bottom_y - self.roi_top_y)
            relative_depth = np.clip(relative_depth, 0.0, 1.0)
        
        # 反轉相對深度，使0表示近端（底部），1表示遠端（頂部）
        depth_factor = 1.0 - relative_depth
        
        # 應用非線性曲線，使校正在遠端更強
        # depth_factor值域是0到1，0是近端，1是遠端
        # 通過冪運算使曲線變得非線性
        depth_factor = pow(depth_factor, self.depth_correction_curve)
        
        # 根據深度因子計算最終校正因子
        # 從1.0（底部/近端）到depth_correction_factor（頂部/遠端）的線性插值
        correction = 1.0 + (self.depth_correction_factor - 1.0) * depth_factor
        
        # 球體大小校正（如果啟用且提供了球體面積）
        if self.enable_ball_size_correction and ball_area is not None and ball_area > 0 and REFERENCE_BALL_AREA > 0:
            # 球體大小比例（與參考大小比較）
            size_ratio = math.sqrt(REFERENCE_BALL_AREA / ball_area)
            # 限制比例在合理範圍內
            size_ratio = np.clip(size_ratio, 0.5, 2.0)
            # 混合深度校正和球體大小校正
            correction = correction * 0.7 + size_ratio * 0.3
        
        # 保存最後的校正信息以便調試
        self.last_correction_info = {
            'y_position': y_position,
            'relative_depth': relative_depth,
            'depth_factor': depth_factor,
            'ball_area': ball_area,
            'correction': correction
        }
        
        return correction
    
    def get_real_distance(self, px1, py1, px2, py2, ball_area=None):
        """
        計算兩點在實際空間中的距離（公分），包含深度校正
        
        Args:
            px1, py1: 第一個點的像素座標
            px2, py2: 第二個點的像素座標
            ball_area: 球體面積（可選，用於球體大小校正）
            
        Returns:
            float: 校正後的實際距離（公分）
        """
        # 轉換兩點到實際座標
        real_x1, real_y1 = self.get_approximate_real_position(px1, py1)
        real_x2, real_y2 = self.get_approximate_real_position(px2, py2)
        
        # 計算歐氏距離
        distance_cm = np.sqrt((real_x2 - real_x1) ** 2 + (real_y2 - real_y1) ** 2)
        
        # 計算平均y位置，用於深度校正
        avg_y = (py1 + py2) / 2
        
        # 應用深度校正
        correction_factor = self.calculate_depth_correction_factor(avg_y, ball_area)
        corrected_distance = distance_cm * correction_factor
        
        # 保存輔助信息以便調試
        self.last_correction_info['raw_distance_cm'] = distance_cm
        self.last_correction_info['corrected_distance_cm'] = corrected_distance
        
        return corrected_distance
    
    def get_approximate_real_position(self, px, py):
        """使用查找表獲取近似的實際座標，以提高效率"""
        # 將座標對齊到最近的查找表網格點
        step = 10
        lookup_x = round(px / step) * step
        lookup_y = round(py / step) * step
        
        # 限制在查找表範圍內
        lookup_x = max(self.roi_start_x, min(self.roi_end_x, lookup_x))
        lookup_y = max(self.roi_top_y, min(self.roi_bottom_y, lookup_y))
        
        # 從查找表中獲取近似值
        if (lookup_x, lookup_y) in self.position_lookup:
            return self.position_lookup[(lookup_x, lookup_y)]
        
        # 如果查找表中沒有，計算精確值
        return self.pixel_to_real_position(px, py)
    
    def visualize_grid(self, frame, grid_step_cm=20):
        """在影像上繪製實際空間的網格，用於調試和可視化"""
        # 創建一個視覺化用的拷貝
        vis_frame = frame.copy()
        
        # 畫橫線（等間隔實際距離）
        for real_y in range(0, int(self.table_length_cm) + 1, grid_step_cm):
            points = []
            for real_x in range(int(-self.near_width_cm), int(self.near_width_cm) + 1, 5):
                px, py = self.real_to_pixel_position(real_x, real_y)
                if (px >= 0 and px < self.frame_width and py >= 0 and py < self.frame_height):
                    points.append((int(px), int(py)))
            
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(vis_frame, points[i], points[i+1], GRID_COLOR_BGR, 1)
                
                # 標註距離
                mid_point = points[len(points) // 2]
                cv2.putText(vis_frame, f"{real_y}cm", 
                           (mid_point[0] - 20, mid_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_COLOR_BGR, 1)
        
        # 畫縱線（等間隔實際距離）
        for real_x in range(int(-self.near_width_cm/2), int(self.near_width_cm/2) + 1, grid_step_cm):
            points = []
            for real_y in range(0, int(self.table_length_cm) + 1, 5):
                px, py = self.real_to_pixel_position(real_x, real_y)
                if (px >= 0 and px < self.frame_width and py >= 0 and py < self.frame_height):
                    points.append((int(px), int(py)))
            
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(vis_frame, points[i], points[i+1], GRID_COLOR_BGR, 1)
                
                # 標註距離
                mid_point = points[len(points) // 2]
                cv2.putText(vis_frame, f"{real_x}cm", 
                           (mid_point[0], mid_point[1] + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_COLOR_BGR, 1)
        
        # 顯示深度校正因子曲線
        if SHOW_DEPTH_CORRECTION and self.enable_depth_correction:
            # 繪製深度校正曲線
            curve_width = 200
            curve_height = 100
            curve_x = 50
            curve_y = self.frame_height - curve_height - 50
            
            # 繪製座標軸
            cv2.line(vis_frame, (curve_x, curve_y + curve_height), 
                    (curve_x + curve_width, curve_y + curve_height), (255, 255, 255), 1)
            cv2.line(vis_frame, (curve_x, curve_y + curve_height), 
                    (curve_x, curve_y), (255, 255, 255), 1)
                    
            # 繪製曲線
            prev_point = None
            for i in range(0, curve_width + 1, 5):
                relative_pos = i / curve_width
                # 計算相對深度 (0到1)
                depth_factor = relative_pos
                # 應用非線性曲線
                corrected_factor = pow(depth_factor, self.depth_correction_curve)
                # 計算校正因子
                factor = 1.0 + (self.depth_correction_factor - 1.0) * corrected_factor
                
                # 繪製到曲線上
                point_x = curve_x + i
                factor_scaled = np.clip(factor, 1.0, self.depth_correction_factor)
                factor_normalized = (factor_scaled - 1.0) / (self.depth_correction_factor - 1.0)
                point_y = curve_y + curve_height - int(factor_normalized * curve_height)
                
                if prev_point:
                    cv2.line(vis_frame, prev_point, (point_x, point_y), (0, 255, 0), 2)
                prev_point = (point_x, point_y)
            
            # 標註軸線
            cv2.putText(vis_frame, "遠端", (curve_x + curve_width - 30, curve_y + curve_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_frame, "近端", (curve_x - 40, curve_y + curve_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"x{self.depth_correction_factor:.1f}", (curve_x - 40, curve_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_frame, "深度校正曲線", (curve_x + 40, curve_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 如果有最近的校正信息，顯示在曲線上
            if 'depth_factor' in self.last_correction_info and 'correction' in self.last_correction_info:
                depth_factor = self.last_correction_info['depth_factor']
                correction = self.last_correction_info['correction']
                
                # 計算點的位置
                point_x = curve_x + int(depth_factor * curve_width)
                corr_normalized = (correction - 1.0) / (self.depth_correction_factor - 1.0)
                point_y = curve_y + curve_height - int(corr_normalized * curve_height)
                
                # 繪製當前點
                cv2.circle(vis_frame, (point_x, point_y), 5, (0, 0, 255), -1)
                
                # 標註校正信息
                cv2.putText(vis_frame, f"校正: x{correction:.2f}", (point_x + 10, point_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return vis_frame


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
        self.ball_area = None  # 新增: 儲存球體面積


class EventRecord:
    """Record for potential center line crossing events."""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False, ball_area=None):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False
        self.ball_area = ball_area  # 新增: 儲存球體面積，用於校正


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
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT,
                 depth_correction_factor=DEPTH_CORRECTION_FACTOR,
                 depth_correction_curve=DEPTH_CORRECTION_CURVE):
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
        self.last_ball_area = None  # 新增: 記錄最後的球體面積

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)

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
        self.depth_correction_factor = depth_correction_factor
        self.depth_correction_curve = depth_correction_curve
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # 創建進階透視校正器
        self.perspective_corrector = AdvancedPerspectiveCorrector(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            table_length_cm=self.table_length_cm,
            near_width_cm=self.near_side_width_cm,
            far_width_cm=self.far_side_width_cm,
            roi_start_x=self.roi_start_x,
            roi_end_x=self.roi_end_x,
            roi_top_y=self.roi_top_y,
            roi_bottom_y=self.roi_bottom_y,
            depth_correction_factor=self.depth_correction_factor,
            depth_correction_curve=self.depth_correction_curve,
            enable_depth_correction=DEPTH_CORRECTION_ENABLE,
            enable_ball_size_correction=BALL_SIZE_CORRECTION_ENABLE
        )

        # 為保持向後兼容性，創建原來的查找表
        self._create_perspective_lookup_table()
        self._precalculate_overlay()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10): # step by 10px
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        # 原本的透視校正方法 (保持向後兼容性)
        y_eff = min(y_global, self.roi_bottom_y) 
        
        if self.roi_bottom_y == 0:
            relative_y = 0.5
        else:
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)

        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        
        roi_width_px = self.roi_end_x - self.roi_start_x
        if current_width_cm > 0:
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
        if len(self.prev_frames_gray_roi) < 3:
            return None
        
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
        else:
            opened_mask = thresh_mask
        
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        return closed_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        
        potential_balls = []
        for i in range(1, num_labels):
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
                    'contour_roi': contours[0] if contours else None
                })

        if not potential_balls: return None, None

        best_ball_info = self._select_best_ball_candidate(potential_balls)
        if not best_ball_info: return None, None

        cx_roi, cy_roi = best_ball_info['position_roi']
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        current_timestamp = time.monotonic()
        if self.use_video_file:
            current_timestamp = self.frame_counter / self.actual_fps
        
        self.last_detection_timestamp = time.monotonic()
        
        # 記錄球體面積，用於球體大小校正
        self.last_ball_area = best_ball_info['area']
        
        if self.is_counting_active:
            self.check_center_crossing(cx_global, current_timestamp, self.last_ball_area)
        
        self.trajectory.append((cx_global, cy_global, current_timestamp))
        
        if self.debug_mode:
            print(f"Ball: ROI({cx_roi},{cy_roi}), Global({cx_global},{cy_global}), Area:{best_ball_info['area']:.1f}, Circ:{best_ball_info['circularity']:.3f}")
        
        return best_ball_info['position_roi'], best_ball_info.get('contour_roi')

    def _select_best_ball_candidate(self, candidates):
        if not candidates: return None

        if not self.trajectory:
            highly_circular = [b for b in candidates if b['circularity'] > MIN_BALL_CIRCULARITY]
            if highly_circular:
                return max(highly_circular, key=lambda b: b['circularity'])
            return max(candidates, key=lambda b: b['area'])

        last_x_global, last_y_global, _ = self.trajectory[-1]

        for ball_info in candidates:
            cx_roi, cy_roi = ball_info['position_roi']
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y

            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            ball_info['distance_from_last'] = distance
            
            if distance > self.frame_width * 0.2:
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
                self._generate_outputs_async()
            self.output_generated_for_session = True

    def check_center_crossing(self, ball_x_global, current_timestamp, ball_area=None):
        if self.last_ball_x_global is None:
            self.last_ball_x_global = ball_x_global
            return

        time_since_last_net_cross = current_timestamp - self.last_net_crossing_detection_time
        if time_since_last_net_cross < CENTER_DETECTION_COOLDOWN_S:
            self.last_ball_x_global = ball_x_global
            return

        self._record_potential_crossing(ball_x_global, current_timestamp, ball_area)
        self.last_ball_x_global = ball_x_global

    def _record_potential_crossing(self, ball_x_global, current_timestamp, ball_area=None):
        crossed_l_to_r = (self.last_ball_x_global < self.center_line_end_x and ball_x_global >= self.center_line_end_x)
        crossed_r_to_l = (self.last_ball_x_global > self.center_line_start_x and ball_x_global <= self.center_line_start_x)
        
        actual_crossing_detected = False
        if self.net_crossing_direction == 'left_to_right' and crossed_l_to_r: actual_crossing_detected = True
        elif self.net_crossing_direction == 'right_to_left' and crossed_r_to_l: actual_crossing_detected = True
        elif self.net_crossing_direction == 'both' and (crossed_l_to_r or crossed_r_to_l): actual_crossing_detected = True

        if actual_crossing_detected and self.current_ball_speed_kmh > 0:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False, ball_area=ball_area)
            self.event_buffer_center_cross.append(event)
            return

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
                    can_add_prediction = True
                    for ev in self.event_buffer_center_cross:
                        if ev.predicted and abs(ev.timestamp - predicted_timestamp_future) < 0.1:
                            can_add_prediction = False
                            break
                    if can_add_prediction:
                        event = EventRecord(predicted_x_future, predicted_timestamp_future, self.current_ball_speed_kmh, predicted=True, ball_area=ball_area)
                        self.event_buffer_center_cross.append(event)

    def _process_crossing_events(self):
        if not self.is_counting_active or self.output_generated_for_session:
            return

        current_eval_time = time.monotonic()
        if self.use_video_file: current_eval_time = self.frame_counter / self.actual_fps
        
        events_to_commit = []
        
        processed_indices = []
        for i, event in enumerate(self.event_buffer_center_cross):
            if event.processed: continue
            if not event.predicted:
                events_to_commit.append(event)
                event.processed = True
                for j, other_event in enumerate(self.event_buffer_center_cross):
                    if i !=j and other_event.predicted and not other_event.processed and \
                       abs(event.timestamp - other_event.timestamp) < 0.2:
                        other_event.processed = True

        for event in self.event_buffer_center_cross:
            if event.processed: continue
            if event.predicted and (current_eval_time - event.timestamp) > 0.1:
                events_to_commit.append(event)
                event.processed = True

        events_to_commit.sort(key=lambda e: e.timestamp)

        for event in events_to_commit:
            if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect:
                break

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
            
            status_msg = "預測" if event.predicted else "實際"
            ball_area_msg = f", 球體面積={event.ball_area:.0f}" if event.ball_area and self.debug_mode else ""
            print(f"Net Speed #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg}{ball_area_msg})")

        self.event_buffer_center_cross = deque(
            [e for e in self.event_buffer_center_cross if not e.processed],
            maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS
        )

        if len(self.collected_net_speeds) >= self.max_net_speeds_to_collect and not self.output_generated_for_session:
            print(f"Target {self.max_net_speeds_to_collect} speeds collected. Generating output.")
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

        # 使用先進的透視校正計算實際距離，包含深度校正
        dist_cm = self.perspective_corrector.get_real_distance(
            x1_glob, y1_glob, x2_glob, y2_glob, self.last_ball_area
        )
        
        delta_t = t2 - t1
        if delta_t > 0:
            speed_cm_per_time_unit = dist_cm / delta_t
            speed_kmh = speed_cm_per_time_unit * KMH_CONVERSION_FACTOR 
            
            if self.current_ball_speed_kmh > 0:
                self.current_ball_speed_kmh = (1 - SPEED_SMOOTHING_FACTOR) * self.current_ball_speed_kmh + \
                                           SPEED_SMOOTHING_FACTOR * speed_kmh
            else:
                self.current_ball_speed_kmh = speed_kmh
            
            if self.debug_mode:
                correction_info = self.perspective_corrector.last_correction_info
                raw_dist = correction_info.get('raw_distance_cm', 0)
                corr_dist = correction_info.get('corrected_distance_cm', 0)
                corr_factor = correction_info.get('correction', 1.0)
                depth_factor = correction_info.get('depth_factor', 0)
                print(f"Speed: 原始距離={raw_dist:.2f}cm, 校正距離={corr_dist:.2f}cm (x{corr_factor:.2f}, 深度因子={depth_factor:.2f}), 時間={delta_t:.4f}s -> 速度={speed_kmh:.1f}km/h")
        else:
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
        """
        原始方法保留但不使用，已被新的透視校正取代
        這裡保留代碼是為了向後兼容性
        """
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_filename, dpi=150)
        plt.close()

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
        global DEPTH_CORRECTION_ENABLE
        global BALL_SIZE_CORRECTION_ENABLE
        
        vis_frame = display_frame
        
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0

        if is_full_draw:
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)

        # 如果處於調試模式並每30幀顯示一次透視網格
        if self.debug_mode and frame_data_obj.frame_counter % DRAW_GRID_INTERVAL == 0:
            vis_frame = self.perspective_corrector.visualize_grid(vis_frame, grid_step_cm=20)
            cv2.putText(vis_frame, "Grid: 20cm x 20cm", (10, 310), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRID_COLOR_BGR, 1)

        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
            if frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)
            
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1)

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

        # 顯示深度校正狀態
        corr_status = "ON" if DEPTH_CORRECTION_ENABLE else "OFF"
        corr_color = (0, 255, 0) if DEPTH_CORRECTION_ENABLE else (0, 0, 255)
        cv2.putText(vis_frame, f"Depth Corr: {corr_status} (x{self.depth_correction_factor:.1f})", (10, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, corr_color, 1)

        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if self.debug_mode and frame_data_obj.debug_display_text:
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        global DEPTH_CORRECTION_ENABLE
        
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame) 
        
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        ball_area = None
        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi)
            ball_area = self.last_ball_area
            self._calculate_ball_speed() 
        
        self._check_timeout_and_reset()
        
        if self.is_counting_active:
            self._process_crossing_events()

        # 球體大小校正信息
        corr_text = ""
        if self.debug_mode and DEPTH_CORRECTION_ENABLE:
            corr_text = f"深度校正: "
            if 'correction' in self.perspective_corrector.last_correction_info:
                corr = self.perspective_corrector.last_correction_info['correction']
                depth = self.perspective_corrector.last_correction_info.get('depth_factor', 0)
                raw_dist = self.perspective_corrector.last_correction_info.get('raw_distance_cm', 0)
                corr_dist = self.perspective_corrector.last_correction_info.get('corrected_distance_cm', 0)
                corr_text += f"因子={corr:.2f}, 深度={depth:.2f}, 距離={raw_dist:.1f}cm→{corr_dist:.1f}cm"

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
            debug_display_text=corr_text if corr_text else 
                              f"Traj: {len(self.trajectory)}, Events: {len(self.event_buffer_center_cross)}",
            frame_counter=self.frame_counter
        )
        frame_data.ball_area = ball_area
        
        if self.trajectory:
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        return frame_data

    def run(self):
        global DEPTH_CORRECTION_ENABLE
        global BALL_SIZE_CORRECTION_ENABLE
        
        print("=== 乒乓球速度追蹤器 v11.1 增強版 (進階深度校正) ===")
        print(self.instruction_text)
        print(f"透視校正: 近端 {self.near_side_width_cm}cm, 遠端 {self.far_side_width_cm}cm, 長度 {self.table_length_cm}cm")
        print(f"深度校正: {'啟用' if DEPTH_CORRECTION_ENABLE else '停用'}, 係數 x{self.depth_correction_factor:.1f}, 曲線 {self.depth_correction_curve:.1f}")
        print(f"球體大小校正: {'啟用' if BALL_SIZE_CORRECTION_ENABLE else '停用'}")
        print(f"穿越方向: {self.net_crossing_direction}")
        print(f"目標收集速度數: {self.max_net_speeds_to_collect}")
        if self.debug_mode: print("調試模式已啟用 (會顯示透視網格和校正資訊).")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker v11.1 Enhanced'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: print("影片結束或讀取錯誤.")
                    else: print("相機錯誤或串流結束.")
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("串流結束但有未處理的數據. 正在生成輸出.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                
                frame_data_obj = self.process_single_frame(frame)
                
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # ESC
                    self.running = False
                    if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                        print("退出程序但有未處理的數據. 正在生成輸出.")
                        self._generate_outputs_async()
                        self.output_generated_for_session = True
                    break
                elif key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"調試模式: {'開啟' if self.debug_mode else '關閉'}")
                # 新增: 增加/減少深度校正因子
                elif key == ord('+') or key == ord('='):
                    self.depth_correction_factor += 0.1
                    self.perspective_corrector.depth_correction_factor = self.depth_correction_factor
                    print(f"深度校正因子增加到: {self.depth_correction_factor:.1f}")
                elif key == ord('-') or key == ord('_'):
                    self.depth_correction_factor = max(1.0, self.depth_correction_factor - 0.1)
                    self.perspective_corrector.depth_correction_factor = self.depth_correction_factor
                    print(f"深度校正因子減少到: {self.depth_correction_factor:.1f}")
                # 新增: 開關深度校正
                elif key == ord('c'):
                    DEPTH_CORRECTION_ENABLE = not DEPTH_CORRECTION_ENABLE
                    self.perspective_corrector.enable_depth_correction = DEPTH_CORRECTION_ENABLE
                    print(f"深度校正: {'啟用' if DEPTH_CORRECTION_ENABLE else '停用'}")

        except KeyboardInterrupt:
            print("程序被用戶中斷.")
            if self.is_counting_active and self.collected_net_speeds and not self.output_generated_for_session:
                print("中斷程序但有未處理的數據. 正在生成輸出.")
                self._generate_outputs_async()
                self.output_generated_for_session = True
        finally:
            self.running = False
            print("正在關閉系統...")
            self.reader.stop()
            print("影像讀取器已停止.")
            self.file_writer_executor.shutdown(wait=True)
            print("檔案寫入器已停止.")
            cv2.destroyAllWindows()
            print("系統關閉完成.")


def main():
    global DEPTH_CORRECTION_ENABLE
    global BALL_SIZE_CORRECTION_ENABLE
    
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器 v11.1 進階深度校正版')
    parser.add_argument('--video', type=str, default=None, help='視頻檔路徑. 如果不指定則使用網路攝影機.')
    parser.add_argument('--camera_idx', type=int, default=DEFAULT_CAMERA_INDEX, help='網路攝影機索引.')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='目標幀率.')
    parser.add_argument('--width', type=int, default=DEFAULT_FRAME_WIDTH, help='影像寬度.')
    parser.add_argument('--height', type=int, default=DEFAULT_FRAME_HEIGHT, help='影像高度.')
    parser.add_argument('--table_len', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='桌球桌長度 (cm).')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='球體偵測超時 (秒).')
    
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION_DEFAULT,
                        choices=['left_to_right', 'right_to_left', 'both'], help='網中心穿越方向.')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS_TO_COLLECT, help='每個會話收集的速度數.')
    
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM_DEFAULT, help='ROI近端實際寬度 (cm).')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM_DEFAULT, help='ROI遠端實際寬度 (cm).')
    
    parser.add_argument('--depth_corr', type=float, default=DEPTH_CORRECTION_FACTOR, help='深度校正係數 (預設1.6).')
    parser.add_argument('--depth_curve', type=float, default=DEPTH_CORRECTION_CURVE, help='深度校正曲線 (預設2.0).')
    parser.add_argument('--disable_depth_corr', action='store_true', help='停用深度校正.')
    parser.add_argument('--disable_size_corr', action='store_true', help='停用球體大小校正.')
    
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE_DEFAULT, help='啟用調試輸出.')

    args = parser.parse_args()
    
    # 根據命令列參數設置全局變數
    if args.disable_depth_corr:
        DEPTH_CORRECTION_ENABLE = False
    
    if args.disable_size_corr:
        BALL_SIZE_CORRECTION_ENABLE = False

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
        far_width_cm=args.far_width,
        depth_correction_factor=args.depth_corr,
        depth_correction_curve=args.depth_curve
    )
    tracker.run()

if __name__ == '__main__':
    main()