#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v11 Enhanced
# 方向感知版本：優化球體追蹤，降低背景干擾，實現更靈活的方向優先機制

import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import argparse
import matplotlib.pyplot as plt
import os
import csv
import threading
import queue
import concurrent.futures

# —— 全局參數設定 ——
# 基本設定
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 120
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_TABLE_LENGTH_CM = 142  # 乒乓球桌長度，單位 cm

# 偵測相關參數
DEFAULT_DETECTION_TIMEOUT = 0.05  # 球體偵測超時，超過此時間將重置軌跡
DEFAULT_ROI_START_RATIO = 0.2  # ROI 區域開始比例 (左側) - 從0.3調整到0.2以更完整捕獲左側
DEFAULT_ROI_END_RATIO = 0.7  # ROI 區域結束比例 (右側) - 從0.65擴大到0.7以確保完整捕獲
DEFAULT_ROI_BOTTOM_RATIO = 0.9  # ROI 區域底部比例 (從0.8擴大到0.9，捕獲更多區域)
MAX_TRAJECTORY_POINTS = 80  # 最大軌跡點數

# 方向優先參數 (新增/修改)
DIRECTION_PREFERENCE = True  # 是否啟用方向優先機制（較柔性的機制，不完全過濾）
MIN_DIRECTION_CONFIDENCE = 0.30  # 最低方向置信度 (從0.60降低到0.30更寬鬆)
DIRECTION_HISTORY_SIZE = 5  # 方向歷史紀錄大小
LEFT_TO_RIGHT_BIAS = 1.3  # 左到右方向的權重偏好值(從1.5降至1.3)

# 背景處理參數（新增）
USE_BACKGROUND_SUBTRACTION = True  # 是否啟用背景減除
BACKGROUND_HISTORY = 500  # 背景模型歷史幀數
BACKGROUND_THRESHOLD = 400  # 背景減除閾值
LEARNING_RATE = 0.01  # 背景學習率

# 中心線偵測參數
CENTER_LINE_WIDTH = 30  # 中心線寬度 (像素) - 從20增大到30使檢測更容易
CENTER_DETECTION_COOLDOWN = 0.10  # 中心點偵測冷卻時間 (秒) - 從0.15減小到0.10提高靈敏度
MAX_NET_SPEEDS = 100  # 紀錄的最大網中心速度數量
NET_CROSSING_DIRECTION = 'left_to_right'  # 默認只記錄左到右方向
AUTO_STOP_AFTER_COLLECTION = False  # 不自動停止程序
OUTPUT_FOLDER = 'real_time_output'  # 輸出資料夾名稱

# 透視校正參數
NEAR_SIDE_WIDTH_CM = 29  # 較近側的實際寬度（公分）
FAR_SIDE_WIDTH_CM = 72   # 較遠側的實際寬度（公分）

# FMO (Fast Moving Object) 相關參數
MAX_PREV_FRAMES = 5  # 保留前幾幀的最大數量
OPENING_KERNEL_SIZE = (3, 3)  # 開運算內核大小 - 從(5,5)減小到(3,3)保留更多細節
CLOSING_KERNEL_SIZE = (10, 10)  # 閉運算內核大小 - 從(15,15)減小到(10,10)
THRESHOLD_VALUE = 6  # 二值化閾值 - 從8調整到6提高靈敏度

# 球體偵測參數
MIN_BALL_AREA = 8  # 最小球體面積 - 從10減小到8，捕獲更小的球
MAX_BALL_AREA = 7000  # 最大球體面積
MIN_CIRCULARITY = 0.4  # 最小圓度閾值 - 從0.5減小到0.4

# 速度計算參數
SPEED_SMOOTHING = 0.5  # 速度平滑因子
KMH_CONVERSION = 0.036  # 轉換為公里/小時的係數

# FPS 計算參數
FPS_SMOOTHING = 0.4  # FPS 平滑因子
MAX_FRAME_TIMES = 20  # FPS 計算用的最大時間樣本數

# 視覺化參數
TRAJECTORY_COLOR = (0, 0, 255)  # 軌跡顏色 (BGR)
BALL_COLOR = (0, 255, 255)  # 球體顏色 (BGR)
CONTOUR_COLOR = (255, 0, 0)  # 輪廓顏色 (BGR)
ROI_COLOR = (0, 255, 0)  # ROI 邊界顏色 (BGR)
SPEED_TEXT_COLOR = (0, 0, 255)  # 速度文字顏色 (BGR)
FPS_TEXT_COLOR = (0, 255, 0)  # FPS 文字顏色 (BGR)
CENTER_LINE_COLOR = (0, 255, 255)  # 中心線顏色 (BGR)
NET_SPEED_TEXT_COLOR = (255, 0, 0)  # 網中心速度文字顏色 (BGR)
FONT_SCALE = 1  # 文字大小
FONT_THICKNESS = 2  # 文字粗細

# 多執行緒參數
MAX_QUEUE_SIZE = 10  # 最大佇列大小
VISUALIZATION_INTERVAL = 2  # 每隔多少幀繪製一次完整視覺元素

# 事件緩衝區參數
EVENT_BUFFER_SIZE = 20  # 事件緩衝區大小
PREDICTION_LOOKAHEAD = 5  # 預測未來多少幀

# 調試參數
DEBUG_MODE = False  # 是否啟用調試模式
SHOW_MASK = False  # 是否顯示遮罩視窗（用於調試）

# —— 啟用最佳化與多線程 ——
cv2.setUseOptimized(True)
cv2.setNumThreads(10)

class FrameData:
    """用於在執行緒間傳遞幀資料的結構"""
    def __init__(self, frame=None, roi=None, ball_position=None, ball_contour=None, ball_speed=0, 
                 fps=0, is_counting=False, net_speeds=None, last_net_speed=0, 
                 relative_times=None, display_text=None, frame_count=0, mask=None):
        self.frame = frame
        self.roi = roi
        self.ball_position = ball_position
        self.ball_contour = ball_contour
        self.ball_speed = ball_speed
        self.fps = fps
        self.is_counting = is_counting
        self.net_speeds = net_speeds if net_speeds is not None else []
        self.last_net_speed = last_net_speed
        self.relative_times = relative_times if relative_times is not None else []
        self.display_text = display_text
        self.frame_count = frame_count
        self.trajectory_points = []
        self.direction_confidence = 0.0  # 方向置信度
        self.mask = mask  # 用於調試的遮罩

class EventRecord:
    """用於記錄潛在中心線穿越事件的結構"""
    def __init__(self, ball_x, timestamp, speed, predicted=False):
        self.ball_x = ball_x
        self.timestamp = timestamp
        self.speed = speed
        self.predicted = predicted
        self.processed = False

class DirectionVector:
    """表示運動方向向量的類"""
    def __init__(self, dx=0, dy=0, confidence=0.0):
        self.dx = dx
        self.dy = dy
        self.confidence = confidence
        # 計算水平方向
        if dx > 0:
            self.horizontal = "right"
        elif dx < 0:
            self.horizontal = "left"
        else:
            self.horizontal = "none"

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM, 
                 detection_timeout=DEFAULT_DETECTION_TIMEOUT, use_video_file=False, target_fps=DEFAULT_TARGET_FPS,
                 debug_mode=DEBUG_MODE, direction_preference=DIRECTION_PREFERENCE,
                 use_background_subtraction=USE_BACKGROUND_SUBTRACTION,
                 show_mask=SHOW_MASK):
        """初始化乒乓球速度追蹤器"""
        # 設置調試模式
        self.debug_mode = debug_mode
        self.show_mask = show_mask
        
        # 設置方向優先
        self.direction_preference = direction_preference
        
        # 設置背景減除
        self.use_background_subtraction = use_background_subtraction
        
        # 初始化視訊捕獲
        self.cap = cv2.VideoCapture(video_source)
        self.use_video_file = use_video_file
        self._setup_capture(target_fps)
        
        # 設定參數
        self.table_length_cm = table_length_cm
        self.detection_timeout = detection_timeout
        self.pixels_per_cm = self.frame_width / table_length_cm
        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_end_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        
        # 初始化軌跡與速度追蹤
        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.ball_speed = 0
        self.last_detection_time = time.time()
        
        # FMO 前置處理
        self.prev_frames = deque(maxlen=MAX_PREV_FRAMES)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE)
        
        # 初始化背景減除器
        if self.use_background_subtraction:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=BACKGROUND_HISTORY, 
                varThreshold=BACKGROUND_THRESHOLD, 
                detectShadows=False
            )

        # 影片檔專用計數器
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # 網中心速度追蹤
        self.center_x = self.frame_width // 2
        self.center_line_start = self.center_x - CENTER_LINE_WIDTH // 2
        self.center_line_end = self.center_x + CENTER_LINE_WIDTH // 2
        self.net_speeds = []  # 儲存經過網中心點的速度
        self.relative_times = []  # 存儲每顆球的相對時間
        self.last_net_detection_time = 0  # 上次經過網中心的時間
        self.last_net_speed = 0  # 最後一次經過網中心的速度
        self.crossed_center = False  # 是否剛經過中心線
        self.last_ball_x = None  # 上次球的x座標
        self.output_generated = False  # 是否已產生輸出結果
        self.should_exit = False  # 是否應該退出程序
        
        # 控制計數開關的變量
        self.is_counting = False  # 初始狀態不計數
        self.count_session = 0  # 計數會話編號
        
        # 時間索引化相關變數
        self.timing_started = False  # 是否已開始計時
        self.first_ball_time = None  # 第一顆球的時間戳記
        
        # 透視校正相關參數
        self.near_side_width_cm = NEAR_SIDE_WIDTH_CM
        self.far_side_width_cm = FAR_SIDE_WIDTH_CM
        self.perspective_ratio = self.far_side_width_cm / self.near_side_width_cm
        self.roi_height = self.roi_end_y
        
        # 執行狀態
        self.running = False
        
        # 事件緩衝區
        self.event_buffer = deque(maxlen=EVENT_BUFFER_SIZE)
        
        # 方向偵測相關參數
        self.direction_history = deque(maxlen=DIRECTION_HISTORY_SIZE)
        self.current_direction = DirectionVector()
        
        # 預計算的畫面元素
        self._setup_precalculated_elements()
        
        # 初始化不同版本的遮罩以便調試
        self.debug_masks = {}

    def _setup_precalculated_elements(self):
        """預先計算一些固定的畫面元素，避免每幀都重新計算"""
        # 建立透明覆蓋層用於繪製固定元素
        self.overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # 繪製 ROI 邊界
        cv2.line(self.overlay, (self.roi_start_x, 0), (self.roi_start_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(self.overlay, (self.roi_end_x, 0), (self.roi_end_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(self.overlay, (0, self.roi_end_y), (self.frame_width, self.roi_end_y), ROI_COLOR, 2)
        
        # 繪製中心線
        cv2.line(self.overlay, (self.center_x, 0), (self.center_x, self.frame_height), CENTER_LINE_COLOR, 2)
        
        # 建立指導訊息文字
        self.instruction_text = "Press SPACE to toggle counting, D for debug, M for masks, ESC or q to quit"
        
        # 創建透視校正查表
        self._create_perspective_lookup()
    
    def _create_perspective_lookup(self):
        """創建透視校正的查表，避免反覆計算"""
        self.perspective_lookup = {}
        # 對常見的y座標預先計算比例
        for y in range(0, self.roi_end_y, 10):  # 每10個像素計算一次
            relative_y = min(1, max(0, y / self.roi_height))
            near_ratio = self.table_length_cm / self.frame_width * (self.near_side_width_cm / self.table_length_cm)
            far_ratio = self.table_length_cm / self.frame_width * (self.far_side_width_cm / self.table_length_cm)
            ratio = near_ratio * relative_y + far_ratio * (1 - relative_y)
            self.perspective_lookup[y] = ratio
    
    def _setup_capture(self, target_fps):
        """設置視訊捕獲的參數"""
        if not self.use_video_file:
            # 設定 webcam 參數
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 確認 FPS 是否合理，否則使用手動計算
            if self.fps <= 0 or self.fps > 1000:
                self.fps = target_fps
                self.manual_fps_calc = True
                self.frame_times = deque(maxlen=MAX_FRAME_TIMES)
            else:
                self.manual_fps_calc = False
        else:
            # 讀取影片檔的 FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.manual_fps_calc = False
        
        # 讀取影像解析度
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_fps(self):
        """更新 FPS 計算（用於手動計算 FPS 的模式）"""
        if not self.manual_fps_calc:
            return
            
        now = time.time()
        self.frame_times.append(now)
        
        if len(self.frame_times) >= 2:
            dt = self.frame_times[-1] - self.frame_times[0]
            if dt > 0:
                measured = (len(self.frame_times) - 1) / dt
                # 使用平滑化的 FPS 計算
                self.fps = (1 - FPS_SMOOTHING) * self.fps + FPS_SMOOTHING * measured
                
        self.last_frame_time = now

    def preprocess_frame(self, frame):
        """
        預處理影像幀 - 使用更高效的 OpenCV 函數
        
        Args:
            frame: 原始影像幀
            
        Returns:
            tuple: (ROI 區域, 灰階圖像)
        """
        # 先嘗試使用標準 NumPy 切片
        roi = frame[:self.roi_end_y, self.roi_start_x:self.roi_end_x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 嘗試使用 GPU 加速灰階轉換，如果支援的話
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # 使用 CUDA 加速灰階轉換
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                gpu_processed = cv2.cuda.cvtColor(gpu_gray, cv2.COLOR_BGR2GRAY)
                gray = gpu_processed.download()
        except Exception as e:
            if self.debug_mode:
                print(f"GPU 加速失敗，使用 CPU 處理: {e}")
        
        # 保存前幀以用於移動物體偵測
        self.prev_frames.append(gray)
        return roi, gray

    def detect_fmo(self, roi):
        """
        使用快速移動物體偵測 (FMO) 方法偵測移動物體 - 改進版本
        結合背景減除和方向偏好機制
        
        Args:
            roi: ROI 區域圖像
            
        Returns:
            tuple: (過濾後的遮罩, 方向置信度, 調試用遮罩字典)
        """
        debug_masks = {}  # 存儲不同階段的遮罩，用於調試
        
        # 需要至少三幀才能進行偵測
        if len(self.prev_frames) < 3:
            return None, 0, debug_masks
            
        # 取得最近三幀
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        
        # 1. 使用 numpy 運算而非 OpenCV 運算來加速差異計算
        diff1 = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).astype(np.uint8)
        diff2 = np.abs(f2.astype(np.int16) - f3.astype(np.int16)).astype(np.uint8)
        
        debug_masks['diff1'] = diff1.copy()
        debug_masks['diff2'] = diff2.copy()
        
        # 2. 使用位元運算找出共同的移動區域
        motion_mask = cv2.bitwise_and(diff1, diff2)
        debug_masks['motion_mask'] = motion_mask.copy()
        
        # 3. 背景減除 (如果啟用)
        if self.use_background_subtraction:
            fg_mask = self.bg_subtractor.apply(f3, learningRate=LEARNING_RATE)
            # 結合運動遮罩和前景遮罩
            combined_mask = cv2.bitwise_and(motion_mask, fg_mask)
            debug_masks['fg_mask'] = fg_mask.copy()
            debug_masks['combined_mask'] = combined_mask.copy()
        else:
            combined_mask = motion_mask
            
        # 4. 二值化處理
        try:
            _, thresh = cv2.threshold(combined_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except:
            # 如果 OTSU 失敗，回退到固定閾值
            _, thresh = cv2.threshold(combined_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        debug_masks['thresh'] = thresh.copy()
        
        # 5. 形態學處理 - 使用更高效的結構元素
        # 如果開運算大小為0，則跳過
        if OPENING_KERNEL_SIZE[0] > 0 and OPENING_KERNEL_SIZE[1] > 0:
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
        else:
            opening = thresh
            
        debug_masks['opening'] = opening.copy()
            
        # 使用 elliptical 結構元素進行閉運算，更接近球形
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)
        debug_masks['closing'] = closing.copy()
        
        # 6. 分析移動方向 (用於方向優先機制)
        direction_confidence = 0.0
        if self.direction_preference:
            # 分析移動方向 - 將圖像分為左右兩半
            height, width = diff1.shape
            left_half_width = width // 2
            
            # 計算左右區域的差異總和
            left_sum_diff1 = np.sum(diff1[:, :left_half_width])
            right_sum_diff1 = np.sum(diff1[:, left_half_width:])
            left_sum_diff2 = np.sum(diff2[:, :left_half_width])
            right_sum_diff2 = np.sum(diff2[:, left_half_width:])
            
            # 判斷方向：left_to_right 的特徵是 diff1 左側活動較多，diff2 右側活動較多
            is_left_to_right = (left_sum_diff1 > right_sum_diff1 * 0.7) and (right_sum_diff2 > left_sum_diff2 * 0.7)
            
            # 建立方向置信度值
            if left_sum_diff1 > 0 and right_sum_diff2 > 0:
                left_right_ratio1 = left_sum_diff1 / (left_sum_diff1 + right_sum_diff1) if (left_sum_diff1 + right_sum_diff1) > 0 else 0
                right_left_ratio2 = right_sum_diff2 / (left_sum_diff2 + right_sum_diff2) if (left_sum_diff2 + right_sum_diff2) > 0 else 0
                direction_confidence = (left_right_ratio1 + right_left_ratio2) / 2
            else:
                direction_confidence = 0
                
            # 更新方向歷史
            self.direction_history.append(DirectionVector(
                dx=1 if is_left_to_right else (-1 if not is_left_to_right else 0),
                dy=0,  # 暫不考慮垂直方向
                confidence=direction_confidence
            ))
            
            # 計算整體方向
            self._update_overall_direction()
            
            # 顯示方向偵測結果
            if self.debug_mode:
                print(f"左右運動比例 - 第一幀左:{left_sum_diff1:.0f} 右:{right_sum_diff1:.0f} " + 
                      f"第二幀左:{left_sum_diff2:.0f} 右:{right_sum_diff2:.0f}, 方向置信度:{direction_confidence:.2f}")
        
        return closing, direction_confidence, debug_masks
    
    def _update_overall_direction(self):
        """更新綜合運動方向"""
        if not self.direction_history:
            self.current_direction = DirectionVector()
            return
            
        # 計算加權平均方向
        dx_sum = 0
        dy_sum = 0
        confidence_sum = 0
        total_weight = 0
        
        for i, direction in enumerate(self.direction_history):
            # 最近的方向有更高權重
            weight = (i + 1) * direction.confidence
            # 左到右方向有額外的偏好權重
            if direction.dx > 0:
                weight *= LEFT_TO_RIGHT_BIAS
                
            dx_sum += direction.dx * weight
            dy_sum += direction.dy * weight
            confidence_sum += direction.confidence
            total_weight += weight
        
        if total_weight > 0:
            avg_dx = dx_sum / total_weight
            avg_dy = dy_sum / total_weight
            avg_confidence = confidence_sum / len(self.direction_history)
            
            self.current_direction = DirectionVector(
                dx=avg_dx,
                dy=avg_dy,
                confidence=avg_confidence
            )
        else:
            self.current_direction = DirectionVector()

    def detect_ball(self, roi, mask):
        """
        在 ROI 區域中偵測乒乓球 - 效能優化和方向感知版本
        
        Args:
            roi: ROI 區域圖像
            mask: 移動物體遮罩
            
        Returns:
            tuple: ((球體中心 x, y), 球體輪廓) 或 (None, None)
        """
        # 方向偏好機制 - 不再完全過濾，而是在選擇球時考慮方向因素
        current_direction = self.current_direction
        
        # 嘗試使用更高效的連通區域分析替代輪廓檢測
        try:
            # 使用 connectedComponentsWithStats 獲取連通區域統計信息
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # 排除背景（索引0）
            potential_balls = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if MIN_BALL_AREA < area < MAX_BALL_AREA:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    cx, cy = centroids[i]
                    
                    # 計算圓度（使用寬高比近似）
                    circularity = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                    
                    # 從標籤創建遮罩以獲取更準確的圓度
                    obj_mask = np.zeros_like(mask)
                    obj_mask[labels == i] = 255
                    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # 計算更準確的圓度
                        contour = contours[0]
                        contour_area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            better_circularity = 4 * math.pi * contour_area / (perimeter * perimeter)
                            circularity = better_circularity  # 使用更準確的圓度
                    
                    # 將相關信息添加到潛在球列表
                    potential_balls.append({
                        'position': (int(cx), int(cy)),
                        'original_x': int(cx) + self.roi_start_x,
                        'area': area,
                        'circularity': circularity,
                        'label': i,
                        'width': width,
                        'height': height,
                        'contour': contours[0] if contours else None
                    })
                    
            # 如果找到潛在球體
            if potential_balls:
                best_ball = self._select_best_ball(potential_balls)
                if best_ball:
                    cx, cy = best_ball['position']
                    cx_orig = best_ball['original_x']
                    
                    # 取得輪廓，如果已有則使用，否則創建
                    if 'contour' in best_ball and best_ball['contour'] is not None:
                        contour = best_ball['contour']
                    else:
                        # 創建一個圓形的近似輪廓
                        radius = int(math.sqrt(best_ball['area'] / math.pi))
                        contour = np.array([[[cx + int(radius * math.cos(a)), cy + int(radius * math.sin(a))]] 
                                         for a in np.linspace(0, 2*math.pi, 20)])
                    
                    # 更新最後偵測時間和處理球體狀態
                    self.last_detection_time = time.time()
                    ts = (self.frame_count / self.fps) if self.use_video_file else time.time()
                    
                    # 如果計數標誌開啟，檢測是否經過中心線
                    if self.is_counting:
                        self.check_center_crossing(cx_orig, ts)
                    
                    # 保存軌跡點
                    self.trajectory.append((cx_orig, cy, ts))
                    
                    if self.debug_mode:
                        print(f"偵測到球：位置({cx_orig}, {cy})，面積:{best_ball['area']:.1f}，圓度:{best_ball['circularity']:.3f}")
                    
                    return (cx, cy), contour
        
        except Exception as e:
            if self.debug_mode:
                print(f"連通區域分析失敗: {e}，回退到傳統輪廓檢測")
            
            # 回退到傳統輪廓檢測方法
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 存儲所有可能的球體
            potential_balls = []
            
            # 按面積大小排序輪廓
            for c in sorted(cnts, key=cv2.contourArea, reverse=True):
                area = cv2.contourArea(c)
                
                # 根據面積過濾，保留可能是球的輪廓
                if MIN_BALL_AREA < area < MAX_BALL_AREA:
                    # 計算輪廓的中心點
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cx_orig = cx + self.roi_start_x
                        
                        # 計算圓度
                        perimeter = cv2.arcLength(c, True)
                        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        # 計算寬高比
                        x, y, w, h = cv2.boundingRect(c)
                        
                        # 添加到潛在球體列表
                        potential_balls.append({
                            'position': (cx, cy),
                            'original_x': cx_orig,
                            'contour': c,
                            'area': area,
                            'circularity': circularity,
                            'width': w,
                            'height': h
                        })
            
            # 如果找到了潛在球體
            if potential_balls:
                best_ball = self._select_best_ball(potential_balls)
                if best_ball:
                    cx, cy = best_ball['position']
                    cx_orig = best_ball['original_x']
                    
                    # 更新最後偵測時間
                    self.last_detection_time = time.time()
                    
                    # 根據輸入源選擇適當的時間戳
                    ts = (self.frame_count / self.fps) if self.use_video_file else time.time()
                    
                    # 檢測是否經過中心線（只有當計數標誌開啟時才進行）
                    if self.is_counting:
                        self.check_center_crossing(cx_orig, ts)
                    
                    # 保存軌跡點
                    self.trajectory.append((cx_orig, cy, ts))
                    
                    if self.debug_mode:
                        print(f"偵測到球：位置({cx_orig}, {cy})，面積:{best_ball['area']:.1f}，圓度:{best_ball['circularity']:.3f}")
                    
                    return (cx, cy), best_ball.get('contour', None)
        
        # 未偵測到合適的球體
        return None, None
    
    def _select_best_ball(self, potential_balls):
        """
        基於多條件選擇最佳球體 - 方向感知優化版
        
        Args:
            potential_balls: 潛在球體列表
            
        Returns:
            dict: 最佳球體資訊或 None
        """
        if not potential_balls:
            return None
            
        # 過濾出符合最低圓度要求的球體
        circular_balls = [ball for ball in potential_balls if ball.get('circularity', 0) > MIN_CIRCULARITY]
        
        # 如果沒有足夠圓的球體，則使用所有球體
        valid_balls = circular_balls if circular_balls else potential_balls
            
        # 如果軌跡為空，選擇最圓的球體作為起始點
        if len(self.trajectory) == 0:
            # 返回最圓的球體
            return max(valid_balls, key=lambda ball: ball.get('circularity', 0))
        
        # 獲取最近一個軌跡點
        last_x, last_y, _ = self.trajectory[-1]
        
        # 計算當前主要運動方向
        expected_dx = self.current_direction.dx if self.direction_preference else 0
        
        # 計算每個潛在球體與最後軌跡點的距離、運動一致性和方向偏好
        for ball in valid_balls:
            x, y = ball['position']
            orig_x = ball['original_x']
            
            # 基礎分數項：圓度
            ball['roundness_score'] = ball.get('circularity', 0)
            
            # 計算與最後軌跡點的直線距離
            distance = math.hypot(orig_x - last_x, y - last_y)
            ball['distance'] = distance
            
            # 距離分數（越近越好）
            if distance < self.frame_width * 0.2:  # 不是太遠
                ball['distance_score'] = 1.0 / (1.0 + distance / 50)  # 距離越小分數越高
            else:
                ball['distance_score'] = 0  # 太遠，視為不可能是同一個球
            
            # 計算 x 方向移動
            dx = orig_x - last_x
            
            # 方向偏好分數 - 如果符合期望方向(左到右)則給予更高的分數
            if self.direction_preference and NET_CROSSING_DIRECTION == 'left_to_right':
                if dx > 0:  # 從左到右移動
                    ball['direction_score'] = 0.5 + 0.5 * min(1, dx / 100)  # 移動越多分數越高
                else:  # 從右到左移動
                    ball['direction_score'] = 0.2  # 給較低但非零的分數
            else:
                ball['direction_score'] = 0.5  # 不考慮方向時給中立分數
            
            # 如果有足夠的軌跡點，檢查運動一致性
            if len(self.trajectory) >= 3:
                # 計算過去的移動向量
                x1, y1, _ = self.trajectory[-2]
                x2, y2, _ = self.trajectory[-1]
                past_dx = x2 - x1
                past_dy = y2 - y1
                
                # 計算當前點與過去向量的一致性
                current_dx = orig_x - x2
                current_dy = y - y2
                
                # 向量點積除以向量模長，值越接近1表示越一致
                dot_product = past_dx * current_dx + past_dy * current_dy
                past_mag = math.sqrt(past_dx**2 + past_dy**2)
                current_mag = math.sqrt(current_dx**2 + current_dy**2)
                
                if past_mag > 0 and current_mag > 0:
                    consistency = dot_product / (past_mag * current_mag)
                    # 運動一致性分數
                    ball['consistency_score'] = max(0, consistency)  # 只考慮正向一致性
                else:
                    ball['consistency_score'] = 0
            else:
                ball['consistency_score'] = 0.5  # 沒有足夠歷史時給中立分數
            
            # 計算綜合分數，賦予不同權重
            ball['score'] = (
                ball['roundness_score'] * 0.25 +  # 圓度
                ball['distance_score'] * 0.35 +   # 距離(最重要)
                ball['direction_score'] * 0.25 +  # 方向偏好
                ball['consistency_score'] * 0.15   # 運動一致性
            )
            
            if self.debug_mode:
                print(f"球體評分: 圓度:{ball['roundness_score']:.2f}, 距離:{ball['distance_score']:.2f}, " +
                      f"方向:{ball['direction_score']:.2f}, 一致性:{ball.get('consistency_score', 0):.2f}, " +
                      f"總分:{ball['score']:.2f}")
        
        # 根據分數排序
        valid_balls.sort(key=lambda ball: ball.get('score', 0), reverse=True)
        
        # 返回分數最高的球體
        if valid_balls and valid_balls[0].get('score', 0) > 0.1:  # 確保有足夠高的分數
            return valid_balls[0]
        return None

    def toggle_counting(self):
        """切換計數狀態"""
        if not self.is_counting:
            # 開始計數
            self.is_counting = True
            self.net_speeds = []  # 清空速度列表
            self.relative_times = []  # 清空相對時間列表
            self.timing_started = False  # 重置時間標記
            self.first_ball_time = None  # 重置第一球時間
            self.event_buffer.clear()  # 清空事件緩衝區
            self.output_generated = False  # 重置輸出狀態
            self.count_session += 1  # 增加會話編號
            print(f"開始計數 (會話 #{self.count_session}) - 目標收集 {MAX_NET_SPEEDS} 個速度值")
        else:
            # 如果已經在計數，但想要中止當前計數並重新開始
            self.is_counting = False
            if len(self.net_speeds) > 0:
                print(f"中止計數 - 已收集 {len(self.net_speeds)} 個速度值")
                # 如果有收集到資料，輸出結果
                self.generate_outputs()
            else:
                print("中止計數 - 未收集到任何速度資料")
                
    def check_center_crossing(self, ball_x, timestamp):
        """
        檢查球是否經過中心線 - 加強版本
        
        Args:
            ball_x: 球的 x 座標
            timestamp: 當前時間戳
        """
        # 初始化 last_ball_x
        if self.last_ball_x is None:
            self.last_ball_x = ball_x
            return
            
        # 當前時間與上次偵測中心線的時間差
        time_since_last_detection = timestamp - self.last_net_detection_time
        
        # 檢查是否冷卻時間已過，避免同一次穿越被多次記錄
        if time_since_last_detection < CENTER_DETECTION_COOLDOWN:
            self.last_ball_x = ball_x
            return
        
        # 記錄中心穿越事件到緩衝區
        self._record_potential_crossing(ball_x, timestamp)
        
        # 更新上一幀的球體位置
        self.last_ball_x = ball_x
    
    def _record_potential_crossing(self, ball_x, timestamp):
        """
        記錄潛在的中心線穿越事件到緩衝區，並進行預測
        
        Args:
            ball_x: 球的 x 座標
            timestamp: 當前時間戳
        """
        # 判斷移動方向
        direction = ball_x - self.last_ball_x
        
        # 只關注左到右方向的移動（如果設定為左到右）
        if NET_CROSSING_DIRECTION == 'left_to_right' and direction <= 0:
            if self.debug_mode:
                print(f"忽略非左到右方向的移動: 當前位置 {ball_x}, 上次位置 {self.last_ball_x}")
            return
        
        # 判斷是否穿過中心線
        crossed_left_to_right = (self.last_ball_x < self.center_line_end and ball_x >= self.center_line_end)
        crossed_right_to_left = (self.last_ball_x > self.center_line_start and ball_x <= self.center_line_start)
        
        # 根據設定的方向確定是否記錄穿越
        record_crossing = False
        if NET_CROSSING_DIRECTION == 'left_to_right' and crossed_left_to_right:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'right_to_left' and crossed_right_to_left:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'both' and (crossed_left_to_right or crossed_right_to_left):
            record_crossing = True
        
        # 如果是實際穿越，將事件添加到緩衝區
        if record_crossing and self.ball_speed > 0:
            event = EventRecord(ball_x, timestamp, self.ball_speed, predicted=False)
            self.event_buffer.append(event)
            # 處理事件列表（實際記錄速度）
            self._process_crossing_events()
            return
        
        # 預測性穿越檢測（比實際穿越更寬松的條件）
        # 檢查是否即將穿過中心線
        will_cross = False
        time_to_cross = 0
        
        if len(self.trajectory) >= 2 and abs(direction) > 3:  # 確保有明顯的運動
            # 計算當前速度和方向
            last_x, _, last_t = self.trajectory[-2]
            curr_x, _, curr_t = self.trajectory[-1]
            
            if curr_t > last_t:  # 避免除以零
                # 計算 x 方向速度 (像素/秒)
                x_velocity = (curr_x - last_x) / (curr_t - last_t)
                
                # 根據網中心方向設定來檢查預測穿越
                if NET_CROSSING_DIRECTION == 'left_to_right' and x_velocity > 0:
                    # 檢查是否會從左到右穿過中心線
                    if ball_x < self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps >= self.center_x:
                        will_cross = True
                        time_to_cross = abs((self.center_x - ball_x) / x_velocity) if x_velocity != 0 else 0
                        
                elif NET_CROSSING_DIRECTION == 'right_to_left' and x_velocity < 0:
                    # 檢查是否會從右到左穿過中心線
                    if ball_x > self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps <= self.center_x:
                        will_cross = True
                        time_to_cross = abs((self.center_x - ball_x) / x_velocity) if x_velocity != 0 else 0
                        
                elif NET_CROSSING_DIRECTION == 'both':
                    # 檢查任一方向的穿越
                    if (x_velocity > 0 and ball_x < self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps >= self.center_x) or \
                       (x_velocity < 0 and ball_x > self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps <= self.center_x):
                        will_cross = True
                        time_to_cross = abs((self.center_x - ball_x) / x_velocity) if x_velocity != 0 else 0
        
        # 如果預測會穿越且球速有效，添加預測事件
        if will_cross and self.ball_speed > 0 and time_to_cross < 0.5:  # 只預測半秒內的穿越
            predicted_time = timestamp + time_to_cross
            
            # 檢查是否已經有相似的預測事件
            for event in self.event_buffer:
                if event.predicted and abs(event.timestamp - predicted_time) < 0.1:
                    # 已有相似預測，更新而非新增
                    event.timestamp = predicted_time  # 更新預測時間
                    event.speed = self.ball_speed  # 更新速度
                    return
            
            # 添加新的預測事件
            event = EventRecord(ball_x, predicted_time, self.ball_speed, predicted=True)
            self.event_buffer.append(event)
    
    def _process_crossing_events(self):
        """處理事件緩衝區中的穿越事件，記錄有效的速度數據"""
        current_time = time.time()
        events_to_process = []
        
        # 收集實際穿越事件和超時的預測事件
        for event in self.event_buffer:
            if not event.processed:
                if not event.predicted:
                    # 實際穿越事件
                    events_to_process.append(event)
                elif current_time - event.timestamp > 0.2:
                    # 超時的預測事件（給予0.2秒等待實際事件替換）
                    events_to_process.append(event)
        
        # 按時間排序要處理的事件
        events_to_process.sort(key=lambda e: e.timestamp)
        
        # 處理排序後的事件
        for event in events_to_process:
            # 標記為已處理
            event.processed = True
            
            # 處理相對時間
            if not self.timing_started:
                self.timing_started = True
                self.first_ball_time = event.timestamp
                relative_time = 0.0
            else:
                relative_time = round(event.timestamp - self.first_ball_time, 2)  # 精確到小數點後兩位
            
            # 記錄速度和時間
            self.crossed_center = True
            self.last_net_speed = event.speed
            self.net_speeds.append(event.speed)
            self.relative_times.append(relative_time)
            self.last_net_detection_time = event.timestamp
            
            status = "預測" if event.predicted else "實際"
            print(f"記錄速度 #{len(self.net_speeds)}: {event.speed:.1f} km/h, 時間: {relative_time}秒 ({status}穿越)")
            
            # 如果達到目標次數，生成輸出並停止計數
            if len(self.net_speeds) >= MAX_NET_SPEEDS and not self.output_generated:
                print(f"已達到目標次數 ({MAX_NET_SPEEDS})，生成輸出並停止計數")
                self.generate_outputs()
                self.is_counting = False  # 停止計數
                self.output_generated = True
                break

    def calculate_speed(self):
        """計算球體速度（公里/小時），使用透視校正 - 效能優化版"""
        if len(self.trajectory) < 2:
            return
            
        # 取最近兩個軌跡點
        p1, p2 = self.trajectory[-2], self.trajectory[-1]
        
        # 提取座標和時間
        x1, y1, t1 = p1
        x2, y2, t2 = p2
        
        # 使用透視校正計算實際距離
        dist_cm = self._calculate_real_distance(x1, y1, x2, y2)
        
        # 計算時間差
        dt = t2 - t1
        
        if dt > 0:
            # 計算速度（公里/小時）
            speed = dist_cm / dt * KMH_CONVERSION
            
            # 平滑化速度數值
            if self.ball_speed > 0:
                self.ball_speed = (1 - SPEED_SMOOTHING) * self.ball_speed + SPEED_SMOOTHING * speed
            else:
                self.ball_speed = speed
            
            if self.debug_mode:
                print(f"速度計算: 距離={dist_cm:.2f}cm, 時間={dt:.4f}s, 速度={speed:.1f}km/h, 平滑後={self.ball_speed:.1f}km/h")

    def _calculate_real_distance(self, x1, y1, x2, y2):
        """
        根據透視校正計算實際距離 - 使用查表法優化
        
        Args:
            x1, y1: 第一個點的座標
            x2, y2: 第二個點的座標
            
        Returns:
            float: 實際距離（公分）
        """
        # 計算兩點的像素距離
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        
        # 使用查表法獲取像素-公分轉換比例
        # 對 y 座標進行四捨五入到最近的 10 倍數
        y1_rounded = round(y1 / 10) * 10
        y2_rounded = round(y2 / 10) * 10
        
        # 查表獲取轉換比例，如果不在表中，則計算
        if y1_rounded in self.perspective_lookup:
            ratio1 = self.perspective_lookup[y1_rounded]
        else:
            ratio1 = self._get_pixel_to_cm_ratio(y1)
            
        if y2_rounded in self.perspective_lookup:
            ratio2 = self.perspective_lookup[y2_rounded]
        else:
            ratio2 = self._get_pixel_to_cm_ratio(y2)
        
        # 使用兩點的平均比例轉換像素距離為實際距離
        avg_ratio = (ratio1 + ratio2) / 2
        real_distance_cm = pixel_distance * avg_ratio
        
        return real_distance_cm
    
    def _get_pixel_to_cm_ratio(self, y):
        """
        根據 y 座標計算像素到公分的轉換比例
        
        Args:
            y: 點的 y 座標
            
        Returns:
            float: 像素到公分的轉換比例
        """
        # 計算相對位置（0為頂部/遠端，1為底部/近端）
        relative_y = min(1, max(0, y / self.roi_height))
        
        # 線性插值計算轉換比例
        near_ratio = self.table_length_cm / self.frame_width * (self.near_side_width_cm / self.table_length_cm)
        far_ratio = self.table_length_cm / self.frame_width * (self.far_side_width_cm / self.table_length_cm)
        
        # 反向插值：relative_y 為 1 時使用 near_ratio，為 0 時使用 far_ratio
        pixel_to_cm_ratio = near_ratio * relative_y + far_ratio * (1 - relative_y)
        
        return pixel_to_cm_ratio

    def generate_outputs(self):
        """生成速度數據的輸出：折線圖、TXT和CSV文件，含時間戳記"""
        # 檢查是否有數據可以輸出
        if len(self.net_speeds) == 0:
            print("沒有可輸出的速度數據")
            return
            
        # 獲取當前時間作為時間戳記
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"生成輸出結果: {len(self.net_speeds)} 個速度值")
        
        # 創建輸出目錄位置
        output_dir = f"{OUTPUT_FOLDER}/session_{self.count_session}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 直接在主執行緒中生成圖表
        chart_filename = self._generate_chart(output_dir, timestamp)
        
        # 使用執行緒池處理文本檔案生成（這些不涉及GUI操作）
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 提交 TXT 文件生成任務
            txt_future = executor.submit(self._generate_txt, output_dir, timestamp)
            # 提交 CSV 文件生成任務
            csv_future = executor.submit(self._generate_csv, output_dir, timestamp)
            
            # 等待文本文件任務完成
            txt_filename = txt_future.result()
            csv_filename = csv_future.result()
        
        print(f"輸出已保存到 {output_dir} 資料夾")
        print(f"- 折線圖: {chart_filename}")
        print(f"- 文字檔: {txt_filename}")
        print(f"- CSV檔: {csv_filename}")

    def _generate_chart(self, output_dir, timestamp):
        """生成速度折線圖 - 不從 (0,0) 開始"""
        # 直接使用實際數據點，不添加起始的 (0,0) 點
        plot_times = self.relative_times
        plot_speeds = self.net_speeds
        
        # 計算統計數據用於圖表
        avg_speed = sum(plot_speeds) / len(plot_speeds) if plot_speeds else 0
        max_speed = max(plot_speeds) if plot_speeds else 0
        min_speed = min(plot_speeds) if plot_speeds else 0
        
        # 創建高質量的折線圖
        plt.figure(figsize=(12, 7))
        
        # 繪製主要數據線
        plt.plot(plot_times, plot_speeds, 'o-', linewidth=2, markersize=8, color='#3498db', label='Speed(km/h)')
        
        # 添加平均線
        plt.axhline(y=avg_speed, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Average: {avg_speed:.1f} km/h')
        
        # 在每個點上標註數值
        for i, (t, s) in enumerate(zip(plot_times, plot_speeds)):
            plt.annotate(f"{s:.1f}", (t, s), 
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none", alpha=0.7))
        
        # 設置圖表標題和標籤
        plt.title(f'Table Tennis Net Speed Record (方向優先版)', fontsize=16, pad=20)
        plt.xlabel('Time(s)', fontsize=12, labelpad=10)
        plt.ylabel('Speed(km/h)', fontsize=12, labelpad=10)
        
        # 改進圖表外觀
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        
        # 調整坐標軸範圍，給數據點周圍留出一些空間
        if plot_times:
            x_min = min(plot_times)
            x_max = max(plot_times)
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.5
            plt.xlim(x_min - x_margin, x_max + x_margin)
            
            y_min = min(min_speed * 0.9, avg_speed * 0.9)
            y_max = max(max_speed * 1.1, avg_speed * 1.1)
            plt.ylim(y_min, y_max)
        
        # 添加圖表說明文字
        plt.figtext(0.02, 0.02, f"總數: {len(plot_speeds)} | 最大值: {max_speed:.1f} km/h | 最小值: {min_speed:.1f} km/h | 方向: 左到右", 
                   fontsize=9, color='#555555')
        
        # 調整圖表布局
        plt.tight_layout()
        
        # 保存高質量圖表
        chart_filename = f'{output_dir}/speed_chart_{timestamp}.png'
        plt.savefig(chart_filename, dpi=150)
        plt.close()
        
        return chart_filename

    def _generate_txt(self, output_dir, timestamp):
        """生成 TXT 格式速度數據文件"""
        # 計算統計數據
        avg_speed = sum(self.net_speeds) / len(self.net_speeds)
        max_speed = max(self.net_speeds)
        min_speed = min(self.net_speeds)
        
        # 將速度數據保存到TXT檔案 (不包含初始零點)
        txt_filename = f'{output_dir}/speed_data_{timestamp}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"桌球網過速度記錄 (km/h) - 會話 {self.count_session}\n")
            f.write("----------------------------------\n")
            f.write(f"方向: {'從左到右' if NET_CROSSING_DIRECTION == 'left_to_right' else '從右到左' if NET_CROSSING_DIRECTION == 'right_to_left' else '雙向'}\n")
            f.write("----------------------------------\n")
            for i, (speed, rel_time) in enumerate(zip(self.net_speeds, self.relative_times), 1):
                f.write(f"{rel_time}s: {speed:.1f} km/h\n")
            
            # 添加統計資訊
            f.write("\n----------------------------------\n")
            f.write(f"平均速度: {avg_speed:.1f} km/h\n")
            f.write(f"最大速度: {max_speed:.1f} km/h\n")
            f.write(f"最小速度: {min_speed:.1f} km/h\n")
            
        return txt_filename

    def _generate_csv(self, output_dir, timestamp):
        """生成 CSV 格式速度數據文件"""
        # 計算統計數據
        avg_speed = sum(self.net_speeds) / len(self.net_speeds)
        max_speed = max(self.net_speeds)
        min_speed = min(self.net_speeds)
        
        # 將速度數據保存到CSV檔案 (不包含初始零點)
        csv_filename = f'{output_dir}/speed_data_{timestamp}.csv'
        with open(csv_filename, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # 寫入標題行
            csv_writer.writerow(["Time(s)", "Speed(km/h)"])
            # 寫入數據
            for i, (speed, rel_time) in enumerate(zip(self.net_speeds, self.relative_times), 1):
                csv_writer.writerow([f"{rel_time:.2f}", f"{speed:.1f}"])
            # 寫入統計資訊
            csv_writer.writerow([])
            csv_writer.writerow(["Statistics", ""])
            csv_writer.writerow(["Average", f"{avg_speed:.1f}"])
            csv_writer.writerow(["Maximum", f"{max_speed:.1f}"])
            csv_writer.writerow(["Minimum", f"{min_speed:.1f}"])
            csv_writer.writerow(["Direction", "Left to Right"])
            
        return csv_filename

    def draw_visualizations(self, frame, frame_data):
        """
        繪製視覺化元素
        
        Args:
            frame: 原始影像幀
            frame_data: 包含要顯示的數據的 FrameData 對象
        """
        # 僅當每 VISUALIZATION_INTERVAL 幀時繪製完整視覺效果
        draw_full = frame_data.frame_count % VISUALIZATION_INTERVAL == 0
        
        # 建立一個乾淨的繪圖圖層
        vis_layer = frame.copy()
        
        # 如果需要繪製完整視覺效果，添加預先計算的元素
        if draw_full:
            # 添加固定的視覺元素
            vis_layer = cv2.addWeighted(vis_layer, 1, self.overlay, 1, 0)
            
            # 繪製球體軌跡
            if frame_data.trajectory_points and len(frame_data.trajectory_points) >= 2:
                pts = np.array(frame_data.trajectory_points, np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_layer, [pts], False, TRAJECTORY_COLOR, 2)
        
        # 如果有偵測到球，繪製球體與其輪廓
        roi_offset_x = self.roi_start_x
        if frame_data.ball_position and frame_data.ball_contour is not None:
            # 畫球體中心點
            cx, cy = frame_data.ball_position
            cv2.circle(frame_data.roi, (cx, cy), 5, BALL_COLOR, -1)
            
            # 畫球體輪廓
            cv2.drawContours(frame_data.roi, [frame_data.ball_contour], 0, CONTOUR_COLOR, 2)
            
            # 在主畫面標記球的位置
            global_cx = cx + roi_offset_x
            cv2.circle(vis_layer, (global_cx, cy), 8, BALL_COLOR, -1)

        # —— 常駐顯示目前球速與 FPS ——
        cv2.putText(
            vis_layer,
            f"Current Speed: {frame_data.ball_speed:.1f} km/h",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            SPEED_TEXT_COLOR,
            FONT_THICKNESS
        )
        cv2.putText(
            vis_layer,
            f"FPS: {frame_data.fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FPS_TEXT_COLOR,
            FONT_THICKNESS
        )
        
        # —— 顯示計數狀態 ——
        count_status = "ON" if frame_data.is_counting else "OFF"
        count_color = (0, 255, 0) if frame_data.is_counting else (0, 0, 255)
        cv2.putText(
            vis_layer,
            f"Counting: {count_status}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            count_color,
            FONT_THICKNESS
        )
        
        # —— 顯示網中心速度 ——
        if frame_data.last_net_speed > 0:
            cv2.putText(
                vis_layer,
                f"Net Speed: {frame_data.last_net_speed:.1f} km/h",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                NET_SPEED_TEXT_COLOR,
                FONT_THICKNESS
            )
            
        # —— 顯示目前已記錄的網中心速度數量 ——
        cv2.putText(
            vis_layer,
            f"Recorded: {len(frame_data.net_speeds)}/{MAX_NET_SPEEDS}",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            NET_SPEED_TEXT_COLOR,
            FONT_THICKNESS
        )
        
        # —— 顯示最後記錄時間 ——
        if len(frame_data.relative_times) > 0:
            cv2.putText(
                vis_layer,
                f"Last Time: {frame_data.relative_times[-1]:.2f}s",
                (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                NET_SPEED_TEXT_COLOR,
                FONT_THICKNESS
            )
            
        # —— 顯示方向狀態 ——
        direction_text = "Left to Right"
        confidence = self.current_direction.confidence
        confidence_color = (0, 255, 0) if confidence > MIN_DIRECTION_CONFIDENCE else (0, 165, 255)
        
        cv2.putText(
            vis_layer,
            f"Direction: {direction_text} ({confidence:.2f})",
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            confidence_color,
            FONT_THICKNESS
        )
        
        # —— 顯示指導訊息 ——
        cv2.putText(
            vis_layer,
            self.instruction_text,
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        # 顯示調試相關信息
        if self.debug_mode and frame_data.display_text:
            cv2.putText(
                vis_layer,
                frame_data.display_text,
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                1
            )
        
        return vis_layer

    def check_timeout(self):
        """檢查是否超過偵測超時時間，若是則重置軌跡和速度"""
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0
            self.crossed_center = False

    def process_frame(self, frame):
        """
        處理單個影像幀
        
        Args:
            frame: 原始影像幀
            
        Returns:
            FrameData: 處理結果
        """
        # 更新幀計數與 FPS
        self.frame_count += 1
        if not self.use_video_file:
            self.update_fps()
            
        # 前處理影像
        roi, gray = self.preprocess_frame(frame)
        
        # 偵測快速移動物體 - 優化版
        mask, direction_confidence, debug_masks = self.detect_fmo(roi)
        
        ball_pos = None
        ball_cnt = None
        direction_display = ""
        
        # 偵測球體並計算速度
        if mask is not None:
            ball_pos, ball_cnt = self.detect_ball(roi, mask)
            self.calculate_speed()
            
            # 方向顯示文本
            if self.direction_preference and self.current_direction.confidence > 0:
                direction_display = f"方向置信度: {self.current_direction.confidence:.2f}"
        
        # 檢查是否超時
        self.check_timeout()
        
        # 準備顯示數據
        frame_data = FrameData(
            frame=frame,
            roi=roi,
            ball_position=ball_pos,
            ball_contour=ball_cnt,
            ball_speed=self.ball_speed,
            fps=self.fps,
            is_counting=self.is_counting,
            net_speeds=self.net_speeds.copy(),
            last_net_speed=self.last_net_speed,
            relative_times=self.relative_times.copy(),
            display_text=direction_display,
            frame_count=self.frame_count,
            mask=mask
        )
        
        # 添加軌跡點
        if len(self.trajectory) > 0:
            frame_data.trajectory_points = [(p[0], p[1]) for p in self.trajectory]
            
        # 添加方向置信度
        frame_data.direction_confidence = self.current_direction.confidence if self.current_direction else 0.0
        
        # 添加調試用遮罩
        self.debug_masks = debug_masks
        
        return frame_data

    def display_debug_masks(self):
        """顯示調試用的遮罩"""
        if not self.debug_masks:
            return
            
        # 組合所有遮罩為一個視圖
        mask_rows = []
        mask_names = []
        
        # 將遮罩按行排列
        for name, mask in self.debug_masks.items():
            # 確保遮罩是可顯示的
            if mask is None:
                continue
                
            # 轉換為3通道以便可以添加文字
            if len(mask.shape) == 2:
                display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                display_mask = mask.copy()
                
            # 添加遮罩名稱
            cv2.putText(
                display_mask,
                name,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            
            # 調整大小以便顯示
            display_mask = cv2.resize(display_mask, (320, 240))
            
            mask_rows.append(display_mask)
            mask_names.append(name)
            
            # 每4個遮罩一行
            if len(mask_rows) == 4:
                row = np.hstack(mask_rows)
                cv2.imshow('Debug Masks: ' + ' | '.join(mask_names), row)
                mask_rows = []
                mask_names = []
                
        # 顯示剩餘的遮罩
        if mask_rows:
            # 如果剩餘遮罩不足4個，填充空白區域
            while len(mask_rows) < 4:
                empty_mask = np.zeros((240, 320, 3), dtype=np.uint8)
                mask_rows.append(empty_mask)
                mask_names.append('')
                
            row = np.hstack(mask_rows)
            cv2.imshow('Debug Masks: ' + ' | '.join(mask_names), row)

    def run(self):
        """執行主循環 - macOS 兼容版本"""
        print("=== 乒乓球速度追蹤器 v11 (加強版) ===")
        print(f"追蹤方向: {'從左到右' if NET_CROSSING_DIRECTION == 'left_to_right' else '從右到左' if NET_CROSSING_DIRECTION == 'right_to_left' else '雙向'}")
        print(f"方向優先: {'已啟用' if self.direction_preference else '已停用'}")
        print(f"背景減除: {'已啟用' if self.use_background_subtraction else '已停用'}")
        print("按下空白鍵開始/停止計數")
        print("按下 'd' 鍵切換調試模式")
        print("按下 'm' 鍵顯示/隱藏遮罩視窗")
        print("按下 'f' 鍵切換方向優先")
        print("按下 'q' 或 ESC 鍵退出程序")
        print(f"使用透視校正: 近端寬度 {self.near_side_width_cm}cm, 遠端寬度 {self.far_side_width_cm}cm")
        
        # 標記執行中
        self.running = True
        
        # 建立處理執行緒池
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        try:
            while self.running:
                # 讀取影像幀
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera/Video end.")
                    # 如果影片結束但尚未輸出結果，仍然生成輸出
                    if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                        self.generate_outputs()
                    break
                
                # 將影像處理提交到執行緒池中非同步處理
                future = executor.submit(self.process_frame, frame.copy())
                
                try:
                    # 等待處理完成，最多等待每幀的目標時間的兩倍
                    timeout = 2.0 / max(1, self.fps)
                    frame_data = future.result(timeout=timeout)
                    
                    # 在主執行緒中繪製視覺化效果
                    display_frame = self.draw_visualizations(frame, frame_data)
                    
                    # 在主執行緒中顯示影像 (macOS 需要在主執行緒中進行 UI 操作)
                    cv2.imshow('Ping Pong Speed', display_frame)
                    
                    # 顯示調試遮罩（如果啟用）
                    if self.show_mask and frame_data.mask is not None:
                        cv2.imshow('Detection Mask', frame_data.mask)
                        
                        # 顯示詳細的調試遮罩
                        if self.debug_mode:
                            self.display_debug_masks()
                    
                except concurrent.futures.TimeoutError:
                    # 如果處理超時，顯示原始影像
                    cv2.putText(
                        frame,
                        "Processing timeout - frame skipped",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow('Ping Pong Speed', frame)
                
                # 在主執行緒中檢查按鍵
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空白鍵，切換計數狀態
                    self.toggle_counting()
                elif key == ord('d'):  # 'd' 鍵，切換調試模式
                    self.debug_mode = not self.debug_mode
                    print(f"調試模式: {'開啟' if self.debug_mode else '關閉'}")
                elif key == ord('m'):  # 'm' 鍵，切換遮罩顯示
                    self.show_mask = not self.show_mask
                    print(f"遮罩顯示: {'開啟' if self.show_mask else '關閉'}")
                    if not self.show_mask:
                        # 關閉所有遮罩視窗
                        cv2.destroyWindow('Detection Mask')
                        for i in range(0, 10):  # 假設最多有10個調試遮罩視窗
                            try:
                                cv2.destroyWindow(f'Debug Masks: {i}')
                            except:
                                pass
                elif key == ord('f'):  # 'f' 鍵，切換方向優先
                    self.direction_preference = not self.direction_preference
                    print(f"方向優先: {'開啟' if self.direction_preference else '關閉'}")
                elif key in (ord('q'), 27):  # 'q' 或 ESC 鍵退出
                    # 如果用戶手動退出但尚未輸出結果，仍然生成輸出
                    if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                        self.generate_outputs()
                    break
                
                # 處理事件緩衝區
                if self.is_counting:
                    self._process_crossing_events()
                
        except KeyboardInterrupt:
            print("程序被用戶中斷")
        finally:
            # 釋放資源
            self.running = False
            executor.shutdown(wait=False)
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """主函數：解析命令列參數並啟動追蹤器"""
    global NET_CROSSING_DIRECTION, MAX_NET_SPEEDS, VISUALIZATION_INTERVAL, DIRECTION_PREFERENCE
    global USE_BACKGROUND_SUBTRACTION, SHOW_MASK
    
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器 - 加強版本')
    parser.add_argument('--video', type=str, default='', help='影片檔路徑，留空用 webcam')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX, help='攝影機編號')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='webcam 目標 FPS')
    parser.add_argument('--table_length', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='球桌長度 (cm)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='偵測超時時間 (秒)')
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION, 
                        choices=['left_to_right', 'right_to_left', 'both'], 
                        help='記錄球經過網中心的方向')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS,
                        help='需要收集的速度數據數量')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    parser.add_argument('--show_mask', action='store_true', help='顯示偵測遮罩')
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM,
                        help='ROI 較近側的實際寬度 (cm)')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM,
                        help='ROI 較遠側的實際寬度 (cm)')
    parser.add_argument('--interval', type=int, default=VISUALIZATION_INTERVAL,
                        help='視覺化間隔，指定每隔多少幀繪製一次完整視覺元素')
    parser.add_argument('--no_direction_pref', action='store_false', dest='direction_preference',
                        help='關閉方向優先機制')
    parser.add_argument('--no_bg_subtract', action='store_false', dest='use_background_subtraction',
                        help='關閉背景減除')
    parser.set_defaults(direction_preference=DIRECTION_PREFERENCE)
    parser.set_defaults(use_background_subtraction=USE_BACKGROUND_SUBTRACTION)
    parser.set_defaults(show_mask=SHOW_MASK)
    
    args = parser.parse_args()

    # 設置全局參數
    NET_CROSSING_DIRECTION = args.direction
    MAX_NET_SPEEDS = args.count
    VISUALIZATION_INTERVAL = args.interval
    DIRECTION_PREFERENCE = args.direction_preference
    USE_BACKGROUND_SUBTRACTION = args.use_background_subtraction
    SHOW_MASK = args.show_mask

    # 根據命令列參數初始化追蹤器
    if args.video:
        tracker = PingPongSpeedTracker(
            args.video, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=True,
            debug_mode=args.debug,
            direction_preference=args.direction_preference,
            use_background_subtraction=args.use_background_subtraction,
            show_mask=args.show_mask
        )
    else:
        tracker = PingPongSpeedTracker(
            args.camera, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=False, 
            target_fps=args.fps,
            debug_mode=args.debug,
            direction_preference=args.direction_preference,
            use_background_subtraction=args.use_background_subtraction,
            show_mask=args.show_mask
        )
    
    # 設置透視校正參數
    tracker.near_side_width_cm = args.near_width
    tracker.far_side_width_cm = args.far_width
    tracker.perspective_ratio = tracker.far_side_width_cm / tracker.near_side_width_cm
        
    # 啟動追蹤器
    tracker.run()


if __name__ == '__main__':
    main()