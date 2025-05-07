#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v9 - CUDA 加速版
# 整合 CUDA 加速功能，加快影像處理和分析速度

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

# —— 全局參數設定 ——
# 基本設定
DEFAULT_CAMERA_INDEX = 0  # 預設相機索引
DEFAULT_TARGET_FPS = 120  # 預設目標 FPS
DEFAULT_FRAME_WIDTH = 1920  # 預設影像寬度
DEFAULT_FRAME_HEIGHT = 1080  # 預設影像高度
DEFAULT_TABLE_LENGTH_CM = 142  # 乒乓球桌長度，單位 cm

# 偵測相關參數
DEFAULT_DETECTION_TIMEOUT = 0.05  # 球體偵測超時，超過此時間將重置軌跡
DEFAULT_ROI_START_RATIO = 0.4  # ROI 區域開始比例 (左側)
DEFAULT_ROI_END_RATIO = 0.6  # ROI 區域結束比例 (右側)
DEFAULT_ROI_BOTTOM_RATIO = 0.8  # ROI 區域底部比例 (排除底部 10%)
MAX_TRAJECTORY_POINTS = 80  # 最大軌跡點數

# 中心線偵測參數
CENTER_LINE_WIDTH = 20  # 中心線寬度 (像素)
CENTER_DETECTION_COOLDOWN = 0.2  # 中心點偵測冷卻時間 (秒)
MAX_NET_SPEEDS = 100  # 紀錄的最大網中心速度數量
NET_CROSSING_DIRECTION = 'left_to_right'  # 'left_to_right' or 'right_to_left' or 'both'
AUTO_STOP_AFTER_COLLECTION = False  # 修改：不要自動停止程序
OUTPUT_FOLDER = 'real_time_output'  # 輸出資料夾名稱

# 透視校正參數
NEAR_SIDE_WIDTH_CM = 29  # 較近側的實際寬度（公分）
FAR_SIDE_WIDTH_CM = 72   # 較遠側的實際寬度（公分）

# FMO (Fast Moving Object) 相關參數
MAX_PREV_FRAMES = 8  # 保留前幾幀的最大數量
OPENING_KERNEL_SIZE = (2, 2)  # 開運算內核大小
CLOSING_KERNEL_SIZE = (30, 30)  # 閉運算內核大小
THRESHOLD_VALUE = 8  # 二值化閾值

# 球體偵測參數
MIN_BALL_AREA = 10  # 最小球體面積
MAX_BALL_AREA = 7000  # 最大球體面積
MIN_CIRCULARITY = 0.5  # 最小圓度閾值，用於多球情況的過濾

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

# CUDA 相關參數
USE_CUDA = True  # 是否使用 CUDA 加速
CUDA_STREAM = None  # CUDA 流物件，用於異步操作
CUDA_BUFFER_POOL = None  # CUDA 緩衝池，用於重用 GPU 記憶體

# 調試參數
DEBUG_MODE = False  # 是否啟用調試模式

# —— 啟用最佳化與多線程 ——
cv2.setUseOptimized(True)
cv2.setNumThreads(10)

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM, 
                 detection_timeout=DEFAULT_DETECTION_TIMEOUT, use_video_file=False, target_fps=DEFAULT_TARGET_FPS,
                 debug_mode=DEBUG_MODE, use_cuda=USE_CUDA):
        """
        初始化乒乓球速度追蹤器
        
        Args:
            video_source: 視訊來源，可以是相機索引或影片檔案路徑
            table_length_cm: 球桌長度 (cm)，用於像素到實際距離的轉換
            detection_timeout: 偵測超時時間 (秒)，超過此時間會重置軌跡
            use_video_file: 是否使用影片檔案作為輸入
            target_fps: 目標 FPS
            debug_mode: 是否啟用調試模式
            use_cuda: 是否使用 CUDA 加速
        """
        # 設置調試模式
        self.debug_mode = debug_mode
        
        # 設置 CUDA 使用
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            try:
                # 初始化 CUDA 環境
                self._init_cuda()
                print(f"CUDA 加速已啟用：使用 {cv2.cuda.getDevice()} 號 GPU")
                print(f"GPU 型號：{cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())}")
            except Exception as e:
                print(f"CUDA 初始化失敗，切換到 CPU 模式: {e}")
                self.use_cuda = False
        
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
        
        # 初始化 CUDA 用的形態學內核
        if self.use_cuda:
            self.opening_kernel_gpu = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, cv2.CV_8UC1, 
                cv2.getStructuringElement(cv2.MORPH_RECT, OPENING_KERNEL_SIZE)
            )
            self.closing_kernel_gpu = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_CLOSE, cv2.CV_8UC1, 
                cv2.getStructuringElement(cv2.MORPH_RECT, CLOSING_KERNEL_SIZE)
            )
        else:
            self.opening_kernel = np.ones(OPENING_KERNEL_SIZE, np.uint8)
            self.closing_kernel = np.ones(CLOSING_KERNEL_SIZE, np.uint8)

        # 影片檔專用計數器
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # 網中心速度追蹤
        self.center_x = self.frame_width // 2
        self.center_line_start = self.center_x - CENTER_LINE_WIDTH // 2
        self.center_line_end = self.center_x + CENTER_LINE_WIDTH // 2
        self.net_speeds = []  # 儲存經過網中心點的速度
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
        self.relative_times = []     # 存儲每顆球的相對時間
        
        # 透視校正相關參數
        self.near_side_width_cm = NEAR_SIDE_WIDTH_CM  # 較近側的實際寬度（公分）
        self.far_side_width_cm = FAR_SIDE_WIDTH_CM    # 較遠側的實際寬度（公分）
        self.perspective_ratio = self.far_side_width_cm / self.near_side_width_cm  # 透視比例
        self.roi_height = self.roi_end_y  # ROI區域高度，用於透視計算

        # 性能計時器
        self.processing_times = deque(maxlen=100)  # 用於計算平均處理時間
    
    def _init_cuda(self):
        """初始化 CUDA 環境和資源"""
        global CUDA_STREAM, CUDA_BUFFER_POOL
        
        # 選擇第一個可用的 CUDA 裝置
        cv2.cuda.setDevice(0)
        
        # 創建 CUDA 流，用於異步操作
        CUDA_STREAM = cv2.cuda.Stream()
        
        # 創建 CUDA 緩衝池，用於管理 GPU 記憶體
        CUDA_BUFFER_POOL = cv2.cuda.BufferPool()
    
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
        預處理影像幀，使用 CUDA 加速（如果可用）
        
        Args:
            frame: 原始影像幀
            
        Returns:
            tuple: (ROI 區域, 灰階圖像)
        """
        # 擷取 ROI 區域，同時限制左右和底部
        roi = frame[:self.roi_end_y, self.roi_start_x:self.roi_end_x]
        
        # 使用 CUDA 進行灰度轉換（如果可用）
        if self.use_cuda:
            # 上傳到 GPU
            gpu_roi = cv2.cuda_GpuMat()
            gpu_roi.upload(roi)
            
            # 轉換為灰階
            gpu_gray = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2GRAY)
            
            # 下載回 CPU
            gray = gpu_gray.download()
        else:
            # CPU 模式的灰度轉換
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 保存前幀以用於移動物體偵測
        self.prev_frames.append(gray)
        return roi, gray

    def detect_fmo(self):
        """
        使用快速移動物體偵測 (FMO) 方法偵測移動物體，支援 CUDA 加速
        
        Returns:
            mask: 移動物體遮罩或 None（如果幀數不足）
        """
        # 需要至少三幀才能進行偵測
        if len(self.prev_frames) < 3:
            return None
            
        # 取得最近三幀
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        
        # CUDA 模式
        if self.use_cuda:
            # 上傳到 GPU
            gpu_f1 = cv2.cuda_GpuMat()
            gpu_f2 = cv2.cuda_GpuMat()
            gpu_f3 = cv2.cuda_GpuMat()
            
            gpu_f1.upload(f1)
            gpu_f2.upload(f2)
            gpu_f3.upload(f3)
            
            # 計算連續幀之間的差異
            gpu_diff1 = cv2.cuda.absdiff(gpu_f1, gpu_f2)
            gpu_diff2 = cv2.cuda.absdiff(gpu_f2, gpu_f3)
            
            # 使用位元運算找出共同的移動區域
            gpu_mask = cv2.cuda.bitwise_and(gpu_diff1, gpu_diff2)
            
            # 二值化處理，保留顯著變化
            gpu_thresh = cv2.cuda.threshold(gpu_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
            
            # 形態學處理：開運算去除雜訊，閉運算填補物體空洞
            gpu_opening = self.opening_kernel_gpu.apply(gpu_thresh)
            gpu_closing = self.closing_kernel_gpu.apply(gpu_opening)
            
            # 下載結果回 CPU
            mask = gpu_closing.download()
            
            # 釋放 GPU 資源
            gpu_f1.release()
            gpu_f2.release()
            gpu_f3.release()
            gpu_diff1.release()
            gpu_diff2.release()
            gpu_mask.release()
            gpu_thresh.release()
            gpu_opening.release()
            gpu_closing.release()
        else:
            # CPU 模式的處理
            # 計算連續幀之間的差異
            diff1 = cv2.absdiff(f1, f2)
            diff2 = cv2.absdiff(f2, f3)
            
            # 使用位元運算找出共同的移動區域
            mask = cv2.bitwise_and(diff1, diff2)
            
            # 二值化處理，保留顯著變化
            _, thresh = cv2.threshold(mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
            
            # 形態學處理：開運算去除雜訊，閉運算填補物體空洞
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
            mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)
        
        return mask

    def detect_ball(self, roi, mask):
        """
        在 ROI 區域中偵測乒乓球，支援多球情境處理
        
        Args:
            roi: ROI 區域圖像
            mask: 移動物體遮罩
            
        Returns:
            tuple: ((球體中心 x, y), 球體輪廓) 或 (None, None)
        """
        # 找出遮罩中的所有輪廓
        # CUDA 目前不支援 findContours 函數，所以無論是否啟用 CUDA，這一步都在 CPU 上執行
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
                    
                    # 計算圓度（圓度接近1表示是較圓的物體，更可能是球）
                    perimeter = cv2.arcLength(c, True)
                    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # 添加到潛在球體列表
                    potential_balls.append({
                        'position': (cx, cy),
                        'original_x': cx_orig,
                        'contour': c,
                        'area': area,
                        'circularity': circularity
                    })
        
        # 如果找到了潛在球體
        if potential_balls:
            # 應用多條件過濾，選擇最可能的球體
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
                
                return (cx, cy), best_ball['contour']
        
        # 未偵測到合適的球體
        return None, None
    
    def _select_best_ball(self, potential_balls):
        """
        基於多條件選擇最佳球體
        
        Args:
            potential_balls: 潛在球體列表，每個球體為包含位置、輪廓等資訊的字典
            
        Returns:
            dict: 最佳球體資訊或 None
        """
        if not potential_balls:
            return None
            
        # 如果軌跡為空，選擇最圓的球體作為起始點
        if len(self.trajectory) == 0:
            best_ball = max(potential_balls, key=lambda ball: ball['circularity'])
            if best_ball['circularity'] > MIN_CIRCULARITY:
                return best_ball
            return potential_balls[0]  # 如果沒有夠圓的，返回第一個
        
        # 獲取最近一個軌跡點
        last_x, last_y, _ = self.trajectory[-1]
        
        # 計算每個潛在球體與最後軌跡點的距離
        for ball in potential_balls:
            x, y = ball['position']
            orig_x = ball['original_x']
            # 計算與最後軌跡點的距離
            ball['distance'] = math.hypot(orig_x - last_x, y - last_y)
            
            # 如果距離太遠，可能是不相關的球
            if ball['distance'] > self.frame_width * 0.2:  # 超過畫面寬度的20%視為太遠
                ball['distance'] = float('inf')  # 設為無限大表示不可能是同一個球
        
        # 根據距離排序
        potential_balls.sort(key=lambda ball: ball['distance'])
        
        # 返回距離最近且圓度合理的球體
        for ball in potential_balls:
            if ball['circularity'] > MIN_CIRCULARITY:  # 圓度閾值
                return ball
        
        # 如果沒有滿足條件的，返回距離最近的
        return potential_balls[0] if potential_balls else None

    def toggle_counting(self):
        """切換計數狀態"""
        if not self.is_counting:
            # 開始計數
            self.is_counting = True
            self.net_speeds = []  # 清空速度列表
            self.relative_times = []  # 清空相對時間列表
            self.timing_started = False  # 重置時間標記
            self.first_ball_time = None  # 重置第一球時間
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
        檢查球是否經過中心線，並記錄相對時間和速度
        
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
            
        # 判斷移動方向
        moving_left_to_right = ball_x > self.last_ball_x
        moving_right_to_left = ball_x < self.last_ball_x
        direction = ball_x - self.last_ball_x
        
        # 判斷是否穿過中心線
        crossed_left_to_right = (self.last_ball_x < self.center_line_end and ball_x >= self.center_line_end)
        crossed_right_to_left = (self.last_ball_x > self.center_line_start and ball_x <= self.center_line_start)
        
        # 預測是否將穿過中心線（增強檢測能力）
        will_cross = False
        if len(self.trajectory) >= 2:
            # 根據當前運動方向預測下一個位置
            next_x = ball_x + direction
            
            # 檢查是否會穿過中心線
            if (self.last_ball_x < self.center_x and next_x >= self.center_x) or \
               (self.last_ball_x > self.center_x and next_x <= self.center_x):
                will_cross = True
        
        # 根據設定的方向過濾
        record_crossing = False
        
        if NET_CROSSING_DIRECTION == 'left_to_right' and (crossed_left_to_right or (will_cross and direction > 0)):
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'right_to_left' and (crossed_right_to_left or (will_cross and direction < 0)):
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'both' and (crossed_left_to_right or crossed_right_to_left or will_cross):
            record_crossing = True
            
        # 如果符合穿越條件，記錄速度
        if record_crossing and self.ball_speed > 0:
            # 處理相對時間
            if not self.timing_started:
                self.timing_started = True
                self.first_ball_time = timestamp
                relative_time = 0.0
            else:
                relative_time = round(timestamp - self.first_ball_time, 2)  # 精確到小數點後兩位
            
            # 記錄資料
            self.crossed_center = True
            self.last_net_speed = self.ball_speed
            self.net_speeds.append(self.ball_speed)
            self.relative_times.append(relative_time)
            self.last_net_detection_time = timestamp
            
            print(f"記錄速度 #{len(self.net_speeds)}: {self.ball_speed:.1f} km/h, 時間: {relative_time}秒")
            
            # 如果達到目標次數，生成輸出並停止計數
            if len(self.net_speeds) >= MAX_NET_SPEEDS and not self.output_generated:
                print(f"已達到目標次數 ({MAX_NET_SPEEDS})，生成輸出並停止計數")
                self.generate_outputs()
                self.is_counting = False  # 停止計數
                self.output_generated = True
                
        # 更新上一幀的球體位置
        self.last_ball_x = ball_x

    def calculate_speed(self):
        """計算球體速度（公里/小時），使用透視校正"""
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
        根據透視校正計算實際距離
        
        Args:
            x1, y1: 第一個點的座標
            x2, y2: 第二個點的座標
            
        Returns:
            float: 實際距離（公分）
        """
        # 計算兩點的像素距離
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        
        # 根據 y 座標（代表深度）計算每個點的像素-公分轉換比例
        ratio1 = self._get_pixel_to_cm_ratio(y1)
        ratio2 = self._get_pixel_to_cm_ratio(y2)
        
        # 使用兩點的平均比例轉換像素距離為實際距離
        avg_ratio = (ratio1 + ratio2) / 2
        real_distance_cm = pixel_distance * avg_ratio
        
        if self.debug_mode:
            print(f"透視校正: 像素距離={pixel_distance:.1f}, 轉換比例1={ratio1:.4f}, 比例2={ratio2:.4f}, 實際距離={real_distance_cm:.2f}cm")
        
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
        
        # 建立輸出資料夾
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            
        # 建立包含起始點的圖表資料
        plot_times = [0.0] + self.relative_times
        plot_speeds = [0.0] + self.net_speeds
        
        # 生成並保存折線圖
        plt.figure(figsize=(10, 6))
        plt.plot(plot_times, plot_speeds, marker='o', linestyle='-')
        
        # 在每個點上標註數值
        for i, (t, s) in enumerate(zip(plot_times, plot_speeds)):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
        
        plt.title(f'Table Tennis Net Speed Record (Session {self.count_session})')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (km/h)')
        plt.grid(True)
        chart_filename = f'{OUTPUT_FOLDER}/speed_chart_{timestamp}.png'
        plt.savefig(chart_filename)
        plt.close()
        
        # 將速度數據保存到TXT檔案 (不包含初始零點)
        txt_filename = f'{OUTPUT_FOLDER}/speed_data_{timestamp}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Table Tennis Net Speed Record (km/h) - Session {self.count_session}\n")
            f.write("----------------------------------\n")
            for i, (speed, rel_time) in enumerate(zip(self.net_speeds, self.relative_times), 1):
                f.write(f"{rel_time}s: {speed:.1f} km/h\n")
            
            # 添加統計資訊
            avg_speed = sum(self.net_speeds) / len(self.net_speeds)
            max_speed = max(self.net_speeds)
            min_speed = min(self.net_speeds)
            f.write("\n----------------------------------\n")
            f.write(f"Average: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum: {max_speed:.1f} km/h\n")
            f.write(f"Minimum: {min_speed:.1f} km/h\n")
            
            # 添加處理性能資訊
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times) * 1000
                f.write(f"Average processing time: {avg_time:.2f} ms/frame\n")
                f.write(f"Using CUDA: {'Yes' if self.use_cuda else 'No'}\n")
        
        # 將速度數據保存到CSV檔案 (不包含初始零點)
        csv_filename = f'{OUTPUT_FOLDER}/speed_data_{timestamp}.csv'
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
                
        print(f"輸出已保存到 {OUTPUT_FOLDER} 資料夾")
        print(f"- 折線圖: {chart_filename}")
        print(f"- 文字檔: {txt_filename}")
        print(f"- CSV檔: {csv_filename}")

    def draw_visualizations(self, frame, roi, ball_position=None, ball_contour=None):
        """
        繪製視覺化元素
        
        Args:
            frame: 原始影像幀
            roi: ROI 區域
            ball_position: 球體中心位置 (可能為 None)
            ball_contour: 球體輪廓 (可能為 None)
        """
        # 繪製 ROI 邊界
        cv2.line(frame, (self.roi_start_x, 0), (self.roi_start_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(frame, (self.roi_end_x, 0), (self.roi_end_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(frame, (0, self.roi_end_y), (self.frame_width, self.roi_end_y), ROI_COLOR, 2)

        # 繪製中心線
        cv2.line(frame, (self.center_x, 0), (self.center_x, self.frame_height), CENTER_LINE_COLOR, 2)

        # 繪製球體軌跡
        if len(self.trajectory) >= 2:
            pts = np.array([(p[0], p[1]) for p in self.trajectory], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, TRAJECTORY_COLOR, 2)

        # 如果有偵測到球，繪製球體與其輪廓
        if ball_position:
            cv2.circle(roi, ball_position, 5, BALL_COLOR, -1)
            cv2.drawContours(roi, [ball_contour], 0, CONTOUR_COLOR, 2)

        # —— 常駐顯示目前球速與 FPS ——
        cv2.putText(
            frame,
            f"Current Speed: {self.ball_speed:.1f} km/h",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            SPEED_TEXT_COLOR,
            FONT_THICKNESS
        )
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FPS_TEXT_COLOR,
            FONT_THICKNESS
        )
        
        # —— 顯示計數狀態 ——
        count_status = "ON" if self.is_counting else "OFF"
        count_color = (0, 255, 0) if self.is_counting else (0, 0, 255)
        cv2.putText(
            frame,
            f"Counting: {count_status}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            count_color,
            FONT_THICKNESS
        )
        
        # —— 顯示網中心速度 ——
        if self.last_net_speed > 0:
            cv2.putText(
                frame,
                f"Net Speed: {self.last_net_speed:.1f} km/h",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                NET_SPEED_TEXT_COLOR,
                FONT_THICKNESS
            )
            
        # —— 顯示目前已記錄的網中心速度數量 ——
        cv2.putText(
            frame,
            f"Recorded: {len(self.net_speeds)}/{MAX_NET_SPEEDS}",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            NET_SPEED_TEXT_COLOR,
            FONT_THICKNESS
        )
        
        # —— 顯示最後記錄時間 ——
        if self.timing_started and len(self.relative_times) > 0:
            cv2.putText(
                frame,
                f"Last Time: {self.relative_times[-1]:.2f}s",
                (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                NET_SPEED_TEXT_COLOR,
                FONT_THICKNESS
            )
        
        # —— 顯示處理性能 ——
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times) * 1000
            cv2.putText(
                frame,
                f"Process: {avg_time:.1f} ms ({self.use_cuda and 'CUDA' or 'CPU'})",
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
        
        # —— 顯示指導訊息 ——
        cv2.putText(
            frame,
            "Press SPACE to toggle counting, ESC or q to quit",
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )

    def check_timeout(self):
        """檢查是否超過偵測超時時間，若是則重置軌跡和速度"""
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0
            self.crossed_center = False

    def run(self):
        """執行主循環"""
        print("=== 乒乓球速度追蹤器 v9 CUDA 加速版 ===")
        print(f"CUDA 加速: {'啟用' if self.use_cuda else '禁用'}")
        print("按下空白鍵開始/停止計數")
        print("按下 'd' 鍵切換調試模式")
        print("按下 'q' 或 ESC 鍵退出程序")
        print(f"使用透視校正: 近端寬度 {self.near_side_width_cm}cm, 遠端寬度 {self.far_side_width_cm}cm")
        
        while True:
            # 如果應該退出，則跳出循環
            if self.should_exit:
                break
            
            # 處理幀開始時間
            start_process_time = time.time()
                
            # 讀取影像幀
            ret, frame = self.cap.read()
            if not ret:
                print("Camera/Video end.")
                # 如果影片結束但尚未輸出結果，仍然生成輸出
                if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                    self.generate_outputs()
                break
                
            # 更新幀計數與 FPS
            self.frame_count += 1
            if not self.use_video_file:
                self.update_fps()
                
            # 前處理影像
            roi, gray = self.preprocess_frame(frame)
            
            # 偵測快速移動物體
            mask = self.detect_fmo()
            
            # 偵測球體並計算速度
            if mask is not None:
                ball_pos, ball_cnt = self.detect_ball(roi, mask)
                self.calculate_speed()
            else:
                ball_pos, ball_cnt = None, None
                
            # 繪製視覺化效果
            self.draw_visualizations(frame, roi, ball_pos, ball_cnt)
            
            # 檢查是否超時
            self.check_timeout()
            
            # 計算處理時間
            process_time = time.time() - start_process_time
            self.processing_times.append(process_time)

            # 顯示影像
            cv2.imshow('Ping Pong Speed', frame)
            
            # 按鍵處理
            frame_interval_ms = int(1000 / self.fps)
            key = cv2.waitKey(frame_interval_ms) & 0xFF
            
            if key == ord(' '):  # 空白鍵，切換計數狀態
                self.toggle_counting()
            elif key == ord('d'):  # 'd' 鍵，切換調試模式
                self.debug_mode = not self.debug_mode
                print(f"調試模式: {'開啟' if self.debug_mode else '關閉'}")
            elif key == ord('c'):  # 'c' 鍵，切換 CUDA 模式（如果可用）
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.use_cuda = not self.use_cuda
                    print(f"CUDA 模式: {'開啟' if self.use_cuda else '關閉'}")
                    # 清空處理時間統計，重新收集新的數據
                    self.processing_times.clear()
                else:
                    print("沒有可用的 CUDA 裝置")
            elif key in (ord('q'), 27):  # 'q' 或 ESC 鍵退出
                # 如果用戶手動退出但尚未輸出結果，仍然生成輸出
                if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                    self.generate_outputs()
                break

            # 精準控制幀率
            elapsed = time.time() - start_process_time
            sleep_time = max(0, 1.0 / self.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 釋放資源
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 如果有使用 CUDA，釋放 CUDA 資源
        if self.use_cuda:
            cv2.cuda.resetDevice()


def main():
    """主函數：解析命令列參數並啟動追蹤器"""
    global NET_CROSSING_DIRECTION, MAX_NET_SPEEDS, USE_CUDA
    
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器 - CUDA 加速版')
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
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM,
                        help='ROI 較近側的實際寬度 (cm)')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM,
                        help='ROI 較遠側的實際寬度 (cm)')
    parser.add_argument('--cuda', action='store_true', default=USE_CUDA, 
                        help='是否使用 CUDA 加速，若無 CUDA 裝置則自動降級為 CPU 模式')
    args = parser.parse_args()

    # 設置全局穿越方向和參數
    NET_CROSSING_DIRECTION = args.direction
    MAX_NET_SPEEDS = args.count
    USE_CUDA = args.cuda

    # 檢查是否有可用的 CUDA 裝置
    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if USE_CUDA and not cuda_available:
        print("警告：沒有可用的 CUDA 裝置，自動切換到 CPU 模式")
        USE_CUDA = False

    # 根據命令列參數初始化追蹤器
    if args.video:
        tracker = PingPongSpeedTracker(
            args.video, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=True,
            debug_mode=args.debug,
            use_cuda=USE_CUDA
        )
    else:
        tracker = PingPongSpeedTracker(
            args.camera, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=False, 
            target_fps=args.fps,
            debug_mode=args.debug,
            use_cuda=USE_CUDA
        )
    
    # 設置透視校正參數
    tracker.near_side_width_cm = args.near_width
    tracker.far_side_width_cm = args.far_width
    tracker.perspective_ratio = tracker.far_side_width_cm / tracker.near_side_width_cm
        
    # 啟動追蹤器
    tracker.run()


if __name__ == '__main__':
    main()