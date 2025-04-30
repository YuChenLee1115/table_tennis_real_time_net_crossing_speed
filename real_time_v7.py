#!/usr/bin/env python3
# 新增按空白鍵開始計數功能，收集完數據後不自動關閉
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
DEFAULT_TARGET_FPS = 60  # 預設目標 FPS
DEFAULT_FRAME_WIDTH = 1920  # 預設影像寬度
DEFAULT_FRAME_HEIGHT = 1080  # 預設影像高度
DEFAULT_TABLE_LENGTH_CM = 100  # 乒乓球桌長度，單位 cm

# 偵測相關參數
DEFAULT_DETECTION_TIMEOUT = 0.15  # 球體偵測超時，超過此時間將重置軌跡
DEFAULT_ROI_START_RATIO = 0.4  # ROI 區域開始比例 (左側)
DEFAULT_ROI_END_RATIO = 0.6  # ROI 區域結束比例 (右側)
DEFAULT_ROI_BOTTOM_RATIO = 0.8  # ROI 區域底部比例 (排除底部 20%)
MAX_TRAJECTORY_POINTS = 80  # 最大軌跡點數

# 新增: 中心線偵測參數
CENTER_LINE_WIDTH = 10  # 中心線寬度 (像素)
CENTER_DETECTION_COOLDOWN = 0.5  # 中心點偵測冷卻時間 (秒)
MAX_NET_SPEEDS = 30  # 紀錄的最大網中心速度數量
NET_CROSSING_DIRECTION = 'left_to_right'  # 'left_to_right' or 'right_to_left' or 'both'
AUTO_STOP_AFTER_COLLECTION = False  # 修改：不要自動停止程序
OUTPUT_FOLDER = 'real_time_output'  # 輸出資料夾名稱

# FMO (Fast Moving Object) 相關參數
MAX_PREV_FRAMES = 7  # 保留前幾幀的最大數量
OPENING_KERNEL_SIZE = (1, 1)  # 開運算內核大小
CLOSING_KERNEL_SIZE = (9, 9)  # 閉運算內核大小
THRESHOLD_VALUE = 8  # 二值化閾值

# 球體偵測參數
MIN_BALL_AREA = 10  # 最小球體面積
MAX_BALL_AREA = 1700  # 最大球體面積

# 速度計算參數
SPEED_SMOOTHING = 0.5  # 速度平滑因子
KMH_CONVERSION = 0.036  # 轉換為公里/小時的係數

# FPS 計算參數
FPS_SMOOTHING = 0.3  # FPS 平滑因子
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

# —— 啟用最佳化與多線程 ——
cv2.setUseOptimized(True)
cv2.setNumThreads(10)

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM, 
                 detection_timeout=DEFAULT_DETECTION_TIMEOUT, use_video_file=False, target_fps=DEFAULT_TARGET_FPS):
        """
        初始化乒乓球速度追蹤器
        
        Args:
            video_source: 視訊來源，可以是相機索引或影片檔案路徑
            table_length_cm: 球桌長度 (cm)，用於像素到實際距離的轉換
            detection_timeout: 偵測超時時間 (秒)，超過此時間會重置軌跡
            use_video_file: 是否使用影片檔案作為輸入
            target_fps: 目標 FPS
        """
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
        self.opening_kernel = np.ones(OPENING_KERNEL_SIZE, np.uint8)
        self.closing_kernel = np.ones(CLOSING_KERNEL_SIZE, np.uint8)

        # 影片檔專用計數器
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # 新增: 網中心速度追蹤
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
        
        # 新增: 控制計數開關的變量
        self.is_counting = False  # 初始狀態不計數
        self.count_session = 0  # 計數會話編號
    
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
        預處理影像幀
        
        Args:
            frame: 原始影像幀
            
        Returns:
            tuple: (ROI 區域, 灰階圖像)
        """
        # 擷取 ROI 區域，同時限制左右和底部
        roi = frame[:self.roi_end_y, self.roi_start_x:self.roi_end_x]
        # 轉換為灰階
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 保存前幀以用於移動物體偵測
        self.prev_frames.append(gray)
        return roi, gray

    def detect_fmo(self):
        """
        使用快速移動物體偵測 (FMO) 方法偵測移動物體
        
        Returns:
            mask: 移動物體遮罩或 None（如果幀數不足）
        """
        # 需要至少三幀才能進行偵測
        if len(self.prev_frames) < 3:
            return None
            
        # 取得最近三幀
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        
        # 計算連續幀之間的差異
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        
        # 使用位元運算找出共同的移動區域
        mask = cv2.bitwise_and(diff1, diff2)
        
        # 二值化處理，保留顯著變化
        _, thresh = cv2.threshold(mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # 形態學處理：開運算去除雜訊，閉運算填補物體空洞
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
        return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)

    def detect_ball(self, roi, mask):
        """
        在 ROI 區域中偵測乒乓球
        
        Args:
            roi: ROI 區域圖像
            mask: 移動物體遮罩
            
        Returns:
            tuple: ((球體中心 x, y), 球體輪廓) 或 (None, None)
        """
        # 找出遮罩中的所有輪廓
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面積大小排序輪廓（由大到小）
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            # 計算輪廓面積
            area = cv2.contourArea(c)
            
            # 根據面積過濾，保留可能是球的輪廓
            if MIN_BALL_AREA < area < MAX_BALL_AREA:
                # 計算輪廓的中心點
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 轉換回原始座標系
                    cx_orig = cx + self.roi_start_x
                    
                    # 更新最後偵測時間
                    self.last_detection_time = time.time()
                    
                    # 根據輸入源選擇適當的時間戳
                    ts = (self.frame_count / self.fps) if self.use_video_file else time.time()
                    
                    # 檢測是否經過中心線（只有當計數標誌開啟時才進行）
                    if self.is_counting:
                        self.check_center_crossing(cx_orig, ts)
                    
                    # 保存軌跡點
                    self.trajectory.append((cx_orig, cy, ts))
                    
                    return (cx, cy), c
                    
        # 未偵測到合適的球體
        return None, None

    def toggle_counting(self):
        """切換計數狀態"""
        if not self.is_counting:
            # 開始計數
            self.is_counting = True
            self.net_speeds = []  # 清空速度列表
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
        檢查球是否經過中心線
        
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
        
        # 判斷是否穿過中心線
        crossed_left_to_right = (self.last_ball_x < self.center_line_end and ball_x >= self.center_line_end)
        crossed_right_to_left = (self.last_ball_x > self.center_line_start and ball_x <= self.center_line_start)
        
        # 根據設定的方向過濾
        record_crossing = False
        
        if NET_CROSSING_DIRECTION == 'left_to_right' and crossed_left_to_right:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'right_to_left' and crossed_right_to_left:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'both' and (crossed_left_to_right or crossed_right_to_left):
            record_crossing = True
            
        # 只有在計數狀態下且符合穿越條件時才記錄速度
        if record_crossing and self.ball_speed > 0:
            self.crossed_center = True
            self.last_net_speed = self.ball_speed
            self.net_speeds.append(self.ball_speed)
            self.last_net_detection_time = timestamp
            
            print(f"記錄速度 #{len(self.net_speeds)}: {self.ball_speed:.1f} km/h")
            
            # 如果達到目標次數，生成輸出並停止計數
            if len(self.net_speeds) >= MAX_NET_SPEEDS and not self.output_generated:
                print(f"已達到目標次數 ({MAX_NET_SPEEDS})，生成輸出並停止計數")
                self.generate_outputs()
                self.is_counting = False  # 停止計數
                self.output_generated = True
                
        # 更新上一幀的球體位置
        self.last_ball_x = ball_x

    def calculate_speed(self):
        """計算球體速度（公里/小時）"""
        if len(self.trajectory) < 2:
            return
            
        # 取最近兩個軌跡點
        p1, p2 = self.trajectory[-2], self.trajectory[-1]
        
        # 計算像素距離
        dp = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
        # 轉換為實際距離（厘米）
        dist_cm = dp / self.pixels_per_cm
        
        # 計算時間差
        dt = p2[2] - p1[2]
        
        if dt > 0:
            # 計算速度（公里/小時）
            speed = dist_cm / dt * KMH_CONVERSION
            
            # 平滑化速度數值
            if self.ball_speed > 0:
                self.ball_speed = (1 - SPEED_SMOOTHING) * self.ball_speed + SPEED_SMOOTHING * speed
            else:
                self.ball_speed = speed

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
            
        # 生成並保存折線圖
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.net_speeds) + 1), self.net_speeds, marker='o', linestyle='-')
        plt.title(f'Table Tennis Net Speed Record (Session {self.count_session})')
        plt.xlabel('Count')
        plt.ylabel('Speed (km/h)')
        plt.grid(True)
        chart_filename = f'{OUTPUT_FOLDER}/speed_chart_{timestamp}.png'
        plt.savefig(chart_filename)
        plt.close()
        
        # 將速度數據保存到TXT檔案
        txt_filename = f'{OUTPUT_FOLDER}/speed_data_{timestamp}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Table Tennis Net Speed Record (km/h) - Session {self.count_session}\n")
            f.write("----------------------------------\n")
            for i, speed in enumerate(self.net_speeds, 1):
                f.write(f"{i}: {speed:.1f} km/h\n")
            
            # 添加統計資訊
            avg_speed = sum(self.net_speeds) / len(self.net_speeds)
            max_speed = max(self.net_speeds)
            min_speed = min(self.net_speeds)
            f.write("\n----------------------------------\n")
            f.write(f"Average: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum: {max_speed:.1f} km/h\n")
            f.write(f"Minimum: {min_speed:.1f} km/h\n")
        
        # 將速度數據保存到CSV檔案
        csv_filename = f'{OUTPUT_FOLDER}/speed_data_{timestamp}.csv'
        with open(csv_filename, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # 寫入標題行
            csv_writer.writerow(["Count", "Speed(km/h)"])
            # 寫入數據
            for i, speed in enumerate(self.net_speeds, 1):
                csv_writer.writerow([i, f"{speed:.1f}"])
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
        print("=== 乒乓球速度追蹤器 v7 ===")
        print("按下空白鍵開始/停止計數")
        print("按下 'q' 或 ESC 鍵退出程序")
        
        while True:
            # 如果應該退出，則跳出循環
            if self.should_exit:
                break
                
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

            # 每幀開始前記錄時間
            start_time = time.time()

            # 顯示影像
            cv2.imshow('Ping Pong Speed', frame)
            
            # 依照目標 FPS 延遲顯示並檢查按鍵事件
            frame_interval_ms = int(1000 / self.fps)
            key = cv2.waitKey(frame_interval_ms) & 0xFF
            
            # 按鍵處理
            if key == ord(' '):  # 空白鍵，切換計數狀態
                self.toggle_counting()
            elif key in (ord('q'), 27):  # 'q' 或 ESC 鍵退出
                # 如果用戶手動退出但尚未輸出結果，仍然生成輸出
                if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                    self.generate_outputs()
                break

            # 精準控制幀率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 / self.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 釋放資源
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """主函數：解析命令列參數並啟動追蹤器"""
    global NET_CROSSING_DIRECTION, MAX_NET_SPEEDS
    
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器')
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
    args = parser.parse_args()

    # 設置全局穿越方向
    NET_CROSSING_DIRECTION = args.direction
    # 設置收集數量
    MAX_NET_SPEEDS = args.count

    # 根據命令列參數初始化追蹤器
    if args.video:
        tracker = PingPongSpeedTracker(
            args.video, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=True
        )
    else:
        tracker = PingPongSpeedTracker(
            args.camera, 
            table_length_cm=args.table_length,
            detection_timeout=args.timeout,
            use_video_file=False, 
            target_fps=args.fps
        )
        
    # 啟動追蹤器
    tracker.run()


if __name__ == '__main__':
    main()