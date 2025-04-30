#!/usr/bin/env python3
import cv2
import numpy as np
import time
from collections import deque
import math
import argparse

# —— 全局參數設定 ——
# 基本設定
DEFAULT_CAMERA_INDEX = 0  # 預設相機索引
DEFAULT_TARGET_FPS = 60  # 預設目標 FPS
DEFAULT_FRAME_WIDTH = 1920  # 預設影像寬度
DEFAULT_FRAME_HEIGHT = 1080  # 預設影像高度
DEFAULT_TABLE_LENGTH_CM = 274  # 乒乓球桌長度，單位 cm

# 偵測相關參數
DEFAULT_DETECTION_TIMEOUT = 0.3  # 球體偵測超時，超過此時間將重置軌跡
DEFAULT_ROI_START_RATIO = 0.4  # ROI 區域開始比例
DEFAULT_ROI_END_RATIO = 0.6  # ROI 區域結束比例
MAX_TRAJECTORY_POINTS = 50  # 最大軌跡點數

# FMO (Fast Moving Object) 相關參數
MAX_PREV_FRAMES = 5  # 保留前幾幀的最大數量
OPENING_KERNEL_SIZE = (2, 2)  # 開運算內核大小
CLOSING_KERNEL_SIZE = (7, 7)  # 閉運算內核大小
THRESHOLD_VALUE = 10  # 二值化閾值

# 球體偵測參數
MIN_BALL_AREA = 5  # 最小球體面積
MAX_BALL_AREA = 900  # 最大球體面積

# 速度計算參數
SPEED_SMOOTHING = 0.7  # 速度平滑因子
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
        # 擷取 ROI 區域
        roi = frame[:, self.roi_start_x:self.roi_end_x]
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
                    
                    # 保存軌跡點
                    self.trajectory.append((cx_orig, cy, ts))
                    
                    return (cx, cy), c
                    
        # 未偵測到合適的球體
        return None, None

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

        # 繪製球體軌跡
        if len(self.trajectory) >= 2:
            pts = np.array([(p[0], p[1]) for p in self.trajectory], np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, TRAJECTORY_COLOR, 2)

        # 如果有偵測到球，繪製球體與其輪廓
        if ball_position:
            cv2.circle(roi, ball_position, 5, BALL_COLOR, -1)
            cv2.drawContours(roi, [ball_contour], 0, CONTOUR_COLOR, 2)

        # —— 常駐顯示球速與 FPS ——
        cv2.putText(
            frame,
            f"Speed: {self.ball_speed:.1f} km/h",
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

    def check_timeout(self):
        """檢查是否超過偵測超時時間，若是則重置軌跡和速度"""
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0

    def run(self):
        """執行主循環"""
        while True:
            # 讀取影像幀
            ret, frame = self.cap.read()
            if not ret:
                print("Camera/Video end.")
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
            
            # 依照目標 FPS 延遲顯示
            frame_interval_ms = int(1000 / self.fps)
            key = cv2.waitKey(frame_interval_ms) & 0xFF
            if key in (ord('q'), 27):  # 'q' 或 ESC 鍵退出
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
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器')
    parser.add_argument('--video', type=str, default='', help='影片檔路徑，留空用 webcam')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX, help='攝影機編號')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='webcam 目標 FPS')
    parser.add_argument('--table_length', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='球桌長度 (cm)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='偵測超時時間 (秒)')
    args = parser.parse_args()

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