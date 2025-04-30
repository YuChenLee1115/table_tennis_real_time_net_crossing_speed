#!/usr/bin/env python3
import cv2
import numpy as np
import time
from collections import deque
import math
import argparse

# —— 啟用最佳化與多線程 ——  
cv2.setUseOptimized(True)
cv2.setNumThreads(10)

class PingPongSpeedTracker:
    def __init__(self, video_source=0, table_length_cm=274, detection_timeout=0.3, use_video_file=False, target_fps=60):
        # 初始化
        self.cap = cv2.VideoCapture(video_source)
        self.use_video_file = use_video_file

        if not self.use_video_file:
            # 嘗試設定 webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 1000:
                self.fps = 60
                self.manual_fps_calc = True
                self.frame_times = deque(maxlen=20)
            else:
                self.manual_fps_calc = False
        else:
            # 讀影片檔
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.manual_fps_calc = False

        # 讀取解析度
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 參數
        self.table_length_cm = table_length_cm
        self.pixels_per_cm = self.frame_width / table_length_cm
        self.roi_start_x = int(self.frame_width * 0.4)
        self.roi_end_x   = int(self.frame_width * 0.6)
        self.detection_timeout = detection_timeout
        self.last_detection_time = time.time()

        # 軌跡與速度
        self.trajectory = deque(maxlen=50)
        self.ball_speed = 0

        # FMO 前置
        self.prev_frames = deque(maxlen=5)
        self.opening_kernel = np.ones((2,2), np.uint8)
        self.closing_kernel = np.ones((7,7), np.uint8)

        # 影片檔專用
        self.frame_count = 0
        self.last_frame_time = time.time()

    def update_fps(self):
        if self.manual_fps_calc:
            now = time.time()
            self.frame_times.append(now)
            if len(self.frame_times) >= 2:
                dt = self.frame_times[-1] - self.frame_times[0]
                if dt > 0:
                    measured = (len(self.frame_times)-1)/dt
                    self.fps = 0.7*self.fps + 0.3*measured
            self.last_frame_time = now

    def preprocess_frame(self, frame):
        roi = frame[:, self.roi_start_x:self.roi_end_x]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.prev_frames.append(gray)
        return roi, gray

    def detect_fmo(self):
        if len(self.prev_frames) < 3:
            return None
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        mask = cv2.bitwise_and(diff1, diff2)
        _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
        return cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)

    def detect_ball(self, roi, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            a = cv2.contourArea(c)
            if 5 < a < 900:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cx_orig = cx + self.roi_start_x
                    self.last_detection_time = time.time()
                    ts = (self.frame_count/self.fps) if self.use_video_file else time.time()
                    self.trajectory.append((cx_orig, cy, ts))
                    return (cx, cy), c
        return None, None

    def calculate_speed(self):
        if len(self.trajectory) >= 2:
            p1, p2 = self.trajectory[-2], self.trajectory[-1]
            dp = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            dist_cm = dp / self.pixels_per_cm
            dt = p2[2] - p1[2]
            if dt > 0:
                spd = dist_cm/dt * 0.036  # km/h
                self.ball_speed = 0.7*self.ball_speed + 0.3*spd if self.ball_speed>0 else spd

    def draw_visualizations(self, frame, roi, ball_position=None, ball_contour=None):
        # Draw ROI 邊界
        cv2.line(frame, (self.roi_start_x,0), (self.roi_start_x, self.frame_height), (0,255,0),2)
        cv2.line(frame, (self.roi_end_x, 0), (self.roi_end_x, self.frame_height), (0,255,0),2)

        # 畫出整條軌跡
        if len(self.trajectory) >= 2:
            pts = np.array([(p[0], p[1]) for p in self.trajectory], np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [pts], False, (0,0,255), 2)

        # 如果有偵測到，就畫球與輪廓
        if ball_position:
            cv2.circle(roi, ball_position, 5, (0,255,255), -1)
            cv2.drawContours(roi, [ball_contour], 0, (255,0,0), 2)

        # —— 常駐顯示球速與 FPS ——
        cv2.putText(frame,
                    f"Speed: {self.ball_speed:.1f} km/h",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame,
                    f"FPS:   {self.fps:.1f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def check_timeout(self):
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera/Video end.")
                break
            self.frame_count += 1
            if not self.use_video_file:
                self.update_fps()
            roi, gray = self.preprocess_frame(frame)
            mask = self.detect_fmo()
            if mask is not None:
                ball_pos, ball_cnt = self.detect_ball(roi, mask)
                self.calculate_speed()
            else:
                ball_pos, ball_cnt = None, None
            self.draw_visualizations(frame, roi, ball_pos, ball_cnt)
            self.check_timeout()

            # 每幀開始前記錄時間
            start_time = time.time()

            cv2.imshow('Ping Pong Speed', frame)
            # 依照目標 FPS 延遲顯示
            frame_interval_ms = int(1000 / self.fps)
            key = cv2.waitKey(frame_interval_ms) & 0xFF
            if key in (ord('q'), 27):
                break

            # （可選）若希望更精準，補上剩餘的睡眠時間
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0/self.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='', help='影片檔路徑，留空用 webcam')
    parser.add_argument('--camera', type=int, default=0, help='攝影機編號')
    parser.add_argument('--fps',    type=int, default=60, help='webcam 目標 FPS')
    args = parser.parse_args()

    if args.video:
        tracker = PingPongSpeedTracker(args.video, use_video_file=True)
    else:
        tracker = PingPongSpeedTracker(args.camera, use_video_file=False, target_fps=args.fps)
    tracker.run()

if __name__ == '__main__':
    main()