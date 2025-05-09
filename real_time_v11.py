#!/usr/bin/env python3
# 乒乓球速度追蹤系統 v10
# 輕量化優化版本：顯示與處理分離、OpenCV 函數優化、記錄機制改進

import cv2
import numpy as np
import time
import datetime
from collections import deque
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
import platform

# —— 全局參數設定 ——
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 120
DEFAULT_FRAME_WIDTH = 1280  # 降低解析度以提高性能
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_TABLE_LENGTH_CM = 142
DEFAULT_DETECTION_TIMEOUT = 0.05
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.8
MAX_TRAJECTORY_POINTS = 50  # 減少軌跡點數量
CENTER_LINE_WIDTH = 20
CENTER_DETECTION_COOLDOWN = 0.15
MAX_NET_SPEEDS = 27
NET_CROSSING_DIRECTION = 'left_to_right'
OUTPUT_FOLDER = 'real_time_output'
NEAR_SIDE_WIDTH_CM = 29
FAR_SIDE_WIDTH_CM = 72
MAX_PREV_FRAMES = 3  # 減少保留的幀數
OPENING_KERNEL_SIZE = (5, 5)  # 減小內核大小
CLOSING_KERNEL_SIZE = (10, 10)
THRESHOLD_VALUE = 10
MIN_BALL_AREA = 10
MAX_BALL_AREA = 7000
MIN_CIRCULARITY = 0.5
SPEED_SMOOTHING = 0.5
KMH_CONVERSION = 0.036
FPS_SMOOTHING = 0.4
MAX_FRAME_TIMES = 20
TRAJECTORY_COLOR = (0, 0, 255)
BALL_COLOR = (0, 255, 255)
CONTOUR_COLOR = (255, 0, 0)
ROI_COLOR = (0, 255, 0)
SPEED_TEXT_COLOR = (0, 0, 255)
FPS_TEXT_COLOR = (0, 255, 0)
CENTER_LINE_COLOR = (0, 255, 255)
NET_SPEED_TEXT_COLOR = (255, 0, 0)
FONT_SCALE = 1
FONT_THICKNESS = 2
MAX_QUEUE_SIZE = 10
VISUALIZATION_INTERVAL = 2
EVENT_BUFFER_SIZE = 20
PREDICTION_LOOKAHEAD = 5
DEBUG_MODE = False

# —— 啟用最佳化與多線程 ——
cv2.setUseOptimized(True)
cv2.setNumThreads(10)

class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM, 
                 detection_timeout=DEFAULT_DETECTION_TIMEOUT, use_video_file=False, target_fps=DEFAULT_TARGET_FPS,
                 debug_mode=DEBUG_MODE):
        self.debug_mode = debug_mode
        self.cap = cv2.VideoCapture(video_source)
        self.use_video_file = use_video_file
        self._setup_capture(target_fps)
        self.table_length_cm = table_length_cm
        self.detection_timeout = detection_timeout
        self.pixels_per_cm = self.frame_width / table_length_cm
        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_end_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.ball_speed = 0
        self.last_detection_time = time.perf_counter()
        self.prev_frames = deque(maxlen=MAX_PREV_FRAMES)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE)
        self.frame_count = 0
        self.last_frame_time = time.perf_counter()
        self.center_x = self.frame_width // 2
        self.center_line_start = self.center_x - CENTER_LINE_WIDTH // 2
        self.center_line_end = self.center_x + CENTER_LINE_WIDTH // 2
        self.net_speeds = []
        self.relative_times = []
        self.last_net_detection_time = 0
        self.last_net_speed = 0
        self.crossed_center = False
        self.last_ball_x = None
        self.output_generated = False
        self.is_counting = False
        self.count_session = 0
        self.timing_started = False
        self.first_ball_time = None
        self.near_side_width_cm = NEAR_SIDE_WIDTH_CM
        self.far_side_width_cm = FAR_SIDE_WIDTH_CM
        self.perspective_ratio = self.far_side_width_cm / self.near_side_width_cm
        self.roi_height = self.roi_end_y
        self.running = False
        self.event_buffer = deque(maxlen=EVENT_BUFFER_SIZE)
        self._setup_precalculated_elements()
        self.key_pressed = None

    def _setup_precalculated_elements(self):
        self.overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.overlay, (self.roi_start_x, 0), (self.roi_start_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(self.overlay, (self.roi_end_x, 0), (self.roi_end_x, self.frame_height), ROI_COLOR, 2)
        cv2.line(self.overlay, (0, self.roi_end_y), (self.frame_width, self.roi_end_y), ROI_COLOR, 2)
        cv2.line(self.overlay, (self.center_x, 0), (self.center_x, self.frame_height), CENTER_LINE_COLOR, 2)
        self.instruction_text = "Press SPACE to toggle counting, ESC or q to quit"
        self._create_perspective_lookup()

    def _create_perspective_lookup(self):
        self.perspective_lookup = {}
        for y in range(0, self.roi_end_y, 10):
            relative_y = min(1, max(0, y / self.roi_height))
            near_ratio = self.table_length_cm / self.frame_width * (self.near_side_width_cm / self.table_length_cm)
            far_ratio = self.table_length_cm / self.frame_width * (self.far_side_width_cm / self.table_length_cm)
            ratio = near_ratio * relative_y + far_ratio * (1 - relative_y)
            self.perspective_lookup[y] = ratio

    def _setup_capture(self, target_fps):
        if not self.use_video_file:
            if platform.system() == 'Darwin':  # MacOS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 1000:
                    self.fps = target_fps
                    self.manual_fps_calc = True
                    self.frame_times = deque(maxlen=MAX_FRAME_TIMES)
                else:
                    self.manual_fps_calc = False
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 1000:
                    self.fps = target_fps
                    self.manual_fps_calc = True
                    self.frame_times = deque(maxlen=MAX_FRAME_TIMES)
                else:
                    self.manual_fps_calc = False
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.manual_fps_calc = False
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_fps(self):
        if not self.manual_fps_calc:
            return
        now = time.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) >= 2:
            dt = self.frame_times[-1] - self.frame_times[0]
            if dt > 0:
                measured = (len(self.frame_times) - 1) / dt
                self.fps = (1 - FPS_SMOOTHING) * self.fps + FPS_SMOOTHING * measured
        self.last_frame_time = now

    def preprocess_frame(self, frame):
        roi = frame[:self.roi_end_y, self.roi_start_x:self.roi_end_x]
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
        _, thresh = cv2.threshold(mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)
        return closing

    def detect_ball(self, roi, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        potential_balls = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_BALL_AREA < area < MAX_BALL_AREA:
                cx, cy = centroids[i]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                circularity = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                potential_balls.append({
                    'position': (int(cx), int(cy)),
                    'original_x': int(cx) + self.roi_start_x,
                    'area': area,
                    'circularity': circularity,
                    'label': i
                })
        if potential_balls:
            best_ball = self._select_best_ball(potential_balls)
            if best_ball:
                cx, cy = best_ball['position']
                cx_orig = best_ball['original_x']
                label_mask = (labels == best_ball['label']).astype(np.uint8) * 255
                contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0] if contours else None
                self.last_detection_time = time.perf_counter()
                ts = (self.frame_count / self.fps) if self.use_video_file else time.perf_counter()
                if self.is_counting:
                    self.check_center_crossing(cx_orig, ts)
                self.trajectory.append((cx_orig, cy, ts))
                return (cx, cy), contour
        return None, None

    def _select_best_ball(self, potential_balls):
        if not potential_balls:
            return None
        if len(self.trajectory) == 0:
            circular_balls = [ball for ball in potential_balls if ball.get('circularity', 0) > MIN_CIRCULARITY]
            if circular_balls:
                return max(circular_balls, key=lambda ball: ball.get('circularity', 0))
            return max(potential_balls, key=lambda ball: ball.get('area', 0))
        last_x, last_y, _ = self.trajectory[-1]
        for ball in potential_balls:
            x, y = ball['position']
            orig_x = ball['original_x']
            distance = math.hypot(orig_x - last_x, y - last_y)
            ball['distance'] = distance if distance < self.frame_width * 0.2 else float('inf')
            if len(self.trajectory) >= 3:
                x1, y1, _ = self.trajectory[-2]
                x2, y2, _ = self.trajectory[-1]
                past_dx = x2 - x1
                past_dy = y2 - y1
                current_dx = orig_x - x2
                current_dy = y - y2
                dot_product = past_dx * current_dx + past_dy * current_dy
                past_mag = math.sqrt(past_dx**2 + past_dy**2)
                current_mag = math.sqrt(current_dx**2 + current_dy**2)
                if past_mag > 0 and current_mag > 0:
                    consistency = dot_product / (past_mag * current_mag)
                    ball['consistency'] = max(0, consistency)
                else:
                    ball['consistency'] = 0
            else:
                ball['consistency'] = 0
        for ball in potential_balls:
            distance_score = 1.0 / (1.0 + ball.get('distance', float('inf')))
            consistency_score = ball.get('consistency', 0)
            circularity_score = ball.get('circularity', 0)
            ball['score'] = distance_score * 0.5 + consistency_score * 0.3 + circularity_score * 0.2
        potential_balls.sort(key=lambda ball: ball.get('score', 0), reverse=True)
        return potential_balls[0] if potential_balls else None

    def toggle_counting(self):
        if not self.is_counting:
            self.is_counting = True
            self.net_speeds = []
            self.relative_times = []
            self.timing_started = False
            self.first_ball_time = None
            self.event_buffer.clear()
            self.output_generated = False
            self.count_session += 1
            print(f"開始計數 (會話 #{self.count_session}) - 目標收集 {MAX_NET_SPEEDS} 個速度值")
        else:
            self.is_counting = False
            if len(self.net_speeds) > 0:
                print(f"中止計數 - 已收集 {len(self.net_speeds)} 個速度值")
                self.generate_outputs()
            else:
                print("中止計數 - 未收集到任何速度資料")

    def check_center_crossing(self, ball_x, timestamp):
        if self.last_ball_x is None:
            self.last_ball_x = ball_x
            return
        time_since_last_detection = timestamp - self.last_net_detection_time
        if time_since_last_detection < CENTER_DETECTION_COOLDOWN:
            self.last_ball_x = ball_x
            return
        self._record_potential_crossing(ball_x, timestamp)
        self.last_ball_x = ball_x

    def _record_potential_crossing(self, ball_x, timestamp):
        direction = ball_x - self.last_ball_x
        crossed_left_to_right = (self.last_ball_x < self.center_line_end and ball_x >= self.center_line_end)
        crossed_right_to_left = (self.last_ball_x > self.center_line_start and ball_x <= self.center_line_start)
        record_crossing = False
        if NET_CROSSING_DIRECTION == 'left_to_right' and crossed_left_to_right:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'right_to_left' and crossed_right_to_left:
            record_crossing = True
        elif NET_CROSSING_DIRECTION == 'both' and (crossed_left_to_right or crossed_right_to_left):
            record_crossing = True
        if record_crossing and self.ball_speed > 0:
            event = {'ball_x': ball_x, 'timestamp': timestamp, 'speed': self.ball_speed, 'predicted': False, 'processed': False}
            self.event_buffer.append(event)
            self._process_crossing_events()
            return
        will_cross = False
        time_to_cross = 0
        if len(self.trajectory) >= 2 and direction != 0:
            last_x, _, last_t = self.trajectory[-2]
            curr_x, _, curr_t = self.trajectory[-1]
            if curr_t > last_t:
                x_velocity = (curr_x - last_x) / (curr_t - last_t)
                if (x_velocity > 0 and ball_x < self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps >= self.center_x) or \
                   (x_velocity < 0 and ball_x > self.center_x and ball_x + x_velocity * PREDICTION_LOOKAHEAD / self.fps <= self.center_x):
                    will_cross = True
                    time_to_cross = abs((self.center_x - ball_x) / x_velocity) if x_velocity != 0 else 0
        if will_cross and self.ball_speed > 0 and time_to_cross < 0.5:
            predicted_time = timestamp + time_to_cross
            prediction_valid = False
            if (NET_CROSSING_DIRECTION == 'left_to_right' and direction > 0) or \
               (NET_CROSSING_DIRECTION == 'right_to_left' and direction < 0) or \
               NET_CROSSING_DIRECTION == 'both':
                prediction_valid = True
            if prediction_valid:
                for event in self.event_buffer:
                    if event['predicted'] and abs(event['timestamp'] - predicted_time) < 0.1:
                        event['timestamp'] = predicted_time
                        event['speed'] = self.ball_speed
                        return
                event = {'ball_x': ball_x, 'timestamp': predicted_time, 'speed': self.ball_speed, 'predicted': True, 'processed': False}
                self.event_buffer.append(event)

    def _process_crossing_events(self):
        current_time = time.perf_counter()
        events_to_process = []
        for event in self.event_buffer:
            if not event['processed']:
                if not event['predicted']:
                    events_to_process.append(event)
                elif current_time - event['timestamp'] > 0.2:
                    events_to_process.append(event)
        events_to_process.sort(key=lambda e: e['timestamp'])
        for event in events_to_process:
            event['processed'] = True
            if not self.timing_started:
                self.timing_started = True
                self.first_ball_time = event['timestamp']
                relative_time = 0.0
            else:
                relative_time = round(event['timestamp'] - self.first_ball_time, 2)
            self.crossed_center = True
            self.last_net_speed = event['speed']
            self.net_speeds.append(event['speed'])
            self.relative_times.append(relative_time)
            self.last_net_detection_time = event['timestamp']
            status = "預測" if event['predicted'] else "實際"
            print(f"記錄速度 #{len(self.net_speeds)}: {event['speed']:.1f} km/h, 時間: {relative_time}秒 ({status}穿越)")
            if len(self.net_speeds) >= MAX_NET_SPEEDS and not self.output_generated:
                print(f"已達到目標次數 ({MAX_NET_SPEEDS})，生成輸出並停止計數")
                self.generate_outputs()
                self.is_counting = False
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
            self.perspective_lookup[y1_rounded] = ratio1
            
        if y2_rounded in self.perspective_lookup:
            ratio2 = self.perspective_lookup[y2_rounded]
        else:
            ratio2 = self._get_pixel_to_cm_ratio(y2)
            self.perspective_lookup[y2_rounded] = ratio2
        
        # 使用兩點的平均比例轉換像素距離為實際距離
        avg_ratio = (ratio1 + ratio2) / 2
        real_distance_cm = pixel_distance * avg_ratio
        
        if self.debug_mode:
            print(f"透視校正: 像素距離={pixel_distance:.1f}, 轉換比例1={ratio1:.4f}(y={y1}), 比例2={ratio2:.4f}(y={y2}), 實際距離={real_distance_cm:.2f}cm")
        
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
        
        if self.debug_mode:
            print(f"比例計算: y={y}, 相對位置={relative_y:.2f}, 近端比例={near_ratio:.4f}, 遠端比例={far_ratio:.4f}, 結果={pixel_to_cm_ratio:.4f}")
        
        return pixel_to_cm_ratio

    def generate_outputs(self):
        if len(self.net_speeds) == 0:
            print("沒有可輸出的速度數據")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"生成輸出結果: {len(self.net_speeds)} 個速度值")
        output_dir = f"{OUTPUT_FOLDER}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        chart_filename = self._generate_chart(output_dir, timestamp)
        txt_filename = self._generate_txt(output_dir, timestamp)
        csv_filename = self._generate_csv(output_dir, timestamp)
        print(f"輸出已保存到 {output_dir} 資料夾")
        print(f"- 折線圖: {chart_filename}")
        print(f"- 文字檔: {txt_filename}")
        print(f"- CSV檔: {csv_filename}")

    def _generate_chart(self, output_dir, timestamp):
        plot_times = self.relative_times
        plot_speeds = self.net_speeds
        avg_speed = sum(plot_speeds) / len(plot_speeds) if plot_speeds else 0
        max_speed = max(plot_speeds) if plot_speeds else 0
        min_speed = min(plot_speeds) if plot_speeds else 0
        plt.figure(figsize=(12, 7))
        plt.plot(plot_times, plot_speeds, 'o-', linewidth=2, markersize=8, color='#3498db', label='Speed(km/h)')
        plt.axhline(y=avg_speed, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Average: {avg_speed:.1f} km/h')
        for i, (t, s) in enumerate(zip(plot_times, plot_speeds)):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none", alpha=0.7))
        plt.title(f'Table Tennis Net Speed Record', fontsize=16, pad=20)
        plt.xlabel('Time(s)', fontsize=12, labelpad=10)
        plt.ylabel('Speed(km/h)', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        if plot_times:
            x_min = min(plot_times)
            x_max = max(plot_times)
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.5
            plt.xlim(x_min - x_margin, x_max + x_margin)
            y_min = min(min_speed * 0.9, avg_speed * 0.9)
            y_max = max(max_speed * 1.1, avg_speed * 1.1)
            plt.ylim(y_min, y_max)
        plt.figtext(0.02, 0.02, f"Total: {len(plot_speeds)} | Max: {max_speed:.1f} km/h | Min: {min_speed:.1f} km/h", fontsize=9, color='#555555')
        plt.tight_layout()
        chart_filename = f'{output_dir}/speed_chart_{timestamp}.png'
        plt.savefig(chart_filename, dpi=150)
        plt.close()
        return chart_filename

    def _generate_txt(self, output_dir, timestamp):
        avg_speed = sum(self.net_speeds) / len(self.net_speeds)
        max_speed = max(self.net_speeds)
        min_speed = min(self.net_speeds)
        txt_filename = f'{output_dir}/speed_data_{timestamp}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Table Tennis Net Speed Record (km/h) - Session {self.count_session}\n")
            f.write("----------------------------------\n")
            for i, (speed, rel_time) in enumerate(zip(self.net_speeds, self.relative_times), 1):
                f.write(f"{rel_time}s: {speed:.1f} km/h\n")
            f.write("\n----------------------------------\n")
            f.write(f"Average: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum: {max_speed:.1f} km/h\n")
            f.write(f"Minimum: {min_speed:.1f} km/h\n")
        return txt_filename

    def _generate_csv(self, output_dir, timestamp):
        avg_speed = sum(self.net_speeds) / len(self.net_speeds)
        max_speed = max(self.net_speeds)
        min_speed = min(self.net_speeds)
        csv_filename = f'{output_dir}/speed_data_{timestamp}.csv'
        with open(csv_filename, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Time(s)", "Speed(km/h)"])
            for i, (speed, rel_time) in enumerate(zip(self.net_speeds, self.relative_times), 1):
                csv_writer.writerow([f"{rel_time:.2f}", f"{speed:.1f}"])
            csv_writer.writerow([])
            csv_writer.writerow(["Statistics", ""])
            csv_writer.writerow(["Average", f"{avg_speed:.1f}"])
            csv_writer.writerow(["Maximum", f"{max_speed:.1f}"])
            csv_writer.writerow(["Minimum", f"{min_speed:.1f}"])
        return csv_filename

    def draw_visualizations(self, frame, frame_data):
        vis_layer = frame.copy()
        if frame_data['frame_count'] % VISUALIZATION_INTERVAL == 0:
            vis_layer = cv2.addWeighted(vis_layer, 1, self.overlay, 1, 0)
            if frame_data['trajectory_points'] and len(frame_data['trajectory_points']) >= 2:
                trajectory_points = np.array(frame_data['trajectory_points'], dtype=np.int32)
                cv2.polylines(vis_layer, [trajectory_points], False, TRAJECTORY_COLOR, 2)
        if frame_data['ball_position'] and frame_data['ball_contour'] is not None:
            cx, cy = frame_data['ball_position']
            global_cx = cx + self.roi_start_x
            cv2.circle(vis_layer, (global_cx, cy), 8, BALL_COLOR, -1)
        cv2.putText(vis_layer, f"Current Speed: {frame_data['ball_speed']:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, SPEED_TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(vis_layer, f"FPS: {frame_data['fps']:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FPS_TEXT_COLOR, FONT_THICKNESS)
        count_status = "ON" if frame_data['is_counting'] else "OFF"
        count_color = (0, 255, 0) if frame_data['is_counting'] else (0, 0, 255)
        cv2.putText(vis_layer, f"Counting: {count_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, count_color, FONT_THICKNESS)
        if frame_data['last_net_speed'] > 0:
            cv2.putText(vis_layer, f"Net Speed: {frame_data['last_net_speed']:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NET_SPEED_TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(vis_layer, f"Recorded: {len(frame_data['net_speeds'])}/{MAX_NET_SPEEDS}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NET_SPEED_TEXT_COLOR, FONT_THICKNESS)
        if len(frame_data['relative_times']) > 0:
            cv2.putText(vis_layer, f"Last Time: {frame_data['relative_times'][-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NET_SPEED_TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(vis_layer, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if self.debug_mode and 'display_text' in frame_data and frame_data['display_text']:
            cv2.putText(vis_layer, frame_data['display_text'], (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        return vis_layer

    def check_timeout(self):
        if time.perf_counter() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0
            self.crossed_center = False

    def process_frame(self, frame):
        self.frame_count += 1
        if not self.use_video_file:
            self.update_fps()
        roi, gray = self.preprocess_frame(frame)
        mask = self.detect_fmo()
        ball_pos = None
        ball_cnt = None
        if mask is not None:
            ball_pos, ball_cnt = self.detect_ball(roi, mask)
            self.calculate_speed()
        self.check_timeout()
        frame_data = {
            'frame': frame,
            'roi': roi,
            'ball_position': ball_pos,
            'ball_contour': ball_cnt,
            'ball_speed': self.ball_speed,
            'fps': self.fps,
            'is_counting': self.is_counting,
            'net_speeds': self.net_speeds.copy(),
            'last_net_speed': self.last_net_speed,
            'relative_times': self.relative_times.copy(),
            'frame_count': self.frame_count,
            'trajectory_points': [(p[0], p[1]) for p in self.trajectory]
        }
        return frame_data

    def run(self):
        print("=== 乒乓球速度追蹤器 v10 ===")
        print("按下空白鍵開始/停止計數")
        print("按下 'd' 鍵切換調試模式")
        print("按下 'q' 或 ESC 鍵退出程序")
        print(f"使用透視校正: 近端寬度 {self.near_side_width_cm}cm, 遠端寬度 {self.far_side_width_cm}cm")
        self.running = True
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera/Video end.")
                    if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                        self.generate_outputs()
                    break
                future = executor.submit(self.process_frame, frame.copy())
                try:
                    timeout = 2.0 / max(1, self.fps)
                    frame_data = future.result(timeout=timeout)
                    display_frame = self.draw_visualizations(frame, frame_data)
                    cv2.imshow('Ping Pong Speed', display_frame)
                except concurrent.futures.TimeoutError:
                    cv2.putText(frame, "Processing timeout - frame skipped", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Ping Pong Speed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"調試模式: {'開啟' if self.debug_mode else '關閉'}")
                elif key in (ord('q'), 27):
                    if self.is_counting and len(self.net_speeds) > 0 and not self.output_generated:
                        self.generate_outputs()
                    break
                if self.is_counting:
                    self._process_crossing_events()
        except KeyboardInterrupt:
            print("程序被用戶中斷")
        finally:
            self.running = False
            executor.shutdown(wait=False)
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    global NET_CROSSING_DIRECTION, MAX_NET_SPEEDS, VISUALIZATION_INTERVAL
    parser = argparse.ArgumentParser(description='乒乓球速度追蹤器')
    parser.add_argument('--video', type=str, default='', help='影片檔路徑，留空用 webcam')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX, help='攝影機編號')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help='webcam 目標 FPS')
    parser.add_argument('--table_length', type=int, default=DEFAULT_TABLE_LENGTH_CM, help='球桌長度 (cm)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_DETECTION_TIMEOUT, help='偵測超時時間 (秒)')
    parser.add_argument('--direction', type=str, default=NET_CROSSING_DIRECTION, choices=['left_to_right', 'right_to_left', 'both'], help='記錄球經過網中心的方向')
    parser.add_argument('--count', type=int, default=MAX_NET_SPEEDS, help='需要收集的速度數據數量')
    parser.add_argument('--debug', action='store_true', help='啟用調試模式')
    parser.add_argument('--near_width', type=int, default=NEAR_SIDE_WIDTH_CM, help='ROI 較近側的實際寬度 (cm)')
    parser.add_argument('--far_width', type=int, default=FAR_SIDE_WIDTH_CM, help='ROI 較遠側的實際寬度 (cm)')
    parser.add_argument('--interval', type=int, default=VISUALIZATION_INTERVAL, help='視覺化間隔，指定每隔多少幀繪製一次完整視覺元素')
    args = parser.parse_args()
    NET_CROSSING_DIRECTION = args.direction
    MAX_NET_SPEEDS = args.count
    VISUALIZATION_INTERVAL = args.interval
    if args.video:
        tracker = PingPongSpeedTracker(args.video, table_length_cm=args.table_length, detection_timeout=args.timeout, use_video_file=True, debug_mode=args.debug)
    else:
        tracker = PingPongSpeedTracker(args.camera, table_length_cm=args.table_length, detection_timeout=args.timeout, use_video_file=False, target_fps=args.fps, debug_mode=args.debug)
    tracker.near_side_width_cm = args.near_width
    tracker.far_side_width_cm = args.far_width
    tracker.perspective_ratio = tracker.far_side_width_cm / tracker.near_side_width_cm
    tracker.run()

if __name__ == '__main__':
    main()