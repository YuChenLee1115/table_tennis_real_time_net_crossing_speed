# %%
# 匯入所需套件
import cv2
import numpy as np
import time
import os
import datetime
from datetime import datetime
from collections import deque
import math
import matplotlib
matplotlib.use("Agg")  # 必須在匯入 pyplot 前設定
import matplotlib.pyplot as plt
import csv
import threading
import queue
import concurrent.futures

# —— 基本設定 ——
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1920 
DEFAULT_FRAME_HEIGHT = 1080
DEFAULT_TARGET_FPS = 120
DEFAULT_VIDEO_CODEC = 'avc1'  # 改為 avc1 以支援硬體加速
DEFAULT_TABLE_LENGTH_CM = 142

# —— 偵測參數 ——
DEFAULT_DETECTION_TIMEOUT = 0.1
DEFAULT_ROI_START_RATIO = 0.4
DEFAULT_ROI_END_RATIO = 0.6
DEFAULT_ROI_BOTTOM_RATIO = 0.8
MAX_TRAJECTORY_POINTS = 120

# —— 中心線偵測 ——
CENTER_LINE_WIDTH_PIXELS = 500
CENTER_DETECTION_COOLDOWN_S = 0.003
NET_CROSSING_DIRECTION_DEFAULT = 'left_to_right'  # 'left_to_right', 'right_to_left', 'both'

# —— 透視校正 ——
NEAR_SIDE_WIDTH_CM_DEFAULT = 29
FAR_SIDE_WIDTH_CM_DEFAULT = 72

# FMO (Fast Moving Object) Parameters
MAX_PREV_FRAMES_FMO = 5  # 減少以節省內存
OPENING_KERNEL_SIZE_FMO = (15, 15)  # 減少核大小
CLOSING_KERNEL_SIZE_FMO = (20, 20)  # 減少核大小
THRESHOLD_VALUE_FMO = 20

# Ball Detection Parameters
MIN_BALL_AREA_PX = 40
MAX_BALL_AREA_PX = 6000
MIN_BALL_CIRCULARITY = 0.4

# —— 速度計算 ——
SPEED_SMOOTHING_FACTOR = 0.3
KMH_CONVERSION_FACTOR = 0.036

# —— FPS 計算 ——
FPS_SMOOTHING_FACTOR = 0.4
MAX_FRAME_TIMES_FPS_CALC = 20

# —— 視覺化參數 ——
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
VISUALIZATION_DRAW_INTERVAL = 4  # 增加到每4幀畫一次詳細視覺化

# —— 執行緒與佇列參數 ——
FRAME_QUEUE_SIZE = 20  # 增加佇列大小
EVENT_BUFFER_SIZE_CENTER_CROSS = 70
PREDICTION_LOOKAHEAD_FRAMES = 15

# —— OpenCV 最佳化 ——
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 10)
except AttributeError:
    cv2.setNumThreads(10)

# %%
class FrameData:
    """傳遞幀相關資訊的資料結構"""
    def __init__(self, frame=None, roi_sub_frame=None, ball_position_in_roi=None,
                 ball_contour_in_roi=None, current_ball_speed_kmh=0,
                 display_fps=0, is_counting_active=False, collected_net_speeds=None,
                 last_recorded_net_speed_kmh=0, collected_relative_times=None,
                 debug_display_text=None, frame_counter=0):
        self.frame = frame
        self.roi_sub_frame = roi_sub_frame  # ROI 部分的幀
        self.ball_position_in_roi = ball_position_in_roi  # 球在 ROI 中的位置 (x,y)
        self.ball_contour_in_roi = ball_contour_in_roi  # 球在 ROI 中的輪廓點
        self.current_ball_speed_kmh = current_ball_speed_kmh
        self.display_fps = display_fps
        self.is_counting_active = is_counting_active
        self.collected_net_speeds = collected_net_speeds if collected_net_speeds is not None else []
        self.last_recorded_net_speed_kmh = last_recorded_net_speed_kmh
        self.collected_relative_times = collected_relative_times if collected_relative_times is not None else []
        self.debug_display_text = debug_display_text
        self.frame_counter = frame_counter
        self.trajectory_points_global = []  # 全域座標中的軌跡點 (x,y)

class EventRecord:
    """潛在中心線穿越事件的記錄"""
    def __init__(self, ball_x_global, timestamp, speed_kmh, predicted=False):
        self.ball_x_global = ball_x_global
        self.timestamp = timestamp
        self.speed_kmh = speed_kmh
        self.predicted = predicted
        self.processed = False

class FrameReader:
    """在單獨執行緒中從相機或影片檔讀取幀"""
    def __init__(self, video_source, target_fps, use_video_file, frame_width, frame_height):
        self.video_source = video_source
        self.target_fps = target_fps
        self.use_video_file = use_video_file
        
        # 在 macOS 上明確使用 AVFoundation 後端
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
        self._configure_capture(frame_width, frame_height)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
             self.actual_fps = self.target_fps  # 如果網路攝影機 FPS 不可靠，則使用目標 FPS

    def _configure_capture(self, frame_width, frame_height):
        if not self.use_video_file:
            # 設定 MJPG 格式以提高幀率
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 檢查設定是否成功
            actual_codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = ''.join([chr((actual_codec >> 8 * i) & 0xFF) for i in range(4)])
            print(f"Camera codec set to: {codec_str}")
            print(f"Requested FPS: {self.target_fps}, Camera reports: {self.cap.get(cv2.CAP_PROP_FPS)}")
            
        if not self.cap.isOpened():
            raise IOError(f"無法開啟影片來源: {self.video_source}")

    def _read_frames(self):
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False  # 影片結束或相機錯誤
                    self.frame_queue.put((False, None))  # 傳送結束信號
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(1.0 / (self.target_fps * 4))  # 佇列已滿時避免忙碌等待，降低CPU使用率

    def start(self):
        self.running = True
        self.thread.start()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)  # 等待最多 1 秒取得幀
        except queue.Empty:
            return False, None  # 逾時

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)  # 等待執行緒結束
        if self.cap.isOpened():
            self.cap.release()

    def get_properties(self):
        return self.actual_fps, self.frame_width, self.frame_height

# %%
class PingPongSpeedTracker:
    def __init__(self, video_source=DEFAULT_CAMERA_INDEX, table_length_cm=DEFAULT_TABLE_LENGTH_CM,
                 detection_timeout_s=DEFAULT_DETECTION_TIMEOUT, use_video_file=False,
                 target_fps=DEFAULT_TARGET_FPS, frame_width=DEFAULT_FRAME_WIDTH,
                 frame_height=DEFAULT_FRAME_HEIGHT, debug_mode=False,
                 net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
                 near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
                 far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT,
                 output_folder=None,  # 新增輸出資料夾參數
                 output_basename=None):  # 新增輸出檔案基本名稱參數
        self.debug_mode = debug_mode
        self.use_video_file = use_video_file
        self.target_fps = target_fps
        self.output_folder = output_folder
        self.output_basename = output_basename

        self.reader = FrameReader(video_source, target_fps, use_video_file, frame_width, frame_height)
        self.actual_fps, self.frame_width, self.frame_height = self.reader.get_properties()
        self.display_fps = self.actual_fps  # 初始顯示 FPS

        self.table_length_cm = table_length_cm
        self.detection_timeout_s = detection_timeout_s
        self.pixels_per_cm_nominal = self.frame_width / self.table_length_cm  # 名義上的，在透視失敗時使用

        self.roi_start_x = int(self.frame_width * DEFAULT_ROI_START_RATIO)
        self.roi_end_x = int(self.frame_width * DEFAULT_ROI_END_RATIO)
        self.roi_top_y = 0  # ROI 從幀的頂部開始
        self.roi_bottom_y = int(self.frame_height * DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_height_px = self.roi_bottom_y - self.roi_top_y

        self.trajectory = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self.current_ball_speed_kmh = 0
        self.last_detection_timestamp = time.time()

        self.prev_frames_gray_roi = deque(maxlen=MAX_PREV_FRAMES_FMO)
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPENING_KERNEL_SIZE_FMO)
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSING_KERNEL_SIZE_FMO)
        self.last_motion_mask = None  # 保存上一幀的運動遮罩

        self.frame_counter = 0
        self.last_frame_timestamp_for_fps = time.time()
        self.frame_timestamps_for_fps = deque(maxlen=MAX_FRAME_TIMES_FPS_CALC)

        self.center_x_global = self.frame_width // 2
        self.center_line_start_x = self.center_x_global - CENTER_LINE_WIDTH_PIXELS // 2
        self.center_line_end_x = self.center_x_global + CENTER_LINE_WIDTH_PIXELS // 2
        
        self.net_crossing_direction = net_crossing_direction
        self.max_net_speeds_to_collect = float('inf')  # 不限制記錄數量
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_net_crossing_detection_time = 0
        self.last_recorded_net_speed_kmh = 0
        self.last_ball_x_global = None
        self.output_generated_for_session = False
        
        self.is_counting_active = False  # 預設不啟用計數，由用戶控制
        self.count_session_id = 1
        self.timing_started_for_session = False
        self.first_ball_crossing_timestamp = None
        
        self.near_side_width_cm = near_width_cm
        self.far_side_width_cm = far_width_cm
        
        self.event_buffer_center_cross = deque(maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS)
        
        self.running = False
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 性能監控
        self.performance_history = deque(maxlen=20)  # 保存最近20幀的處理時間

        self._precalculate_overlay()
        self._create_perspective_lookup_table()

    def _precalculate_overlay(self):
        self.static_overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_top_y), (self.roi_start_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_end_x, self.roi_top_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.roi_start_x, self.roi_bottom_y), (self.roi_end_x, self.roi_bottom_y), ROI_COLOR_BGR, 2)
        cv2.line(self.static_overlay, (self.center_x_global, 0), (self.center_x_global, self.frame_height), CENTER_LINE_COLOR_BGR, 2)
        self.instruction_text = "SPACE: Start/Stop Recording | D: Debug | Q/ESC: Quit"

    def _create_perspective_lookup_table(self):
        self.perspective_lookup_px_to_cm = {}
        for y_in_roi_rounded in range(0, self.roi_height_px + 1, 10):  # 每 10px 一步
            self.perspective_lookup_px_to_cm[y_in_roi_rounded] = self._get_pixel_to_cm_ratio(y_in_roi_rounded + self.roi_top_y)

    def _get_pixel_to_cm_ratio(self, y_global):
        y_eff = min(y_global, self.roi_bottom_y) 
        
        if self.roi_bottom_y == 0:
            relative_y = 0.5
        else:
            relative_y = np.clip(y_eff / self.roi_bottom_y, 0.0, 1.0)
        
        current_width_cm = self.far_side_width_cm * (1 - relative_y) + self.near_side_width_cm * relative_y
        
        roi_width_px = self.roi_end_x - self.roi_start_x
        if current_width_cm > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px  # 每像素釐米數
        else:
            pixel_to_cm_ratio = self.table_length_cm / self.frame_width  # 備用

        return pixel_to_cm_ratio

    def _update_display_fps(self):
        if self.use_video_file:  # 影片檔的 FPS 是固定的
            self.display_fps = self.actual_fps
            return

        now = time.monotonic()  # 更適合間隔計時
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
        # 每2幀處理一次以降低CPU負擔
        if self.frame_counter % 2 != 0 and self.last_motion_mask is not None:
            return self.last_motion_mask
        
        if len(self.prev_frames_gray_roi) < 3:
            return None
        
        f1, f2, f3 = self.prev_frames_gray_roi[-3], self.prev_frames_gray_roi[-2], self.prev_frames_gray_roi[-1]
        
        # 降低處理解析度以提高速度
        scale_factor = 0.7  # 降低到70%解析度
        if scale_factor < 1.0:
            h, w = f1.shape
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            f1_small = cv2.resize(f1, (new_w, new_h))
            f2_small = cv2.resize(f2, (new_w, new_h))
            f3_small = cv2.resize(f3, (new_w, new_h))
            
            diff1 = cv2.absdiff(f1_small, f2_small)
            diff2 = cv2.absdiff(f2_small, f3_small)
            motion_mask_small = cv2.bitwise_and(diff1, diff2)
            
            # 簡化閾值處理
            _, thresh_mask_small = cv2.threshold(motion_mask_small, THRESHOLD_VALUE_FMO, 255, cv2.THRESH_BINARY)
            
            # 簡化形態學操作
            opened_mask_small = cv2.morphologyEx(thresh_mask_small, cv2.MORPH_OPEN, self.opening_kernel)
            closed_mask_small = cv2.morphologyEx(opened_mask_small, cv2.MORPH_CLOSE, self.closing_kernel)
            
            # 處理完再放大回原始尺寸
            motion_mask = cv2.resize(closed_mask_small, (w, h))
        else:
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
            
            motion_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        
        # 保存上次的處理結果供下一幀使用
        self.last_motion_mask = motion_mask
        return motion_mask

    def _detect_ball_in_roi(self, motion_mask_roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask_roi, connectivity=8)
        
        potential_balls = []
        for i in range(1, num_labels):  # 跳過背景標籤 0
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
        
        if self.is_counting_active:
            self.check_center_crossing(cx_global, current_timestamp)
        
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
            print(f"Recording started - Session #{self.count_session_id}")
            # 只有在開始新會話時才重置時間參考點
            if not self.timing_started_for_session:
                self.timing_started_for_session = False
                self.first_ball_crossing_timestamp = None
            self.event_buffer_center_cross.clear()
        else:
            print(f"Recording paused - Session #{self.count_session_id}")
            # 暫停記錄不會生成輸出，輸出只在分析完全結束時生成

    def check_center_crossing(self, ball_x_global, current_timestamp):
        if self.last_ball_x_global is None:
            self.last_ball_x_global = ball_x_global
            return

        time_since_last_net_cross = current_timestamp - self.last_net_crossing_detection_time
        if time_since_last_net_cross < CENTER_DETECTION_COOLDOWN_S:
            self.last_ball_x_global = ball_x_global
            return

        self._record_potential_crossing(ball_x_global, current_timestamp)
        self.last_ball_x_global = ball_x_global

    def _record_potential_crossing(self, ball_x_global, current_timestamp):
        crossed_l_to_r = (self.last_ball_x_global < self.center_line_end_x and ball_x_global >= self.center_line_end_x)
        crossed_r_to_l = (self.last_ball_x_global > self.center_line_start_x and ball_x_global <= self.center_line_start_x)
        
        actual_crossing_detected = False
        if self.net_crossing_direction == 'left_to_right' and crossed_l_to_r: actual_crossing_detected = True
        elif self.net_crossing_direction == 'right_to_left' and crossed_r_to_l: actual_crossing_detected = True
        elif self.net_crossing_direction == 'both' and (crossed_l_to_r or crossed_r_to_l): actual_crossing_detected = True

        if actual_crossing_detected and self.current_ball_speed_kmh > 0:
            event = EventRecord(ball_x_global, current_timestamp, self.current_ball_speed_kmh, predicted=False)
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
                        event = EventRecord(predicted_x_future, predicted_timestamp_future, self.current_ball_speed_kmh, predicted=True)
                        self.event_buffer_center_cross.append(event)

    def _process_crossing_events(self):
        if not self.is_counting_active:
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
            
            status_msg = "Pred" if event.predicted else "Actual"
            print(f"Net Speed #{len(self.collected_net_speeds)}: {event.speed_kmh:.1f} km/h @ {relative_time:.2f}s ({status_msg})")

        self.event_buffer_center_cross = deque(
            [e for e in self.event_buffer_center_cross if not e.processed],
            maxlen=EVENT_BUFFER_SIZE_CENTER_CROSS
        )

    def _calculate_ball_speed(self):
        if len(self.trajectory) < 2:
            self.current_ball_speed_kmh = 0
            return

        p1_glob, p2_glob = self.trajectory[-2], self.trajectory[-1]
        x1_glob, y1_glob, t1 = p1_glob
        x2_glob, y2_glob, t2 = p2_glob

        dist_cm = self._calculate_real_distance_cm_global(x1_glob, y1_glob, x2_glob, y2_glob)
        
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
                print(f"Speed: {dist_cm:.2f}cm in {delta_t:.4f}s -> Raw {speed_kmh:.1f}km/h, Smooth {self.current_ball_speed_kmh:.1f}km/h")
        else:
            self.current_ball_speed_kmh *= (1 - SPEED_SMOOTHING_FACTOR)

    def _calculate_real_distance_cm_global(self, x1_g, y1_g, x2_g, y2_g):
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

    def _generate_outputs(self):
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return None
        
        if self.output_folder is None or self.output_basename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = f"speed_data_{timestamp_str}"
            os.makedirs(output_dir_path, exist_ok=True)
        else:
            output_dir_path = self.output_folder
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 設定輸出基本名稱
        if self.output_basename:
            base_filename = f"{self.output_basename}_{timestamp_str}"
        else:
            base_filename = f"speed_data_{timestamp_str}"

        avg_speed = sum(self.collected_net_speeds) / len(self.collected_net_speeds)
        max_speed = max(self.collected_net_speeds)
        min_speed = min(self.collected_net_speeds)

        # 生成圖表
        chart_filename = f'{output_dir_path}/{base_filename}.png'
        plt.figure(figsize=(12, 7))
        plt.plot(self.collected_relative_times, self.collected_net_speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', label=f'Avg: {avg_speed:.1f} km/h')
        for t, s in zip(self.collected_relative_times, self.collected_net_speeds):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.title(f'Net Crossing Speeds - {timestamp_str}', fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        if self.collected_relative_times:
            x_margin = (max(self.collected_relative_times) - min(self.collected_relative_times)) * 0.05 if max(self.collected_relative_times) > min(self.collected_relative_times) else 0.5
            plt.xlim(min(self.collected_relative_times) - x_margin, max(self.collected_relative_times) + x_margin)
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(min_speed - y_range*0.1, max_speed + y_range*0.1)
        plt.figtext(0.02, 0.02, f"Count: {len(self.collected_net_speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_filename, dpi=150)
        plt.close()

        # 生成 TXT
        txt_filename = f'{output_dir_path}/{base_filename}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"Net Speeds - Session {self.count_session_id} - {timestamp_str}\n")
            f.write("---------------------------------------\n")
            for i, (t, s) in enumerate(zip(self.collected_relative_times, self.collected_net_speeds)):
                f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            f.write("---------------------------------------\n")
            f.write(f"Total Points: {len(self.collected_net_speeds)}\n")
            f.write(f"Average Speed: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
            f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")

        # 生成 CSV
        csv_filename = f'{output_dir_path}/{base_filename}.csv'
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Point Number', 'Relative Time (s)', 'Speed (km/h)'])
            for i, (t, s) in enumerate(zip(self.collected_relative_times, self.collected_net_speeds)):
                writer.writerow([timestamp_str, i+1, f"{t:.2f}", f"{s:.1f}"])
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Total Points', len(self.collected_net_speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
        
        print(f"Output files saved to {output_dir_path}")
        
        # 回傳結果資料
        return {
            'chart': chart_filename,
            'txt': txt_filename,
            'csv': csv_filename,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'collected_net_speeds': self.collected_net_speeds,
            'collected_relative_times': self.collected_relative_times
        }

    def _draw_visualizations(self, display_frame, frame_data_obj: FrameData):
        vis_frame = display_frame
        
        # 减少视觉化频率，提高性能
        is_full_draw = frame_data_obj.frame_counter % VISUALIZATION_DRAW_INTERVAL == 0

        if is_full_draw:
            vis_frame = cv2.addWeighted(vis_frame, 1.0, self.static_overlay, 0.7, 0)
            if frame_data_obj.trajectory_points_global and len(frame_data_obj.trajectory_points_global) >= 2:
                pts = np.array(frame_data_obj.trajectory_points_global, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR_BGR, thickness=2)

        if frame_data_obj.ball_position_in_roi and frame_data_obj.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data_obj.ball_position_in_roi
            cv2.circle(frame_data_obj.roi_sub_frame, (cx_roi, cy_roi), 5, BALL_COLOR_BGR, -1)
            # 只在完整繪製幀繪製球輪廓，減少計算負擔
            if is_full_draw and frame_data_obj.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data_obj.roi_sub_frame, [frame_data_obj.ball_contour_in_roi], 0, CONTOUR_COLOR_BGR, 2)

            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(vis_frame, (cx_global, cy_global), 8, BALL_COLOR_BGR, -1)

        # 始終顯示關鍵信息
        cv2.putText(vis_frame, f"Speed: {frame_data_obj.current_ball_speed_kmh:.1f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        cv2.putText(vis_frame, f"FPS: {frame_data_obj.display_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, FPS_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
        
        # 顯示記錄狀態
        count_status_text = "RECORDING" if frame_data_obj.is_counting_active else "PAUSED"
        count_color = (0, 255, 0) if frame_data_obj.is_counting_active else (0, 0, 255)
        cv2.putText(vis_frame, f"Status: {count_status_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                   FONT_SCALE_VIS, count_color, FONT_THICKNESS_VIS)
        
        # 只在完整繪製幀或有新記錄時顯示詳細信息
        if is_full_draw or frame_data_obj.last_recorded_net_speed_kmh > 0:
            if frame_data_obj.last_recorded_net_speed_kmh > 0:
                cv2.putText(vis_frame, f"Last Net: {frame_data_obj.last_recorded_net_speed_kmh:.1f} km/h", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
            
            cv2.putText(vis_frame, f"Recorded: {len(frame_data_obj.collected_net_speeds)}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)
            
            if frame_data_obj.collected_relative_times:
                cv2.putText(vis_frame, f"Last Time: {frame_data_obj.collected_relative_times[-1]:.2f}s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_VIS, NET_SPEED_TEXT_COLOR_BGR, FONT_THICKNESS_VIS)

        # 始終顯示操作説明
        cv2.putText(vis_frame, self.instruction_text, (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if self.debug_mode and frame_data_obj.debug_display_text and is_full_draw:
            cv2.putText(vis_frame, frame_data_obj.debug_display_text, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
        return vis_frame

    def _check_timeout_and_reset(self):
        if time.monotonic() - self.last_detection_timestamp > self.detection_timeout_s:
            self.trajectory.clear()
            self.current_ball_speed_kmh = 0

    def process_single_frame(self, frame):
        start_time = time.monotonic()  # 性能監控開始
        
        self.frame_counter += 1
        self._update_display_fps()
            
        roi_sub_frame, gray_roi_for_fmo = self._preprocess_frame(frame) 
        
        motion_mask_roi = self._detect_fmo()
        
        ball_pos_in_roi, ball_contour_in_roi = None, None
        if motion_mask_roi is not None:
            ball_pos_in_roi, ball_contour_in_roi = self._detect_ball_in_roi(motion_mask_roi)
            self._calculate_ball_speed() 
        
        self._check_timeout_and_reset()
        
        if self.is_counting_active:
            self._process_crossing_events()

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
            debug_display_text=f"Traj: {len(self.trajectory)}, Events: {len(self.event_buffer_center_cross)}" if self.debug_mode else None,
            frame_counter=self.frame_counter
        )
        if self.trajectory:
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in self.trajectory]
        
        # 性能監控
        frame_processing_time = time.monotonic() - start_time
        self.performance_history.append(frame_processing_time)
        
        # 如果處理時間過長，自動調整視覺化頻率
        if len(self.performance_history) > 5:
            avg_process_time = sum(self.performance_history) / len(self.performance_history)
            if self.debug_mode and self.frame_counter % 30 == 0:
                print(f"Avg processing time: {avg_process_time*1000:.1f}ms (target: {1000/self.target_fps:.1f}ms)")
        
        return frame_data

    def run(self):
        print("=== Ping Pong Speed Tracker ===")
        print(self.instruction_text)
        print(f"Perspective: Near {self.near_side_width_cm}cm, Far {self.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.net_crossing_direction}")
        if self.debug_mode: print("Debug mode ENABLED.")

        self.running = True
        self.reader.start()
        
        window_name = 'Ping Pong Speed Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    if self.use_video_file: 
                        print("Video ended. Generating report automatically...")
                        self.running = False  # 設置為 False 將結束循環
                        break  # 確保退出循環
                    else: 
                        print("Camera error or stream ended.")
                        break
                
                frame_data_obj = self.process_single_frame(frame)
                
                display_frame = self._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                
                cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    self.running = False
                    print("Analysis terminated by user. Generating report...")
                    break
                elif key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        except KeyboardInterrupt:
            print("Process interrupted by user (Ctrl+C).")
        finally:
            self.running = False
            print("Shutting down...")
            self.reader.stop()
            print("Frame reader stopped.")
            cv2.destroyAllWindows()
            print("Speed tracking completed.")
            
            # 生成並回傳輸出結果
            return self._generate_outputs()

# %%
def _video_writer_thread(frame_queue, output_path, fourcc, fps, resolution):
    """專用於寫入影片的線程"""
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    while True:
        try:
            frame = frame_queue.get(timeout=5.0)
            if frame is None:  # 終止信號
                break
            out.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            print("Video writer timeout, exiting")
            break
    out.release()
    print("Video writer thread finished")

def record_video(name, device_index=DEFAULT_CAMERA_INDEX, 
                resolution=(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT), 
                target_fps=DEFAULT_TARGET_FPS, codec=DEFAULT_VIDEO_CODEC):
    """
    使用 AVFoundation 後端錄製高幀率影片
    按空白鍵開始/停止錄製
    
    參數:
    - name: 使用者姓名，用於檔案和資料夾命名
    - device_index: 攝像頭/擷取卡索引
    - resolution: 影片解析度，默認1920x1080 (1080p)
    - target_fps: 目標每秒幀數，默認120fps
    - codec: 輸出影片編碼格式，默認'avc1'
    
    回傳:
    - folder_path: 儲存資料夾路徑
    - output_path: 最後錄製的影片檔案路徑
    """
    # 建立時間戳記 (格式: YYYYMMDD_HHMMSS)
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建資料夾名稱和路徑
    folder_name = f"{name}_{base_timestamp}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    
    # 確保資料夾存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    
    print("=== High FPS Video Recorder ===")
    print(f"User: {name}")
    print(f"All recordings will be saved in: {folder_path}")
    print("Initializing camera backend...")
    
    # 使用 AVFoundation 後端初始化攝像頭/擷取卡
    backend = cv2.CAP_AVFOUNDATION
    print("Using backend: AVFoundation (macOS)")
    
    cap = cv2.VideoCapture(device_index, backend)
    
    # 檢查是否成功打開
    if not cap.isOpened():
        print(f"Error: Unable to open video device with index {device_index}")
        raise IOError(f"Cannot open video device: {device_index}")
    
    # 設定 MJPG 格式 (有助於提高幀率)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 設置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # 設置幀率
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # 檢查實際設置的參數
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera initialized")
    print(f"Reported resolution: {actual_width}x{actual_height}")
    print(f"Reported FPS: {actual_fps}fps")
    
    # 為輸出影片設定編碼器 - 使用 H.264 以獲得硬體加速
    output_fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # 變數初始化
    frame_queue = None
    writer_thread = None
    recording = False
    start_time = None
    frames_recorded = 0
    frame_times = []
    show_fps = True
    recording_count = 1  # 記錄當前是第幾段錄製
    output_path = None  # 將在錄製完成時設定
    
    print("\nReady!")
    print("Press 'SPACE' to start/stop recording")
    print("Press 'f' to show/hide FPS counter")
    print("Press 'q' to exit")
    
    # FPS 測量變數
    fps_start = time.time()
    fps_count = 0
    current_fps = 0
    
    # 記錄每秒的 FPS
    fps_history = []
    
    while True:
        # 擷取影像
        ret, frame = cap.read()
        
        if not ret:
            print("Unable to read video frame")
            time.sleep(0.1)
            continue
        
        # 計算即時FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count
            fps_history.append(current_fps)
            if len(fps_history) > 5:  # 保持最近5秒的數據
                fps_history.pop(0)
            fps_count = 0
            fps_start = time.time()
        
        # 如果正在錄製，寫入影片
        if recording and frame_queue is not None and not frame_queue.full():
            frame_queue.put(frame.copy())  # 複製幀以避免引用問題
            frame_times.append(time.time())
            frames_recorded += 1
        
        # 在畫面上顯示資訊
        status_text = "RECORDING..." if recording else "READY (Press SPACE to start)"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255) if recording else (0, 255, 0), 2)
        
        # 顯示FPS
        if show_fps:
            cv2.putText(frame, f"Real-time FPS: {current_fps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 0), 2)
            
            if len(fps_history) > 0:
                avg_fps = sum(fps_history) / len(fps_history)
                cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (255, 255, 0), 2)
        
        # 如果正在錄製，顯示錄製時間
        if recording and start_time:
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Recording Time: {elapsed:.1f}s", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 165, 255), 2)
            
            # 計算錄製中的平均FPS
            if len(frame_times) > 10:
                record_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                cv2.putText(frame, f"Recording FPS: {record_fps:.1f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 165, 255), 2)
        
        # 顯示預覽
        cv2.imshow('High FPS Camera', frame)
        
        # 處理按鍵
        key = cv2.waitKey(3) & 0xFF
        
        # 按q鍵退出
        if key == ord('q'):
            print("Program ended")
            break
        
        # 按f鍵切換FPS顯示
        if key == ord('f'):
            show_fps = not show_fps
            print(f"FPS display: {'ON' if show_fps else 'OFF'}")
        
        # 按空白鍵開始/停止錄製
        if key == 32:  # 空白鍵的ASCII碼是32
            if not recording:
                # 開始錄製
                recording = True
                start_time = time.time()
                frames_recorded = 0
                frame_times = []
                
                # 創建新的影片文件
                recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{name}_{recording_timestamp}.mp4"
                output_path = os.path.join(folder_path, output_filename)
                
                # 建立線程安全的佇列和寫入線程
                frame_queue = queue.Queue(maxsize=30)  # 限制佇列大小避免內存爆炸
                writer_thread = threading.Thread(
                    target=_video_writer_thread,
                    args=(frame_queue, output_path, output_fourcc, target_fps, (actual_width, actual_height)),
                    daemon=True
                )
                writer_thread.start()
                
                print(f"\nStarting recording #{recording_count}...")
                print(f"Output file: {output_path}")
                print(f"Target FPS: {target_fps}fps")
                
            else:
                # 停止錄製
                recording = False
                elapsed_time = time.time() - start_time
                
                # 計算實際幀率
                if len(frame_times) > 1:
                    actual_recorded_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                else:
                    actual_recorded_fps = 0
                
                # 關閉影片寫入器
                if frame_queue is not None:
                    frame_queue.put(None)  # 發送終止信號
                    if writer_thread is not None:
                        writer_thread.join(timeout=5.0)  # 等待寫入線程結束
                    frame_queue = None
                    writer_thread = None
                
                print(f"\nRecording #{recording_count} completed!")
                print(f"Recording time: {elapsed_time:.2f}s")
                print(f"Recorded frames: {frames_recorded}")
                print(f"Actual average FPS: {actual_recorded_fps:.2f}fps")
                print(f"Video saved as: {output_path}")
                
                # 錄製完成後自動結束錄影階段
                # 增加錄製計數（以防需要多次錄製）
                recording_count += 1
                break
    
    # 釋放資源
    cap.release()
    if frame_queue is not None and writer_thread is not None and writer_thread.is_alive():
        frame_queue.put(None)  # 發送終止信號
        writer_thread.join(timeout=5.0)  # 等待寫入線程結束
    cv2.destroyAllWindows()
    
    print("\nRecording completed")
    print(f"All videos saved in: {folder_path}")
    
    # 回傳資料夾路徑和最後錄製的影片路徑
    return folder_path, output_path

# %%
def main():
    """主執行函式，整合錄影和速度分析兩階段"""
    # 獲取使用者姓名
    user_name = input("請輸入您的姓名: ")
    user_name = "".join(c for c in user_name if c.isalnum() or c in "_ -")
    if not user_name:
        user_name = "User"  # 提供默認名稱
        print("使用默認使用者名稱: User")
    
    print("\n=== 第一階段：影片錄製 ===")
    folder_path, video_path = record_video(
        name=user_name,
        device_index=DEFAULT_CAMERA_INDEX,
        resolution=(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        target_fps=DEFAULT_TARGET_FPS,
        codec=DEFAULT_VIDEO_CODEC
    )
    
    if not video_path:
        print("錄影未完成，無法進行速度分析。")
        return None
    
    print("\n=== 第二階段：球速分析 ===")
    print("視訊即將播放。請按空白鍵開始記錄，再次按空白鍵暫停記錄。")
    print("這可以幫助您避免影片開頭或結尾的雜訊干擾分析結果。")
    print("影片播放完畢後，系統將自動生成報告。")
    print("您也可以隨時按 ESC 或 Q 鍵手動結束分析並生成報告。")
    
    tracker = PingPongSpeedTracker(
        video_source=video_path,
        table_length_cm=DEFAULT_TABLE_LENGTH_CM,
        use_video_file=True,
        target_fps=DEFAULT_TARGET_FPS,
        debug_mode=False,
        net_crossing_direction=NET_CROSSING_DIRECTION_DEFAULT,
        near_width_cm=NEAR_SIDE_WIDTH_CM_DEFAULT,
        far_width_cm=FAR_SIDE_WIDTH_CM_DEFAULT,
        output_folder=folder_path,
        output_basename=user_name
    )
    
    # 執行追蹤並取得結果
    result_data = tracker.run()
    
    return result_data, folder_path, user_name, video_path

# 執行主程式
if __name__ == "__main__":
    result = main()
    if result:
        result_data, folder_path, user_name, video_path = result
    else:
        print("程式執行失敗，無法產生結果。")

# %%
# 只有當此 Notebook 作為主程序運行時才執行
if 'result' not in locals():
    print("尚未執行主程式，請先執行上一個 Cell。")
else:
    if result and result_data:
        print("\n=== 乒乓球速度分析結果 ===")
        print(f"使用者: {user_name}")
        print(f"影片檔案: {video_path}")
        print(f"分析結果儲存於: {folder_path}")
        print(f"偵測到的球速次數: {len(result_data['collected_net_speeds'])}")
        print(f"平均速度: {result_data['average_speed']:.1f} km/h")
        print(f"最高速度: {result_data['max_speed']:.1f} km/h")
        print(f"最低速度: {result_data['min_speed']:.1f} km/h")
        
        # 顯示結果圖表（可選，取決於執行環境）
        print(f"\n速度圖表儲存於: {result_data['chart']}")
        print(f"速度數據(CSV): {result_data['csv']}")
        print(f"速度數據(TXT): {result_data['txt']}")
        
        # 在 Jupyter 中顯示圖表
        from IPython.display import Image, display
        try:
            display(Image(filename=result_data['chart']))
            print("圖表顯示成功")
        except:
            print("無法在此環境顯示圖表，請直接開啟圖表檔案。")
            
        # 讀取並顯示文字檔案內容
        try:
            with open(result_data['txt'], 'r') as f:
                txt_content = f.read()
            print("\n=== 速度數據摘要 ===")
            print(txt_content)
        except:
            print("無法讀取速度數據摘要。")
    else:
        print("未產生球速分析結果，請確認錄影和分析程序正確完成。")

    print("\n程式執行完成。")