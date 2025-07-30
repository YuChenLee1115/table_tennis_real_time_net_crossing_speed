"""
主要追蹤器類
整合所有模組並提供主要的追蹤功能
"""

import cv2
import time
from collections import deque
from typing import Optional
from .config import SystemConfig
from .events import FrameData, EventManager
from ..detection.fmo_detector import FMODetector  
from ..detection.ball_detector import BallDetector
from ..tracking.speed_calculator import SpeedCalculator
from ..tracking.crossing_detector import CrossingDetector
from ..io.frame_reader import FrameReader
from ..io.data_exporter import DataExporter
from ..visualization.renderer import Renderer
from ..utils.perspective import PerspectiveCorrector


class TableTennisTracker:
    """桌球速度追蹤器"""
    
    def __init__(self, config: SystemConfig, video_source=0, 
                 use_video_file: bool = False, video_file_path: Optional[str] = None):
        self.config = config
        self.video_source = video_source
        self.use_video_file = use_video_file
        self.video_file_path = video_file_path
        
        # 初始化核心組件  
        self._init_frame_reader()
        self._init_detection_components()
        self._init_tracking_components()
        self._init_io_components()
        self._init_visualization_components()
        
        # 狀態管理
        self.running = False
        self.frame_counter = 0
        self.is_counting_active = False
        self.count_session_id = 0
        self.output_generated_for_session = False
        
        # 速度收集
        self.collected_net_speeds = []
        self.collected_relative_times = []
        self.last_recorded_net_speed_kmh = 0.0
        
        # FPS計算
        self.frame_timestamps = deque(maxlen=config.max_frame_times_fps_calc)
        self.display_fps = 0.0
        
        # OpenCV優化
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(4)
        except AttributeError:
            pass
    
    def _init_frame_reader(self) -> None:
        """初始化影像讀取器"""
        self.frame_reader = FrameReader(
            self.video_source, self.config.camera, self.config.io, self.use_video_file
        )
        
        # 獲取實際參數
        self.actual_fps, self.frame_width, self.frame_height = self.frame_reader.get_properties()
        self.display_fps = self.actual_fps
        
        # 計算ROI區域
        self.roi_start_x = int(self.frame_width * self.config.detection.roi_start_ratio)
        self.roi_end_x = int(self.frame_width * self.config.detection.roi_end_ratio)  
        self.roi_top_y = 0
        self.roi_bottom_y = int(self.frame_height * self.config.detection.roi_bottom_ratio)
        self.roi_height = self.roi_bottom_y - self.roi_top_y
    
    def _init_detection_components(self) -> None:
        """初始化偵測組件"""
        self.fmo_detector = FMODetector(self.config.detection)
        self.ball_detector = BallDetector(
            self.config.detection, self.roi_start_x, self.roi_top_y, self.frame_width
        )
    
    def _init_tracking_components(self) -> None:
        """初始化追蹤組件"""
        # 透視校正器
        self.perspective_corrector = PerspectiveCorrector(
            self.config.tracking, self.roi_height, self.roi_bottom_y,
            self.roi_start_x, self.roi_end_x, self.frame_width
        )
        
        # 速度計算器
        self.speed_calculator = SpeedCalculator(
            self.config.tracking, self.perspective_corrector
        )
        
        # 事件管理器
        self.event_manager = EventManager(self.config.io.event_buffer_size)
        
        # 穿越偵測器
        self.crossing_detector = CrossingDetector(
            self.config.tracking, self.frame_width, self.frame_height,
            self.display_fps, self.event_manager
        )
    
    def _init_io_components(self) -> None:
        """初始化輸入輸出組件"""
        self.data_exporter = DataExporter(self.config.io)
    
    def _init_visualization_components(self) -> None:
        """初始化視覺化組件"""
        self.renderer = Renderer(
            self.config.visualization, self.frame_width, self.frame_height,
            self.roi_start_x, self.roi_end_x, self.roi_top_y, self.roi_bottom_y
        )
    
    def process_frame(self, frame) -> FrameData:
        """處理單幀"""
        self.frame_counter += 1
        self._update_fps()
        
        # 提取ROI並進行FMO偵測
        roi_frame = frame[self.roi_top_y:self.roi_bottom_y, self.roi_start_x:self.roi_end_x]
        self.fmo_detector.preprocess_frame(roi_frame)
        motion_mask = self.fmo_detector.detect_motion()
        
        # 球體偵測
        ball_event = None
        if motion_mask is not None:
            current_timestamp = time.monotonic()
            if self.use_video_file:
                current_timestamp = self.frame_counter / self.actual_fps
                
            ball_event = self.ball_detector.detect_from_motion_mask(
                motion_mask, current_timestamp, self.use_video_file
            )
        
        # 處理球體偵測結果
        if ball_event:
            # 更新速度
            trajectory = self.ball_detector.get_trajectory()
            current_speed = self.speed_calculator.calculate_speed(trajectory)
            
            # 穿越偵測（僅在計數激活時）
            if self.is_counting_active:
                self.crossing_detector.detect_crossing(
                    ball_event.position.x_global, ball_event.position.y_global,
                    ball_event.position.timestamp, trajectory, current_speed
                )
        else:
            current_speed = self.speed_calculator.get_current_speed()
            # 重置穿越偵測器的當前球位置
            self.crossing_detector.last_ball_x_global = None
        
        # 檢查偵測超時
        if self.ball_detector.is_detection_timeout(time.monotonic()):
            self.ball_detector.reset_trajectory()
            self.speed_calculator.reset()
        
        # 處理穿越事件
        if self.is_counting_active:
            self._process_crossing_events()
        
        # 準備幀數據
        frame_data = FrameData(
            frame=frame,
            roi_sub_frame=roi_frame,
            ball_position_in_roi=(ball_event.position.x_roi, ball_event.position.y_roi) if ball_event else None,
            ball_contour_in_roi=ball_event.contour if ball_event else None,
            current_ball_speed_kmh=current_speed,
            display_fps=self.display_fps,
            is_counting_active=self.is_counting_active,
            collected_net_speeds=list(self.collected_net_speeds),
            last_recorded_net_speed_kmh=self.last_recorded_net_speed_kmh,
            collected_relative_times=list(self.collected_relative_times),
            frame_counter=self.frame_counter
        )
        
        # 添加軌跡點
        trajectory = self.ball_detector.get_trajectory()
        if trajectory:
            frame_data.trajectory_points_global = [(int(p[0]), int(p[1])) for p in trajectory]
        
        # 添加除錯信息
        if self.config.debug_mode:
            frame_data.debug_display_text = self._generate_debug_text()
        
        return frame_data
    
    def _update_fps(self) -> None:
        """更新FPS計算"""
        if self.use_video_file:
            self.display_fps = self.actual_fps
            return
            
        now = time.monotonic()
        self.frame_timestamps.append(now)
        
        if len(self.frame_timestamps) >= 2:
            elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if elapsed > 0:
                measured_fps = (len(self.frame_timestamps) - 1) / elapsed
                self.display_fps = (
                    (1 - self.config.fps_smoothing_factor) * self.display_fps +
                    self.config.fps_smoothing_factor * measured_fps
                )
    
    def _process_crossing_events(self) -> None:
        """處理穿越事件"""
        if self.output_generated_for_session:
            return
            
        current_time = time.monotonic()
        if self.use_video_file:
            current_time = self.frame_counter / self.actual_fps
        
        # 處理事件
        committed_event = self.event_manager.process_events(
            current_time, self.config.tracking.crossing_cooldown_s
        )
        
        if committed_event and len(self.collected_net_speeds) < self.config.tracking.max_net_speeds_to_collect:
            # 記錄速度
            if not self.event_manager.timing_started:
                self.event_manager.timing_started = True
                self.event_manager.first_ball_crossing_timestamp = committed_event.timestamp
            
            relative_time = (committed_event.timestamp - 
                           self.event_manager.first_ball_crossing_timestamp 
                           if self.event_manager.timing_started else 0.0)
            
            self.last_recorded_net_speed_kmh = committed_event.speed_kmh
            self.collected_net_speeds.append(committed_event.speed_kmh)
            self.collected_relative_times.append(round(relative_time, 2))
            
            # 確認球在左側
            self.crossing_detector.ball_on_left_of_center = True
            
            if self.config.debug_mode:
                event_type = "PREDICTED" if committed_event.predicted else "ACTUAL"
                print(f"--- COMMITTED {event_type} Event #{len(self.collected_net_speeds)}: "
                      f"Speed {committed_event.speed_kmh:.1f} at Rel.T {relative_time:.2f}s ---")
        
        # 清理舊事件
        self.event_manager.cleanup_old_events(current_time)
        
        # 檢查是否收集完成
        if (len(self.collected_net_speeds) >= self.config.tracking.max_net_speeds_to_collect and 
            not self.output_generated_for_session):
            print(f"Collected {self.config.tracking.max_net_speeds_to_collect} net speeds. Generating output.")
            self._generate_output()
            self.output_generated_for_session = True
            
            if self.config.tracking.auto_stop_after_collection:
                self.is_counting_active = False
    
    def _generate_debug_text(self) -> str:
        """生成除錯文字"""
        trajectory_len = len(self.ball_detector.get_trajectory())
        buffer_len = len(self.event_manager.crossing_events)
        on_left = "Y" if self.crossing_detector.ball_on_left_of_center else "N"
        last_commit_time = self.event_manager.last_committed_crossing_time
        
        return f"Traj:{trajectory_len} EvtBuf:{buffer_len} OnLeft:{on_left} LastCommitT:{last_commit_time:.2f}"
    
    def toggle_counting(self) -> None:
        """切換計數狀態"""
        self.is_counting_active = not self.is_counting_active
        
        if self.is_counting_active:
            self.count_session_id += 1
            self.collected_net_speeds.clear()
            self.collected_relative_times.clear()
            self.last_recorded_net_speed_kmh = 0.0
            self.output_generated_for_session = False
            
            # 重置相關狀態
            self.event_manager.reset_session()
            self.crossing_detector.reset_state()
            
            print(f"Counting ON (Session #{self.count_session_id}) - "
                  f"Target: {self.config.tracking.max_net_speeds_to_collect} speeds.")
        else:
            print(f"Counting OFF (Session #{self.count_session_id}).")
            if self.collected_net_speeds and not self.output_generated_for_session:
                print(f"Collected {len(self.collected_net_speeds)} speeds. Generating output...")
                self._generate_output()
            self.output_generated_for_session = True
    
    def _generate_output(self) -> None:
        """生成輸出文件"""
        if not self.collected_net_speeds:
            print("No speed data to generate output.")
            return
            
        self.data_exporter.export_async(
            self.collected_net_speeds, self.collected_relative_times,
            self.count_session_id, self.use_video_file, self.video_file_path
        )
    
    def run(self) -> None:
        """運行追蹤器"""
        print("=== Table Tennis Speed Tracker (Refactored) ===")
        print(f"Perspective: Near {self.config.tracking.near_side_width_cm}cm, "
              f"Far {self.config.tracking.far_side_width_cm}cm")
        print(f"Net crossing direction: {self.config.tracking.net_crossing_direction}")
        print(f"Target speeds to collect: {self.config.tracking.max_net_speeds_to_collect}")
        print(f"Crossing cooldown: {self.config.tracking.crossing_cooldown_s}s")
        if self.config.debug_mode:
            print("Debug mode ENABLED.")
        
        self.running = True
        self.frame_reader.start()
        
        window_name = 'Table Tennis Speed Tracker (Refactored)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                ret, frame = self.frame_reader.read()
                if not ret or frame is None:
                    if self.use_video_file:
                        print("Video ended or frame read error.")
                    else:
                        print("Camera error or stream ended.")
                    
                    if (self.is_counting_active and self.collected_net_speeds and 
                        not self.output_generated_for_session):
                        print("End of stream with pending data. Generating output.")
                        self._generate_output()
                        self.output_generated_for_session = True
                    break
                
                # 處理幀
                frame_data = self.process_frame(frame)
                
                # 渲染顯示
                display_frame = self.renderer.render_frame(
                    frame_data, self.config.tracking.max_net_speeds_to_collect, 
                    self.config.debug_mode
                )
                cv2.imshow(window_name, display_frame)
                
                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    self.running = False
                    if (self.is_counting_active and self.collected_net_speeds and 
                        not self.output_generated_for_session):
                        print("Quitting with pending data. Generating output.")
                        self._generate_output()
                        self.output_generated_for_session = True
                    break
                elif key == ord(' '):
                    self.toggle_counting()
                elif key == ord('d'):
                    self.config.debug_mode = not self.config.debug_mode
                    print(f"Debug mode: {'ON' if self.config.debug_mode else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("Process interrupted by user (Ctrl+C).")
            if (self.is_counting_active and self.collected_net_speeds and 
                not self.output_generated_for_session):
                print("Interrupted with pending data. Generating output.")
                self._generate_output()
                self.output_generated_for_session = True
        finally:
            self.running = False
            print("Shutting down...")
            self.frame_reader.stop()
            print("Frame reader stopped.")
            self.data_exporter.shutdown()
            print("Data exporter stopped.")
            cv2.destroyAllWindows()
            print("System shutdown complete.")