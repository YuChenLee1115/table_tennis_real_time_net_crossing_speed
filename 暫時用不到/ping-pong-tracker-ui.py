import sys
import os
import time
import cv2
import numpy as np
import threading
import queue
from datetime import datetime
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTabWidget, QGroupBox, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QMessageBox,
                             QLineEdit, QCheckBox, QSplitter, QSizePolicy, QFrame,
                             QTextEdit, QProgressBar, QColorDialog, QScrollArea,
                             QSlider, QToolButton, QStackedWidget, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor

# 導入您的主要乒乓球速度追蹤模組
from table_tennis_speed_track import (PingPongSpeedTracker, DEFAULT_CAMERA_INDEX, 
                                      DEFAULT_TABLE_LENGTH_CM, DEFAULT_DETECTION_TIMEOUT,
                                      DEFAULT_TARGET_FPS, DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT,
                                      NET_CROSSING_DIRECTION_DEFAULT, NEAR_SIDE_WIDTH_CM_DEFAULT,
                                      FAR_SIDE_WIDTH_CM_DEFAULT, SPEED_SMOOTHING_FACTOR,
                                      KMH_CONVERSION_FACTOR, DEFAULT_ROI_START_RATIO, 
                                      DEFAULT_ROI_END_RATIO, DEFAULT_ROI_BOTTOM_RATIO,
                                      MAX_TRAJECTORY_POINTS, MIN_BALL_AREA_PX, MAX_BALL_AREA_PX,
                                      MIN_BALL_CIRCULARITY, CENTER_LINE_WIDTH_PIXELS,
                                      CENTER_DETECTION_COOLDOWN_S, TRAJECTORY_COLOR_BGR,
                                      BALL_COLOR_BGR, ROI_COLOR_BGR, SPEED_TEXT_COLOR_BGR,
                                      FPS_TEXT_COLOR_BGR, CENTER_LINE_COLOR_BGR,
                                      NET_SPEED_TEXT_COLOR_BGR, VISUALIZATION_DRAW_INTERVAL)

class VideoThread(QThread):
    """處理影片串流的執行緒"""
    update_frame = pyqtSignal(np.ndarray)
    update_fps = pyqtSignal(float)
    recording_status = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index=0, target_fps=120, resolution=(1280, 720)):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.resolution = resolution
        self.running = False
        self.recording = False
        self.output_video = None
        self.output_path = None
        self.output_folder = None
        self.user_name = "User"
        
        # FPS計算
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.display_fps = 0.0
    
    def set_user_name(self, name):
        self.user_name = name
    
    def run(self):
        # 使用 AVFoundation 後端在 macOS 上
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        
        # 配置相機
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not self.cap.isOpened():
            self.error_occurred.emit("無法開啟相機")
            return
        
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.running:  # 只在執行緒運行中時發出錯誤
                    self.error_occurred.emit("相機讀取錯誤")
                break
                
            # 更新FPS計算
            self.fps_frame_count += 1
            elapsed_time = time.time() - self.fps_start_time
            if elapsed_time >= 1.0:
                self.display_fps = self.fps_frame_count / elapsed_time
                self.update_fps.emit(self.display_fps)
                self.fps_frame_count = 0
                self.fps_start_time = time.time()
            
            # 如果錄製中，寫入影片
            if self.recording and self.output_video is not None:
                self.output_video.write(frame)
                
            # 在幀上添加 FPS 和狀態信息 (使用英文)
            status_text = "RECORDING..." if self.recording else "LIVE VIEW"
            status_color = (0, 0, 255) if self.recording else (0, 255, 0)
            
            cv2.putText(frame, f"FPS: {self.display_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            self.update_frame.emit(frame)
            
            # 控制循環速度，避免CPU過載
            time.sleep(max(0.001, 1.0/(self.target_fps*2)))
    
    def start_recording(self):
        if self.recording:
            return False
            
        # 建立輸出資料夾
        if not self.output_folder:
            base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self.user_name}_{base_timestamp}"
            self.output_folder = os.path.join(os.getcwd(), folder_name)
            
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        
        # 創建新的影片檔案
        recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.user_name}_{recording_timestamp}.mp4"
        self.output_path = os.path.join(self.output_folder, output_filename)
        
        # 獲取實際影片參數
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 創建影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 編碼
        self.output_video = cv2.VideoWriter(
            self.output_path, fourcc, self.target_fps, (width, height)
        )
        
        if not self.output_video.isOpened():
            self.error_occurred.emit(f"無法創建輸出影片: {self.output_path}")
            return False
            
        self.recording = True
        self.recording_status.emit(True)
        return True
    
    def stop_recording(self):
        if not self.recording:
            return False
            
        self.recording = False
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None
            
        self.recording_status.emit(False)
        return self.output_path
    
    def stop(self):
        self.running = False
        self.recording = False
        self.wait()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.output_video is not None:
            self.output_video.release()


class AnalysisThread(QThread):
    """處理影片分析的執行緒"""
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.output_folder = None
        self.output_basename = "User"
        self.table_length_cm = DEFAULT_TABLE_LENGTH_CM
        self.net_crossing_direction = NET_CROSSING_DIRECTION_DEFAULT
        self.near_width_cm = NEAR_SIDE_WIDTH_CM_DEFAULT
        self.far_width_cm = FAR_SIDE_WIDTH_CM_DEFAULT
        self.debug_mode = False
        self.tracker = None
        self.running = False
        self.start_analysis = False
        
        # 額外參數
        self.detection_timeout = DEFAULT_DETECTION_TIMEOUT
        self.roi_start_ratio = DEFAULT_ROI_START_RATIO
        self.roi_end_ratio = DEFAULT_ROI_END_RATIO
        self.roi_bottom_ratio = DEFAULT_ROI_BOTTOM_RATIO
        self.max_trajectory_points = MAX_TRAJECTORY_POINTS
        self.speed_smoothing_factor = SPEED_SMOOTHING_FACTOR
        self.min_ball_area = MIN_BALL_AREA_PX
        self.max_ball_area = MAX_BALL_AREA_PX
        self.min_ball_circularity = MIN_BALL_CIRCULARITY
        self.center_line_width = CENTER_LINE_WIDTH_PIXELS
        self.center_detection_cooldown = CENTER_DETECTION_COOLDOWN_S
        self.visualization_draw_interval = VISUALIZATION_DRAW_INTERVAL
        
        # 顏色設定
        self.trajectory_color = TRAJECTORY_COLOR_BGR
        self.ball_color = BALL_COLOR_BGR
        self.roi_color = ROI_COLOR_BGR
        self.speed_text_color = SPEED_TEXT_COLOR_BGR
        self.fps_text_color = FPS_TEXT_COLOR_BGR
        self.center_line_color = CENTER_LINE_COLOR_BGR
        self.net_speed_text_color = NET_SPEED_TEXT_COLOR_BGR
        
        # 用於控制播放速度
        self.playback_speed = 1.0
        
    def set_parameters(self, video_path, output_folder, output_basename,
                      table_length_cm, net_crossing_direction, 
                      near_width_cm, far_width_cm, debug_mode):
        self.video_path = video_path
        self.output_folder = output_folder
        self.output_basename = output_basename
        self.table_length_cm = table_length_cm
        self.net_crossing_direction = net_crossing_direction
        self.near_width_cm = near_width_cm
        self.far_width_cm = far_width_cm
        self.debug_mode = debug_mode
    
    def set_advanced_parameters(self, params_dict):
        """設定進階參數"""
        # 使用 params_dict 中的值更新成員變數，如果存在的話
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_playback_speed(self, speed):
        self.playback_speed = speed
    
    def run(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.error_occurred.emit("影片檔案不存在")
            return
            
        try:
            self.running = True
            self.update_status.emit("初始化分析...")
            
            # 創建追蹤器實例，注入所有參數
            self.tracker = PingPongSpeedTracker(
                video_source=self.video_path,
                table_length_cm=self.table_length_cm,
                detection_timeout_s=self.detection_timeout,
                use_video_file=True,
                target_fps=DEFAULT_TARGET_FPS,
                debug_mode=self.debug_mode,
                net_crossing_direction=self.net_crossing_direction,
                near_width_cm=self.near_width_cm,
                far_width_cm=self.far_width_cm,
                output_folder=self.output_folder,
                output_basename=self.output_basename
            )
            
            # 修改追蹤器的其他進階參數
            self.tracker.roi_start_x = int(self.tracker.frame_width * self.roi_start_ratio)
            self.tracker.roi_end_x = int(self.tracker.frame_width * self.roi_end_ratio)
            self.tracker.roi_bottom_y = int(self.tracker.frame_height * self.roi_bottom_ratio)
            self.tracker.roi_height_px = self.tracker.roi_bottom_y - self.tracker.roi_top_y
            
            # 更新參數
            self.tracker.trajectory = queue.deque(maxlen=self.max_trajectory_points)
            self.tracker._precalculate_overlay()  # 重新計算疊加層
            
            # 開始讀取影片而非直接運行追蹤器
            self.tracker.reader.start()
            
            # 獲取影片總幀數以計算進度
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            frame_counter = 0
            last_process_time = time.time()
            
            while self.running:
                ret, frame = self.tracker.reader.read()
                if not ret or frame is None:
                    self.update_status.emit("影片分析完成")
                    break
                
                # 計算進度
                frame_counter += 1
                progress = int((frame_counter / total_frames) * 100) if total_frames > 0 else 0
                self.progress_update.emit(progress)
                
                if self.start_analysis:
                    # 處理幀並更新UI
                    frame_data_obj = self.tracker.process_single_frame(frame)
                    display_frame = self.tracker._draw_visualizations(frame_data_obj.frame, frame_data_obj)
                    self.update_frame.emit(display_frame)
                    self.update_status.emit(f"正在分析 - 當前速度: {frame_data_obj.current_ball_speed_kmh:.1f} km/h")
                else:
                    # 只顯示原始幀，不分析
                    # 在幀上添加說明 (使用英文)
                    cv2.putText(frame, "Press START ANALYSIS button to begin processing", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    self.update_frame.emit(frame)
                    self.update_status.emit("準備分析 - 按下開始分析按鈕")
                
                # 控制播放速度 (1.0為正常速度)
                current_time = time.time()
                elapsed = current_time - last_process_time
                target_frame_time = 1.0 / (self.tracker.actual_fps * self.playback_speed)
                
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                
                last_process_time = time.time()
                
            # 生成分析結果
            if self.start_analysis:
                result_data = self.tracker._generate_outputs()
                self.analysis_complete.emit(result_data)
            
        except Exception as e:
            self.error_occurred.emit(f"分析過程中發生錯誤: {str(e)}")
        finally:
            self.running = False
            if self.tracker:
                self.tracker.reader.stop()
    
    def toggle_analysis(self):
        self.start_analysis = not self.start_analysis
        status = "開始" if self.start_analysis else "暫停"
        self.update_status.emit(f"{status}分析")
        
        # 如果開始分析，也啟動追蹤器的計數功能
        if self.tracker and self.start_analysis:
            self.tracker.is_counting_active = True
        elif self.tracker:
            self.tracker.is_counting_active = False
    
    def stop(self):
        self.running = False
        self.wait()


class ColorButton(QPushButton):
    """可選擇顏色的按鈕"""
    def __init__(self, color=(0, 0, 255), parent=None):
        super().__init__(parent)
        self.color_bgr = color
        self.setMinimumWidth(80)
        self.updateColor()
        self.clicked.connect(self.showColorDialog)
        
    def updateColor(self):
        r, g, b = self.color_bgr[2], self.color_bgr[1], self.color_bgr[0]  # BGR to RGB
        self.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); color: {'black' if (r+g+b)/3 > 128 else 'white'};")
        self.setText(f"({r},{g},{b})")
        
    def showColorDialog(self):
        r, g, b = self.color_bgr[2], self.color_bgr[1], self.color_bgr[0]  # BGR to RGB
        color = QColorDialog.getColor(QColor(r, g, b), self, "選擇顏色")
        if color.isValid():
            self.color_bgr = (color.blue(), color.green(), color.red())  # RGB to BGR
            self.updateColor()
            
    def getColorBGR(self):
        return self.color_bgr


class AdvancedSettingsWidget(QWidget):
    """進階設定面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        # 使用滾動區域來包含所有設定
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        settings_widget = QWidget()
        main_layout = QVBoxLayout(settings_widget)
        
        # 使用分頁控制進階設定分類
        tab_widget = QTabWidget()
        
        # === 1. 偵測參數 ===
        detection_tab = QWidget()
        detection_layout = QFormLayout(detection_tab)
        
        self.detection_timeout = QDoubleSpinBox()
        self.detection_timeout.setRange(0.01, 1.0)
        self.detection_timeout.setValue(DEFAULT_DETECTION_TIMEOUT)
        self.detection_timeout.setSingleStep(0.05)
        self.detection_timeout.setDecimals(2)
        self.detection_timeout.setSuffix(" 秒")
        detection_layout.addRow("偵測逾時:", self.detection_timeout)
        
        # ROI 參數
        roi_group = QGroupBox("感興趣區域 (ROI)")
        roi_layout = QFormLayout()
        
        self.roi_start_ratio = QDoubleSpinBox()
        self.roi_start_ratio.setRange(0.0, 0.8)
        self.roi_start_ratio.setValue(DEFAULT_ROI_START_RATIO)
        self.roi_start_ratio.setSingleStep(0.05)
        self.roi_start_ratio.setDecimals(2)
        roi_layout.addRow("ROI 起始比例:", self.roi_start_ratio)
        
        self.roi_end_ratio = QDoubleSpinBox()
        self.roi_end_ratio.setRange(0.2, 1.0)
        self.roi_end_ratio.setValue(DEFAULT_ROI_END_RATIO)
        self.roi_end_ratio.setSingleStep(0.05)
        self.roi_end_ratio.setDecimals(2)
        roi_layout.addRow("ROI 結束比例:", self.roi_end_ratio)
        
        self.roi_bottom_ratio = QDoubleSpinBox()
        self.roi_bottom_ratio.setRange(0.1, 1.0)
        self.roi_bottom_ratio.setValue(DEFAULT_ROI_BOTTOM_RATIO)
        self.roi_bottom_ratio.setSingleStep(0.05)
        self.roi_bottom_ratio.setDecimals(2)
        roi_layout.addRow("ROI 底部比例:", self.roi_bottom_ratio)
        
        roi_group.setLayout(roi_layout)
        detection_layout.addRow(roi_group)
        
        # 球體偵測參數
        ball_group = QGroupBox("球體偵測參數")
        ball_layout = QFormLayout()
        
        self.min_ball_area = QSpinBox()
        self.min_ball_area.setRange(10, 1000)
        self.min_ball_area.setValue(MIN_BALL_AREA_PX)
        self.min_ball_area.setSingleStep(10)
        self.min_ball_area.setSuffix(" 像素")
        ball_layout.addRow("最小球面積:", self.min_ball_area)
        
        self.max_ball_area = QSpinBox()
        self.max_ball_area.setRange(100, 10000)
        self.max_ball_area.setValue(MAX_BALL_AREA_PX)
        self.max_ball_area.setSingleStep(100)
        self.max_ball_area.setSuffix(" 像素")
        ball_layout.addRow("最大球面積:", self.max_ball_area)
        
        self.min_ball_circularity = QDoubleSpinBox()
        self.min_ball_circularity.setRange(0.1, 1.0)
        self.min_ball_circularity.setValue(MIN_BALL_CIRCULARITY)
        self.min_ball_circularity.setSingleStep(0.05)
        self.min_ball_circularity.setDecimals(2)
        ball_layout.addRow("最小圓形度:", self.min_ball_circularity)
        
        ball_group.setLayout(ball_layout)
        detection_layout.addRow(ball_group)
        
        tab_widget.addTab(detection_tab, "偵測參數")
        
        # === 2. 速度參數 ===
        speed_tab = QWidget()
        speed_layout = QFormLayout(speed_tab)
        
        self.max_trajectory_points = QSpinBox()
        self.max_trajectory_points.setRange(10, 500)
        self.max_trajectory_points.setValue(MAX_TRAJECTORY_POINTS)
        self.max_trajectory_points.setSingleStep(10)
        self.max_trajectory_points.setSuffix(" 點")
        speed_layout.addRow("最大軌跡點數:", self.max_trajectory_points)
        
        self.speed_smoothing_factor = QDoubleSpinBox()
        self.speed_smoothing_factor.setRange(0.01, 1.0)
        self.speed_smoothing_factor.setValue(SPEED_SMOOTHING_FACTOR)
        self.speed_smoothing_factor.setSingleStep(0.05)
        self.speed_smoothing_factor.setDecimals(2)
        speed_layout.addRow("速度平滑因子:", self.speed_smoothing_factor)
        
        # 中心線參數
        center_group = QGroupBox("中心線檢測參數")
        center_layout = QFormLayout()
        
        self.center_line_width = QSpinBox()
        self.center_line_width.setRange(50, 1000)
        self.center_line_width.setValue(CENTER_LINE_WIDTH_PIXELS)
        self.center_line_width.setSingleStep(50)
        self.center_line_width.setSuffix(" 像素")
        center_layout.addRow("中心線寬度:", self.center_line_width)
        
        self.center_detection_cooldown = QDoubleSpinBox()
        self.center_detection_cooldown.setRange(0.001, 0.5)
        self.center_detection_cooldown.setValue(CENTER_DETECTION_COOLDOWN_S)
        self.center_detection_cooldown.setSingleStep(0.005)
        self.center_detection_cooldown.setDecimals(3)
        self.center_detection_cooldown.setSuffix(" 秒")
        center_layout.addRow("過網檢測冷卻:", self.center_detection_cooldown)
        
        center_group.setLayout(center_layout)
        speed_layout.addRow(center_group)
        
        tab_widget.addTab(speed_tab, "速度參數")
        
        # === 3. 視覺化參數 ===
        visual_tab = QWidget()
        visual_layout = QFormLayout(visual_tab)
        
        self.visualization_draw_interval = QSpinBox()
        self.visualization_draw_interval.setRange(1, 10)
        self.visualization_draw_interval.setValue(VISUALIZATION_DRAW_INTERVAL)
        self.visualization_draw_interval.setSingleStep(1)
        self.visualization_draw_interval.setSuffix(" 幀")
        visual_layout.addRow("繪製間隔:", self.visualization_draw_interval)
        
        # 顏色配置
        color_group = QGroupBox("顏色設定")
        color_layout = QGridLayout()
        
        # 使用自定義顏色按鈕
        self.trajectory_color_btn = ColorButton(TRAJECTORY_COLOR_BGR)
        self.ball_color_btn = ColorButton(BALL_COLOR_BGR)
        self.roi_color_btn = ColorButton(ROI_COLOR_BGR)
        self.speed_text_color_btn = ColorButton(SPEED_TEXT_COLOR_BGR)
        self.fps_text_color_btn = ColorButton(FPS_TEXT_COLOR_BGR)
        self.center_line_color_btn = ColorButton(CENTER_LINE_COLOR_BGR)
        self.net_speed_text_color_btn = ColorButton(NET_SPEED_TEXT_COLOR_BGR)
        
        # 配置網格佈局
        color_layout.addWidget(QLabel("軌跡顏色:"), 0, 0)
        color_layout.addWidget(self.trajectory_color_btn, 0, 1)
        color_layout.addWidget(QLabel("球體顏色:"), 0, 2)
        color_layout.addWidget(self.ball_color_btn, 0, 3)
        
        color_layout.addWidget(QLabel("ROI 顏色:"), 1, 0)
        color_layout.addWidget(self.roi_color_btn, 1, 1)
        color_layout.addWidget(QLabel("速度文字顏色:"), 1, 2)
        color_layout.addWidget(self.speed_text_color_btn, 1, 3)
        
        color_layout.addWidget(QLabel("FPS 文字顏色:"), 2, 0)
        color_layout.addWidget(self.fps_text_color_btn, 2, 1)
        color_layout.addWidget(QLabel("中心線顏色:"), 2, 2)
        color_layout.addWidget(self.center_line_color_btn, 2, 3)
        
        color_layout.addWidget(QLabel("網速文字顏色:"), 3, 0)
        color_layout.addWidget(self.net_speed_text_color_btn, 3, 1)
        
        color_group.setLayout(color_layout)
        visual_layout.addRow(color_group)
        
        # 添加恢復預設值按鈕
        reset_btn = QPushButton("恢復預設值")
        reset_btn.clicked.connect(self.resetToDefaults)
        visual_layout.addRow("", reset_btn)
        
        tab_widget.addTab(visual_tab, "視覺化參數")
        
        main_layout.addWidget(tab_widget)
        
        scroll_area.setWidget(settings_widget)
        
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)
        
    def resetToDefaults(self):
        """恢復所有參數為預設值"""
        # 偵測參數
        self.detection_timeout.setValue(DEFAULT_DETECTION_TIMEOUT)
        self.roi_start_ratio.setValue(DEFAULT_ROI_START_RATIO)
        self.roi_end_ratio.setValue(DEFAULT_ROI_END_RATIO)
        self.roi_bottom_ratio.setValue(DEFAULT_ROI_BOTTOM_RATIO)
        self.min_ball_area.setValue(MIN_BALL_AREA_PX)
        self.max_ball_area.setValue(MAX_BALL_AREA_PX)
        self.min_ball_circularity.setValue(MIN_BALL_CIRCULARITY)
        
        # 速度參數
        self.max_trajectory_points.setValue(MAX_TRAJECTORY_POINTS)
        self.speed_smoothing_factor.setValue(SPEED_SMOOTHING_FACTOR)
        self.center_line_width.setValue(CENTER_LINE_WIDTH_PIXELS)
        self.center_detection_cooldown.setValue(CENTER_DETECTION_COOLDOWN_S)
        
        # 視覺化參數
        self.visualization_draw_interval.setValue(VISUALIZATION_DRAW_INTERVAL)
        
        # 顏色設定
        self.trajectory_color_btn.color_bgr = TRAJECTORY_COLOR_BGR
        self.trajectory_color_btn.updateColor()
        self.ball_color_btn.color_bgr = BALL_COLOR_BGR
        self.ball_color_btn.updateColor()
        self.roi_color_btn.color_bgr = ROI_COLOR_BGR
        self.roi_color_btn.updateColor()
        self.speed_text_color_btn.color_bgr = SPEED_TEXT_COLOR_BGR
        self.speed_text_color_btn.updateColor()
        self.fps_text_color_btn.color_bgr = FPS_TEXT_COLOR_BGR
        self.fps_text_color_btn.updateColor()
        self.center_line_color_btn.color_bgr = CENTER_LINE_COLOR_BGR
        self.center_line_color_btn.updateColor()
        self.net_speed_text_color_btn.color_bgr = NET_SPEED_TEXT_COLOR_BGR
        self.net_speed_text_color_btn.updateColor()
        
    def getParameters(self):
        """獲取所有參數值"""
        return {
            "detection_timeout": self.detection_timeout.value(),
            "roi_start_ratio": self.roi_start_ratio.value(),
            "roi_end_ratio": self.roi_end_ratio.value(),
            "roi_bottom_ratio": self.roi_bottom_ratio.value(),
            "max_trajectory_points": self.max_trajectory_points.value(),
            "speed_smoothing_factor": self.speed_smoothing_factor.value(),
            "min_ball_area": self.min_ball_area.value(),
            "max_ball_area": self.max_ball_area.value(),
            "min_ball_circularity": self.min_ball_circularity.value(),
            "center_line_width": self.center_line_width.value(),
            "center_detection_cooldown": self.center_detection_cooldown.value(),
            "visualization_draw_interval": self.visualization_draw_interval.value(),
            "trajectory_color": self.trajectory_color_btn.getColorBGR(),
            "ball_color": self.ball_color_btn.getColorBGR(),
            "roi_color": self.roi_color_btn.getColorBGR(),
            "speed_text_color": self.speed_text_color_btn.getColorBGR(),
            "fps_text_color": self.fps_text_color_btn.getColorBGR(),
            "center_line_color": self.center_line_color_btn.getColorBGR(),
            "net_speed_text_color": self.net_speed_text_color_btn.getColorBGR()
        }


class ResultsViewer(QWidget):
    """結果顯示元件"""
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 標題標籤
        self.title_label = QLabel("分析結果")
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        
        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 統計信息
        stats_group = QGroupBox("統計資訊")
        stats_layout = QFormLayout()
        
        self.count_label = QLabel("0")
        self.avg_speed_label = QLabel("0.0 km/h")
        self.max_speed_label = QLabel("0.0 km/h")
        self.min_speed_label = QLabel("0.0 km/h")
        
        stats_layout.addRow("偵測到的球速次數:", self.count_label)
        stats_layout.addRow("平均速度:", self.avg_speed_label)
        stats_layout.addRow("最高速度:", self.max_speed_label)
        stats_layout.addRow("最低速度:", self.min_speed_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 圖表預覽
        self.chart_preview = QLabel("圖表將在此顯示")
        self.chart_preview.setAlignment(Qt.AlignCenter)
        self.chart_preview.setMinimumHeight(300)
        layout.addWidget(self.chart_preview)
        
        # 詳細數據顯示
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
        
        # 按鈕
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("另存結果")
        self.save_button.clicked.connect(self.save_results)
        
        self.open_folder_button = QPushButton("開啟資料夾")
        self.open_folder_button.clicked.connect(self.open_results_folder)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.open_folder_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 結果資料
        self.result_data = None
        self.result_folder = None
    
    def update_results(self, result_data, folder_path=None):
        self.result_data = result_data
        self.result_folder = folder_path
        
        if not result_data:
            return
            
        # 更新統計信息
        self.count_label.setText(str(len(result_data['collected_net_speeds'])))
        self.avg_speed_label.setText(f"{result_data['average_speed']:.1f} km/h")
        self.max_speed_label.setText(f"{result_data['max_speed']:.1f} km/h")
        self.min_speed_label.setText(f"{result_data['min_speed']:.1f} km/h")
        
        # 顯示圖表
        if 'chart' in result_data and os.path.exists(result_data['chart']):
            pixmap = QPixmap(result_data['chart'])
            self.chart_preview.setPixmap(pixmap.scaled(
                self.chart_preview.width(), 
                self.chart_preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        
        # 顯示詳細數據
        if 'txt' in result_data and os.path.exists(result_data['txt']):
            try:
                with open(result_data['txt'], 'r') as f:
                    self.details_text.setText(f.read())
            except:
                self.details_text.setText("無法讀取詳細結果數據")
    
    def save_results(self):
        if not self.result_data:
            QMessageBox.warning(self, "警告", "沒有可保存的結果")
            return
        
        # 實作另存功能
        save_folder = QFileDialog.getExistingDirectory(self, "選擇保存目錄")
        if not save_folder:
            return
            
        try:
            # 複製圖表、CSV和TXT文件
            import shutil
            for key in ['chart', 'csv', 'txt']:
                if key in self.result_data and os.path.exists(self.result_data[key]):
                    filename = os.path.basename(self.result_data[key])
                    dest_path = os.path.join(save_folder, filename)
                    shutil.copy2(self.result_data[key], dest_path)
                    
            QMessageBox.information(self, "成功", f"結果已保存到: {save_folder}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"保存結果時發生錯誤: {str(e)}")
    
    def open_results_folder(self):
        if not self.result_folder or not os.path.exists(self.result_folder):
            QMessageBox.warning(self, "警告", "結果資料夾不存在")
            return
            
        # 使用系統默認方式打開文件夾
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(self.result_folder)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", self.result_folder])
        else:  # Linux
            subprocess.Popen(["xdg-open", self.result_folder])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("乒乓球速度追蹤器")
        self.setMinimumSize(1200, 800)
        self.initUI()
        
        # 狀態變數
        self.video_recording_active = False
        self.analysis_active = False
        self.current_video_path = None
        self.current_output_folder = None
        
    def initUI(self):
        # 主視窗布局
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 創建標籤頁
        self.tabs = QTabWidget()
        self.recording_tab = QWidget()
        self.analysis_tab = QWidget()
        self.advanced_tab = QWidget()
        
        self.tabs.addTab(self.recording_tab, "錄影")
        self.tabs.addTab(self.analysis_tab, "分析")
        self.tabs.addTab(self.advanced_tab, "進階設定")
        
        main_layout.addWidget(self.tabs)
        
        # 設置錄影標籤頁
        self.setup_recording_tab()
        
        # 設置分析標籤頁
        self.setup_analysis_tab()
        
        # 設置進階設定標籤頁
        self.setup_advanced_tab()
        
        # 底部狀態欄
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就緒")
        
    def setup_recording_tab(self):
        layout = QVBoxLayout()
        
        # 上半部分：影片預覽和控制
        top_layout = QHBoxLayout()
        
        # 影片預覽
        self.video_preview = QLabel("相機預覽")
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.video_preview.setMinimumSize(640, 360)
        self.video_preview.setStyleSheet("background-color: black;")
        
        # 控制面板
        control_panel = QGroupBox("錄影控制")
        control_layout = QVBoxLayout()
        
        # 用戶名稱輸入
        name_layout = QFormLayout()
        self.user_name_input = QLineEdit("User")
        name_layout.addRow("使用者姓名:", self.user_name_input)
        control_layout.addLayout(name_layout)
        
        # 相機設定
        camera_group = QGroupBox("相機設定")
        camera_layout = QFormLayout()
        
        self.camera_select = QComboBox()
        self.camera_select.addItem("預設相機", DEFAULT_CAMERA_INDEX)
        # 嘗試查找可用攝像頭
        self.refreshCameraList()
        camera_layout.addRow("選擇相機:", self.camera_select)
        
        # 添加刷新相機列表按鈕
        refresh_btn = QPushButton("刷新相機列表")
        refresh_btn.clicked.connect(self.refreshCameraList)
        camera_layout.addRow("", refresh_btn)
        
        self.resolution_select = QComboBox()
        self.resolution_select.addItem("HD (1280x720)", (1280, 720))
        self.resolution_select.addItem("Full HD (1920x1080)", (1920, 1080))
        self.resolution_select.addItem("4K UHD (3840x2160)", (3840, 2160))
        self.resolution_select.addItem("Low Res (640x480)", (640, 480))
        camera_layout.addRow("解析度:", self.resolution_select)
        
        self.fps_select = QSpinBox()
        self.fps_select.setRange(30, 240)
        self.fps_select.setValue(DEFAULT_TARGET_FPS)
        camera_layout.addRow("目標 FPS:", self.fps_select)
        
        camera_group.setLayout(camera_layout)
        control_layout.addWidget(camera_group)
        
        # 顯示實際 FPS
        self.actual_fps_label = QLabel("0.0 FPS")
        control_layout.addWidget(self.actual_fps_label)
        
        # 操作按鈕
        button_layout = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("啟動相機")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        
        self.record_btn = QPushButton("開始錄影")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        
        button_layout.addWidget(self.start_camera_btn)
        button_layout.addWidget(self.record_btn)
        
        control_layout.addLayout(button_layout)
        
        # 錄影狀態
        self.recording_status_label = QLabel("未錄影")
        control_layout.addWidget(self.recording_status_label)
        
        # 分析按鈕（錄影後啟用）
        self.analyze_recorded_btn = QPushButton("分析已錄製的影片")
        self.analyze_recorded_btn.setEnabled(False)
        self.analyze_recorded_btn.clicked.connect(self.analyze_current_video)
        control_layout.addWidget(self.analyze_recorded_btn)
        
        control_panel.setLayout(control_layout)
        
        top_layout.addWidget(self.video_preview, 2)
        top_layout.addWidget(control_panel, 1)
        
        layout.addLayout(top_layout)
        
        # 底部說明
        instruction_group = QGroupBox("操作說明")
        instruction_layout = QVBoxLayout()
        
        instructions = """
        1. 輸入使用者姓名（用於檔案命名）
        2. 選擇相機和影片設定
        3. 點擊「啟動相機」開始預覽
        4. 按「開始錄影」記錄影片
        5. 錄製完成後可直接進行分析
        
        提示：將相機放置在乒乓球台側面，確保畫面中能完整看到球的軌跡和網子位置。
        可在「進階設定」標籤頁中調整更多參數。
        """
        
        instruction_label = QLabel(instructions)
        instruction_layout.addWidget(instruction_label)
        
        instruction_group.setLayout(instruction_layout)
        layout.addWidget(instruction_group)
        
        self.recording_tab.setLayout(layout)
        
        # 初始化影片執行緒
        self.video_thread = VideoThread()
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.update_fps.connect(self.update_fps)
        self.video_thread.recording_status.connect(self.update_recording_status)
        self.video_thread.error_occurred.connect(self.show_error)
        
    def setup_analysis_tab(self):
        layout = QVBoxLayout()
        
        # 分割分析區和結果區
        splitter = QSplitter(Qt.Horizontal)
        
        # 左側：分析控制和影片預覽
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout()
        analysis_widget.setLayout(analysis_layout)
        
        # 影片載入控制
        load_group = QGroupBox("載入影片")
        load_layout = QHBoxLayout()
        
        self.video_path_input = QLineEdit()
        self.video_path_input.setReadOnly(True)
        self.browse_video_btn = QPushButton("瀏覽...")
        self.browse_video_btn.clicked.connect(self.browse_video)
        
        load_layout.addWidget(self.video_path_input, 3)
        load_layout.addWidget(self.browse_video_btn, 1)
        
        load_group.setLayout(load_layout)
        analysis_layout.addWidget(load_group)
        
        # 分析設定
        settings_group = QGroupBox("分析設定")
        settings_layout = QFormLayout()
        
        self.table_length_input = QDoubleSpinBox()
        self.table_length_input.setRange(100, 300)
        self.table_length_input.setValue(DEFAULT_TABLE_LENGTH_CM)
        self.table_length_input.setSingleStep(1)
        self.table_length_input.setSuffix(" cm")
        settings_layout.addRow("球台長度:", self.table_length_input)
        
        self.crossing_direction_select = QComboBox()
        self.crossing_direction_select.addItem("從左到右", "left_to_right")
        self.crossing_direction_select.addItem("從右到左", "right_to_left")
        self.crossing_direction_select.addItem("雙向", "both")
        settings_layout.addRow("球過網方向:", self.crossing_direction_select)
        
        self.near_width_input = QDoubleSpinBox()
        self.near_width_input.setRange(10, 100)
        self.near_width_input.setValue(NEAR_SIDE_WIDTH_CM_DEFAULT)
        self.near_width_input.setSingleStep(1)
        self.near_width_input.setSuffix(" cm")
        settings_layout.addRow("近端寬度:", self.near_width_input)
        
        self.far_width_input = QDoubleSpinBox()
        self.far_width_input.setRange(10, 100)
        self.far_width_input.setValue(FAR_SIDE_WIDTH_CM_DEFAULT)
        self.far_width_input.setSingleStep(1)
        self.far_width_input.setSuffix(" cm")
        settings_layout.addRow("遠端寬度:", self.far_width_input)
        
        self.debug_mode_checkbox = QCheckBox()
        self.debug_mode_checkbox.setChecked(False)
        settings_layout.addRow("除錯模式:", self.debug_mode_checkbox)
        
        settings_group.setLayout(settings_layout)
        analysis_layout.addWidget(settings_group)
        
        # 播放控制
        playback_group = QGroupBox("播放控制")
        playback_layout = QHBoxLayout()
        
        self.playback_speed_select = QComboBox()
        self.playback_speed_select.addItem("0.25x", 0.25)
        self.playback_speed_select.addItem("0.5x", 0.5)
        self.playback_speed_select.addItem("1.0x", 1.0)
        self.playback_speed_select.addItem("2.0x", 2.0)
        self.playback_speed_select.addItem("4.0x", 4.0)
        self.playback_speed_select.setCurrentIndex(2)  # 預設 1.0x
        playback_layout.addWidget(QLabel("播放速度:"))
        playback_layout.addWidget(self.playback_speed_select)
        
        self.playback_speed_select.currentIndexChanged.connect(self.change_playback_speed)
        
        playback_group.setLayout(playback_layout)
        analysis_layout.addWidget(playback_group)
        
        # 分析控制按鈕
        control_layout = QHBoxLayout()
        
        self.start_analysis_btn = QPushButton("開始分析")
        self.start_analysis_btn.setEnabled(False)
        self.start_analysis_btn.clicked.connect(self.toggle_analysis)
        
        self.stop_analysis_btn = QPushButton("停止分析")
        self.stop_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        
        control_layout.addWidget(self.start_analysis_btn)
        control_layout.addWidget(self.stop_analysis_btn)
        
        analysis_layout.addLayout(control_layout)
        
        # 分析狀態
        status_layout = QHBoxLayout()
        self.analysis_status_label = QLabel("未開始分析")
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        
        status_layout.addWidget(self.analysis_status_label, 1)
        status_layout.addWidget(self.analysis_progress, 2)
        
        analysis_layout.addLayout(status_layout)
        
        # 影片預覽
        self.analysis_preview = QLabel("分析預覽")
        self.analysis_preview.setAlignment(Qt.AlignCenter)
        self.analysis_preview.setMinimumSize(640, 360)
        self.analysis_preview.setStyleSheet("background-color: black;")
        
        analysis_layout.addWidget(self.analysis_preview)
        
        # 右側：結果顯示
        self.results_viewer = ResultsViewer()
        
        # 添加到分割器
        splitter.addWidget(analysis_widget)
        splitter.addWidget(self.results_viewer)
        
        layout.addWidget(splitter)
        
        self.analysis_tab.setLayout(layout)
        
        # 初始化分析執行緒
        self.analysis_thread = AnalysisThread()
        self.analysis_thread.update_frame.connect(self.update_analysis_frame)
        self.analysis_thread.update_status.connect(self.update_analysis_status)
        self.analysis_thread.analysis_complete.connect(self.show_analysis_results)
        self.analysis_thread.error_occurred.connect(self.show_error)
        self.analysis_thread.progress_update.connect(self.update_analysis_progress)
    
    def setup_advanced_tab(self):
        """設置進階設定標籤頁"""
        layout = QVBoxLayout()
        
        # 進階設定說明
        description_label = QLabel("""
        下方提供更詳細的參數控制，適合進階使用者調整。
        請注意，不恰當的設定可能會影響分析準確度。
        若不確定參數含義，建議保留預設值。
        """)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        
        # 進階設定元件
        self.advanced_settings = AdvancedSettingsWidget()
        layout.addWidget(self.advanced_settings)
        
        # 添加配置檔案存取功能
        config_group = QGroupBox("配置檔管理")
        config_layout = QHBoxLayout()
        
        self.save_config_btn = QPushButton("保存當前配置")
        self.save_config_btn.clicked.connect(self.save_configuration)
        
        self.load_config_btn = QPushButton("載入配置檔")
        self.load_config_btn.clicked.connect(self.load_configuration)
        
        config_layout.addWidget(self.save_config_btn)
        config_layout.addWidget(self.load_config_btn)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        self.advanced_tab.setLayout(layout)
        
    def refreshCameraList(self):
        """刷新相機列表"""
        current_index = self.camera_select.currentIndex()
        self.camera_select.clear()
        self.camera_select.addItem("預設相機", DEFAULT_CAMERA_INDEX)
        
        # 嘗試尋找可用的相機
        for i in range(10):  # 檢查前10個索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                self.camera_select.addItem(f"相機 {i}", i)
        
        # 嘗試恢復之前的選擇
        if current_index >= 0 and current_index < self.camera_select.count():
            self.camera_select.setCurrentIndex(current_index)
    
    def update_frame(self, frame):
        """更新相機預覽"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.video_preview.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_preview.width(), 
            self.video_preview.height(),
            Qt.KeepAspectRatio
        ))
    
    def update_analysis_frame(self, frame):
        """更新分析預覽"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.analysis_preview.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.analysis_preview.width(), 
            self.analysis_preview.height(),
            Qt.KeepAspectRatio
        ))
    
    def update_fps(self, fps):
        """更新顯示的 FPS"""
        self.actual_fps_label.setText(f"{fps:.1f} FPS")
    
    def update_recording_status(self, is_recording):
        """更新錄影狀態"""
        if is_recording:
            self.recording_status_label.setText("● 正在錄影")
            self.recording_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.record_btn.setText("停止錄影")
            self.video_recording_active = True
        else:
            self.recording_status_label.setText("■ 錄影已停止")
            self.recording_status_label.setStyleSheet("color: black;")
            self.record_btn.setText("開始錄影")
            self.video_recording_active = False
            self.analyze_recorded_btn.setEnabled(True)
    
    def update_analysis_status(self, status):
        """更新分析狀態"""
        self.analysis_status_label.setText(status)
    
    def update_analysis_progress(self, value):
        """更新分析進度"""
        self.analysis_progress.setValue(value)
    
    def toggle_camera(self):
        """開始/停止相機預覽"""
        if not self.video_thread.running:
            # 重新配置影片執行緒
            camera_index = self.camera_select.currentData()
            target_fps = self.fps_select.value()
            resolution = self.resolution_select.currentData()
            user_name = self.user_name_input.text()
            
            self.video_thread = VideoThread(camera_index, target_fps, resolution)
            self.video_thread.set_user_name(user_name)
            self.video_thread.update_frame.connect(self.update_frame)
            self.video_thread.update_fps.connect(self.update_fps)
            self.video_thread.recording_status.connect(self.update_recording_status)
            self.video_thread.error_occurred.connect(self.show_error)
            
            self.video_thread.start()
            self.start_camera_btn.setText("停止相機")
            self.record_btn.setEnabled(True)
        else:
            if self.video_recording_active:
                self.toggle_recording()  # 停止進行中的錄影
                
            self.video_thread.stop()
            self.start_camera_btn.setText("啟動相機")
            self.record_btn.setEnabled(False)
    
    def toggle_recording(self):
        """開始/停止錄影"""
        if not self.video_recording_active:
            # 開始錄影
            if self.video_thread.start_recording():
                self.status_bar.showMessage("開始錄影")
                self.current_output_folder = self.video_thread.output_folder
            else:
                self.status_bar.showMessage("錄影啟動失敗")
        else:
            # 停止錄影
            video_path = self.video_thread.stop_recording()
            if video_path:
                self.status_bar.showMessage(f"錄影已保存: {video_path}")
                self.current_video_path = video_path
            else:
                self.status_bar.showMessage("錄影停止失敗")
    
    def analyze_current_video(self):
        """分析當前錄製的影片"""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            self.show_error("沒有可分析的影片")
            return
            
        # 切換到分析標籤頁
        self.tabs.setCurrentIndex(1)
        
        # 填充影片路徑
        self.video_path_input.setText(self.current_video_path)
        
        # 準備分析
        self.prepare_analysis()
    
    def browse_video(self):
        """瀏覽並選擇影片檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片檔案", "", "影片檔案 (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            self.video_path_input.setText(file_path)
            self.current_video_path = file_path
            self.current_output_folder = os.path.dirname(file_path)
            self.start_analysis_btn.setEnabled(True)
    
    def prepare_analysis(self):
        """準備分析"""
        # 收集基本分析參數
        self.analysis_thread.set_parameters(
            video_path=self.current_video_path,
            output_folder=self.current_output_folder,
            output_basename=self.user_name_input.text(),
            table_length_cm=self.table_length_input.value(),
            net_crossing_direction=self.crossing_direction_select.currentData(),
            near_width_cm=self.near_width_input.value(),
            far_width_cm=self.far_width_input.value(),
            debug_mode=self.debug_mode_checkbox.isChecked()
        )
        
        # 收集進階參數
        advanced_params = self.advanced_settings.getParameters()
        self.analysis_thread.set_advanced_parameters(advanced_params)
        
        # 設置播放速度
        self.change_playback_speed()
        
        # 啟動分析執行緒
        self.analysis_thread.start()
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(True)
        
    def toggle_analysis(self):
        """開始/暫停分析"""
        if not self.analysis_thread.isRunning():
            self.prepare_analysis()
        
        self.analysis_thread.toggle_analysis()
        
        if self.analysis_thread.start_analysis:
            self.start_analysis_btn.setText("暫停分析")
        else:
            self.start_analysis_btn.setText("繼續分析")
    
    def stop_analysis(self):
        """停止分析"""
        if self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.start_analysis_btn.setText("開始分析")
            self.start_analysis_btn.setEnabled(True)
            self.stop_analysis_btn.setEnabled(False)
            self.status_bar.showMessage("分析已停止")
    
    def change_playback_speed(self):
        """變更播放速度"""
        if self.analysis_thread.isRunning():
            speed = self.playback_speed_select.currentData()
            self.analysis_thread.set_playback_speed(speed)
    
    def show_analysis_results(self, result_data):
        """顯示分析結果"""
        self.results_viewer.update_results(result_data, self.current_output_folder)
        self.status_bar.showMessage("分析完成")
    
    def save_configuration(self):
        """保存當前設定到配置檔"""
        config_file, _ = QFileDialog.getSaveFileName(
            self, "保存配置檔", "", "配置檔 (*.json)"
        )
        
        if not config_file:
            return
            
        try:
            import json
            config = {
                # 基本設定
                "table_length_cm": self.table_length_input.value(),
                "net_crossing_direction": self.crossing_direction_select.currentData(),
                "near_width_cm": self.near_width_input.value(),
                "far_width_cm": self.far_width_input.value(),
                "debug_mode": self.debug_mode_checkbox.isChecked(),
                # 進階設定
                "advanced": self.advanced_settings.getParameters()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            self.status_bar.showMessage(f"配置已保存: {config_file}")
            QMessageBox.information(self, "成功", "配置已成功保存")
        except Exception as e:
            self.show_error(f"保存配置時發生錯誤: {str(e)}")
    
    def load_configuration(self):
        """從配置檔載入設定"""
        config_file, _ = QFileDialog.getOpenFileName(
            self, "載入配置檔", "", "配置檔 (*.json)"
        )
        
        if not config_file:
            return
            
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # 套用基本設定
            if "table_length_cm" in config:
                self.table_length_input.setValue(config["table_length_cm"])
                
            if "net_crossing_direction" in config:
                index = self.crossing_direction_select.findData(config["net_crossing_direction"])
                if index >= 0:
                    self.crossing_direction_select.setCurrentIndex(index)
                    
            if "near_width_cm" in config:
                self.near_width_input.setValue(config["near_width_cm"])
                
            if "far_width_cm" in config:
                self.far_width_input.setValue(config["far_width_cm"])
                
            if "debug_mode" in config:
                self.debug_mode_checkbox.setChecked(config["debug_mode"])
            
            # 暫時不處理進階設定，因為目前還未實作 AdvancedSettingsWidget.setParameters 方法
            # 這可以在後續版本中添加
                
            self.status_bar.showMessage(f"配置已載入: {config_file}")
            QMessageBox.information(self, "成功", "配置已成功載入")
        except Exception as e:
            self.show_error(f"載入配置時發生錯誤: {str(e)}")
        
    def show_error(self, error_message):
        """顯示錯誤訊息"""
        QMessageBox.critical(self, "錯誤", error_message)
        self.status_bar.showMessage(f"錯誤: {error_message}")
    
    def closeEvent(self, event):
        """視窗關閉事件處理"""
        # 停止所有執行緒
        if self.video_thread.isRunning():
            self.video_thread.stop()
        
        if self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 設置應用程式風格
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())