"""
配置管理模組
負責管理所有系統參數和設置
"""

from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class CameraConfig:
    """攝像頭配置類別
    
    包含攝像頭和視頻捕獲相關的所有參數設定。
    支援多種攝像頭設備和不同解析度設定。
    
    Attributes:
        default_index (int): 預設攝像頭索引，通常0為內建攝像頭
        target_fps (int): 目標幀率，影響檢測精度和系統性能
        frame_width (int): 影像寬度（像素），建議1280以獲得最佳檢測效果
        frame_height (int): 影像高度（像素），建議720以獲得最佳檢測效果
    """
    default_index: int = 0
    target_fps: int = 60
    frame_width: int = 1280
    frame_height: int = 720


@dataclass
class DetectionConfig:
    """偵測配置類別
    
    包含球體檢測和快速移動物體(FMO)檢測的所有參數。
    這些參數直接影響檢測精度、性能和誤報率。
    
    Attributes:
        timeout_s (float): 球體檢測超時時間，超過此時間將重置軌跡
        roi_start_ratio (float): ROI起始位置比例（0-1），相對於影像寬度
        roi_end_ratio (float): ROI結束位置比例（0-1），相對於影像寬度
        roi_bottom_ratio (float): ROI底部位置比例（0-1），相對於影像高度
        max_trajectory_points (int): 軌跡點最大數量，影響記憶體使用和平滑度
        
        FMO檢測參數:
        max_prev_frames_fmo (int): 用於多幀差分的最大前序幀數
        opening_kernel_size_fmo (Tuple[int, int]): 開運算核心大小，用於去除雜訊
        closing_kernel_size_fmo (Tuple[int, int]): 閉運算核心大小，用於連接斷點
        threshold_value_fmo (int): FMO檢測閾值，較低值更敏感但誤報較多
        
        球體特徵參數:
        min_ball_area_px (int): 最小球體面積（像素），過小可能誤判雜訊
        max_ball_area_px (int): 最大球體面積（像素），過大可能誤判其他物體
        min_ball_circularity (float): 最小圓度要求（0-1），1為完美圓形
    """
    timeout_s: float = 0.2
    roi_start_ratio: float = 0.4
    roi_end_ratio: float = 0.6
    roi_bottom_ratio: float = 0.85
    max_trajectory_points: int = 120
    
    # FMO參數
    max_prev_frames_fmo: int = 10
    opening_kernel_size_fmo: Tuple[int, int] = (12, 12)
    closing_kernel_size_fmo: Tuple[int, int] = (25, 25)
    threshold_value_fmo: int = 9
    
    # 球體偵測參數
    min_ball_area_px: int = 10
    max_ball_area_px: int = 10000
    min_ball_circularity: float = 0.32


@dataclass
class TrackingConfig:
    """追蹤配置類別
    
    包含速度計算、網線穿越檢測和透視校正的所有參數。
    這些參數直接影響速度測量精度和穿越事件檢測。
    
    Attributes:
        透視校正參數:
        table_length_cm (float): 桌子長度（公分），用於計算像素/公分比例
        near_side_width_cm (float): 近端寬度（公分），用於透視校正
        far_side_width_cm (float): 遠端寬度（公分），用於透視校正
        
        速度計算參數:
        speed_smoothing_factor (float): 速度平滑係數（0-1），較高值更平滑但響應較慢
        kmh_conversion_factor (float): 公分/秒轉公里/時的轉換係數
        
        網線穿越參數:
        max_net_speeds_to_collect (int): 每次收集的網線穿越速度數量
        net_crossing_direction (str): 網線穿越方向 ('right_to_left', 'left_to_right', 'both')
        auto_stop_after_collection (bool): 收集完成後是否自動停止計數
        crossing_cooldown_s (float): 穿越事件冷卻時間（秒），防止重複記錄
        center_zone_width_ratio (float): 中線區域寬度比例，用於穿越檢測
    """
    table_length_cm: float = 70.0
    near_side_width_cm: float = 29.0
    far_side_width_cm: float = 72.0
    
    # 速度計算
    speed_smoothing_factor: float = 0.3
    kmh_conversion_factor: float = 0.036
    
    # 中線穿越
    max_net_speeds_to_collect: int = 30
    net_crossing_direction: str = 'right_to_left'
    auto_stop_after_collection: bool = False
    crossing_cooldown_s: float = 0.2
    center_zone_width_ratio: float = 0.05


@dataclass
class VisualizationConfig:
    """視覺化配置類別
    
    包含在影像上繪製各種視覺元素的所有參數設定。
    使用BGR色彩格式，符合OpenCV的慣例。
    
    Attributes:
        色彩設定 (BGR格式):
        trajectory_color_bgr (Tuple[int, int, int]): 軌跡線色彩
        ball_color_bgr (Tuple[int, int, int]): 球體標記色彩
        contour_color_bgr (Tuple[int, int, int]): 輪廓線色彩
        roi_color_bgr (Tuple[int, int, int]): ROI框線色彩
        speed_text_color_bgr (Tuple[int, int, int]): 速度文字色彩
        fps_text_color_bgr (Tuple[int, int, int]): FPS文字色彩
        center_line_color_bgr (Tuple[int, int, int]): 中線色彩
        net_speed_text_color_bgr (Tuple[int, int, int]): 網線速度文字色彩
        
        文字設定:
        font_scale (float): 字型大小縮放比例
        font_thickness (int): 字型粗細（像素）
        draw_interval (int): 繪製間隔（幀數），用於性能優化
    """
    trajectory_color_bgr: Tuple[int, int, int] = (0, 0, 255)
    ball_color_bgr: Tuple[int, int, int] = (0, 255, 255)
    contour_color_bgr: Tuple[int, int, int] = (255, 0, 0)
    roi_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    speed_text_color_bgr: Tuple[int, int, int] = (0, 0, 255)
    fps_text_color_bgr: Tuple[int, int, int] = (0, 255, 0)
    center_line_color_bgr: Tuple[int, int, int] = (0, 255, 255)
    net_speed_text_color_bgr: Tuple[int, int, int] = (255, 0, 0)
    font_scale: float = 1.0
    font_thickness: int = 2
    draw_interval: int = 2


@dataclass
class IOConfig:
    """輸入輸出配置"""
    output_data_folder: str = 'real_time_output'
    frame_queue_size: int = 30
    event_buffer_size: int = 200


@dataclass
class SystemConfig:
    """系統總配置"""
    debug_mode: bool = False
    fps_smoothing_factor: float = 0.4
    max_frame_times_fps_calc: int = 20
    
    def __init__(self):
        self.camera = CameraConfig()
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.visualization = VisualizationConfig()
        self.io = IOConfig()


def create_default_config() -> SystemConfig:
    """創建默認配置"""
    return SystemConfig()


def load_config_from_file(config_path: str) -> SystemConfig:
    """從JSON文件加載配置
    
    Args:
        config_path: JSON配置文件路徑
        
    Returns:
        載入配置後的SystemConfig對象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式錯誤
    """
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 創建基礎配置對象
        config = create_default_config()
        
        # 加載相機配置
        if 'camera' in config_data:
            camera_data = config_data['camera']
            config.camera.default_index = camera_data.get('default_index', config.camera.default_index)
            config.camera.target_fps = camera_data.get('target_fps', config.camera.target_fps)
            config.camera.frame_width = camera_data.get('frame_width', config.camera.frame_width)
            config.camera.frame_height = camera_data.get('frame_height', config.camera.frame_height)
        
        # 加載檢測配置
        if 'detection' in config_data:
            detection_data = config_data['detection']
            config.detection.timeout_s = detection_data.get('timeout_s', config.detection.timeout_s)
            config.detection.roi_start_ratio = detection_data.get('roi_start_ratio', config.detection.roi_start_ratio)
            config.detection.roi_end_ratio = detection_data.get('roi_end_ratio', config.detection.roi_end_ratio)
            config.detection.roi_bottom_ratio = detection_data.get('roi_bottom_ratio', config.detection.roi_bottom_ratio)
            config.detection.max_trajectory_points = detection_data.get('max_trajectory_points', config.detection.max_trajectory_points)
            config.detection.max_prev_frames_fmo = detection_data.get('max_prev_frames_fmo', config.detection.max_prev_frames_fmo)
            
            # 形態學核心大小（轉換為tuple）
            if 'opening_kernel_size_fmo' in detection_data:
                kernel_size = detection_data['opening_kernel_size_fmo']
                config.detection.opening_kernel_size_fmo = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
            if 'closing_kernel_size_fmo' in detection_data:
                kernel_size = detection_data['closing_kernel_size_fmo']
                config.detection.closing_kernel_size_fmo = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
            
            config.detection.threshold_value_fmo = detection_data.get('threshold_value_fmo', config.detection.threshold_value_fmo)
            config.detection.min_ball_area_px = detection_data.get('min_ball_area_px', config.detection.min_ball_area_px)
            config.detection.max_ball_area_px = detection_data.get('max_ball_area_px', config.detection.max_ball_area_px)
            config.detection.min_ball_circularity = detection_data.get('min_ball_circularity', config.detection.min_ball_circularity)
        
        # 加載追蹤配置
        if 'tracking' in config_data:
            tracking_data = config_data['tracking']
            config.tracking.table_length_cm = tracking_data.get('table_length_cm', config.tracking.table_length_cm)
            config.tracking.near_side_width_cm = tracking_data.get('near_side_width_cm', config.tracking.near_side_width_cm)
            config.tracking.far_side_width_cm = tracking_data.get('far_side_width_cm', config.tracking.far_side_width_cm)
            config.tracking.speed_smoothing_factor = tracking_data.get('speed_smoothing_factor', config.tracking.speed_smoothing_factor)
            config.tracking.kmh_conversion_factor = tracking_data.get('kmh_conversion_factor', config.tracking.kmh_conversion_factor)
            config.tracking.max_net_speeds_to_collect = tracking_data.get('max_net_speeds_to_collect', config.tracking.max_net_speeds_to_collect)
            config.tracking.net_crossing_direction = tracking_data.get('net_crossing_direction', config.tracking.net_crossing_direction)
            config.tracking.auto_stop_after_collection = tracking_data.get('auto_stop_after_collection', config.tracking.auto_stop_after_collection)
            config.tracking.crossing_cooldown_s = tracking_data.get('crossing_cooldown_s', config.tracking.crossing_cooldown_s)
            config.tracking.center_zone_width_ratio = tracking_data.get('center_zone_width_ratio', config.tracking.center_zone_width_ratio)
        
        # 加載視覺化配置
        if 'visualization' in config_data:
            viz_data = config_data['visualization']
            
            # 顏色配置（轉換為tuple）
            color_fields = ['trajectory_color_bgr', 'ball_color_bgr', 'contour_color_bgr', 
                          'roi_color_bgr', 'speed_text_color_bgr', 'fps_text_color_bgr',
                          'center_line_color_bgr', 'net_speed_text_color_bgr']
            
            for field in color_fields:
                if field in viz_data:
                    color = viz_data[field]
                    setattr(config.visualization, field, tuple(color) if isinstance(color, list) else color)
            
            config.visualization.font_scale = viz_data.get('font_scale', config.visualization.font_scale)
            config.visualization.font_thickness = viz_data.get('font_thickness', config.visualization.font_thickness)
            config.visualization.draw_interval = viz_data.get('draw_interval', config.visualization.draw_interval)
        
        # 加載IO配置
        if 'io' in config_data:
            io_data = config_data['io']
            config.io.output_data_folder = io_data.get('output_data_folder', config.io.output_data_folder)
            config.io.frame_queue_size = io_data.get('frame_queue_size', config.io.frame_queue_size)
            config.io.event_buffer_size = io_data.get('event_buffer_size', config.io.event_buffer_size)
        
        # 加載系統配置
        if 'system' in config_data:
            system_data = config_data['system']
            config.debug_mode = system_data.get('debug_mode', config.debug_mode)
            config.fps_smoothing_factor = system_data.get('fps_smoothing_factor', config.fps_smoothing_factor)
            config.max_frame_times_fps_calc = system_data.get('max_frame_times_fps_calc', config.max_frame_times_fps_calc)
        
        print(f"成功從 {config_path} 載入配置")
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件JSON格式錯誤: {e}")
    except Exception as e:
        raise ValueError(f"載入配置文件時發生錯誤: {e}")