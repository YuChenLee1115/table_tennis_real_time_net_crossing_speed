"""
配置管理模組
負責管理所有系統參數和設置
"""

from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class CameraConfig:
    """攝像頭配置"""
    default_index: int = 0
    target_fps: int = 60
    frame_width: int = 1280
    frame_height: int = 720


@dataclass
class DetectionConfig:
    """偵測配置"""
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
    """追蹤配置"""
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
    """視覺化配置"""
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
    """從文件加載配置（預留接口）"""
    # TODO: 實現從JSON/YAML文件加載配置
    return create_default_config()