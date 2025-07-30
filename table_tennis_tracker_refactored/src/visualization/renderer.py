"""
視覺化渲染器
負責在影像上繪製各種視覺元素
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..core.config import VisualizationConfig
from ..core.events import FrameData


class Renderer:
    """視覺化渲染器"""
    
    def __init__(self, config: VisualizationConfig, frame_width: int, frame_height: int,
                 roi_start_x: int, roi_end_x: int, roi_top_y: int, roi_bottom_y: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_start_x = roi_start_x
        self.roi_end_x = roi_end_x
        self.roi_top_y = roi_top_y
        self.roi_bottom_y = roi_bottom_y
        self.center_x = frame_width // 2
        
        # 預計算靜態覆蓋層
        self.static_overlay = self._create_static_overlay()
        
        # 界面文字
        self.instruction_text = "SPACE: Toggle Count | D: Debug | Q/ESC: Quit"
    
    def _create_static_overlay(self) -> np.ndarray:
        """創建靜態覆蓋層（ROI框和中線）"""
        overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # ROI框線
        cv2.line(overlay, (self.roi_start_x, self.roi_top_y), 
                (self.roi_start_x, self.roi_bottom_y), self.config.roi_color_bgr, 2)
        cv2.line(overlay, (self.roi_end_x, self.roi_top_y), 
                (self.roi_end_x, self.roi_bottom_y), self.config.roi_color_bgr, 2)
        cv2.line(overlay, (self.roi_start_x, self.roi_bottom_y), 
                (self.roi_end_x, self.roi_bottom_y), self.config.roi_color_bgr, 2)
        
        # 中線
        cv2.line(overlay, (self.center_x, 0), (self.center_x, self.frame_height), 
                self.config.center_line_color_bgr, 2)
        
        return overlay
    
    def render_frame(self, frame_data: FrameData, max_net_speeds: int, 
                    debug_mode: bool = False) -> np.ndarray:
        """渲染完整幀"""
        if frame_data.frame is None:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
        display_frame = frame_data.frame.copy()
        
        # 是否進行完整繪製（降低頻率以提高性能）
        is_full_draw = frame_data.frame_counter % self.config.draw_interval == 0
        
        if is_full_draw:
            # 添加靜態覆蓋層
            display_frame = cv2.addWeighted(
                display_frame, 1.0, self.static_overlay, 0.7, 0
            )
            
            # 繪製軌跡
            self._draw_trajectory(display_frame, frame_data.trajectory_points_global)
        
        # 繪製球體
        self._draw_ball(display_frame, frame_data)
        
        # 繪製文字信息
        self._draw_text_info(display_frame, frame_data, max_net_speeds, debug_mode)
        
        return display_frame
    
    def _draw_trajectory(self, frame: np.ndarray, trajectory_points: List[Tuple[int, int]]) -> None:
        """繪製球體軌跡"""
        if len(trajectory_points) >= 2:
            pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, 
                         color=self.config.trajectory_color_bgr, thickness=2)
    
    def _draw_ball(self, frame: np.ndarray, frame_data: FrameData) -> None:
        """繪製球體"""
        if frame_data.ball_position_in_roi and frame_data.roi_sub_frame is not None:
            cx_roi, cy_roi = frame_data.ball_position_in_roi
            
            # 在ROI子幀上繪製
            cv2.circle(frame_data.roi_sub_frame, (cx_roi, cy_roi), 5, 
                      self.config.ball_color_bgr, -1)
            
            # 繪製輪廓
            if frame_data.ball_contour_in_roi is not None:
                cv2.drawContours(frame_data.roi_sub_frame, 
                               [frame_data.ball_contour_in_roi], 0, 
                               self.config.contour_color_bgr, 2)
            
            # 在全幀上繪製
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            cv2.circle(frame, (cx_global, cy_global), 8, 
                      self.config.ball_color_bgr, -1)
    
    def _draw_text_info(self, frame: np.ndarray, frame_data: FrameData, 
                       max_net_speeds: int, debug_mode: bool) -> None:
        """繪製文字信息"""
        # 速度信息
        cv2.putText(frame, f"Speed: {frame_data.current_ball_speed_kmh:.1f} km/h", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                   self.config.speed_text_color_bgr, self.config.font_thickness)
        
        # FPS信息
        cv2.putText(frame, f"FPS: {frame_data.display_fps:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                   self.config.fps_text_color_bgr, self.config.font_thickness)
        
        # 計數狀態
        count_status = "ON" if frame_data.is_counting_active else "OFF"
        count_color = (0, 255, 0) if frame_data.is_counting_active else (0, 0, 255)
        cv2.putText(frame, f"Counting: {count_status}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                   count_color, self.config.font_thickness)
        
        # 最後記錄的速度
        if frame_data.last_recorded_net_speed_kmh > 0:
            cv2.putText(frame, f"Last Net: {frame_data.last_recorded_net_speed_kmh:.1f} km/h", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                       self.config.net_speed_text_color_bgr, self.config.font_thickness)
        
        # 記錄數量
        cv2.putText(frame, f"Recorded: {len(frame_data.collected_net_speeds)}/{max_net_speeds}", 
                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                   self.config.net_speed_text_color_bgr, self.config.font_thickness)
        
        # 最後時間
        if frame_data.collected_relative_times:
            cv2.putText(frame, f"Last Time: {frame_data.collected_relative_times[-1]:.2f}s", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                       self.config.net_speed_text_color_bgr, self.config.font_thickness)
        
        # 操作說明
        cv2.putText(frame, self.instruction_text, (10, self.frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 除錯信息
        if debug_mode and frame_data.debug_display_text:
            cv2.putText(frame, frame_data.debug_display_text, (10, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)