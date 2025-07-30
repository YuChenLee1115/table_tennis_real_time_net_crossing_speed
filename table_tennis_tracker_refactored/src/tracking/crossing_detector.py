"""
中線穿越偵測器
偵測球體穿越桌子中線的事件
"""

from typing import List, Tuple, Optional
from ..core.config import TrackingConfig
from ..core.events import CrossingEvent, EventManager


class CrossingDetector:
    """中線穿越偵測器"""
    
    def __init__(self, config: TrackingConfig, frame_width: int, frame_height: int, 
                 display_fps: float, event_manager: EventManager):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.display_fps = display_fps
        self.event_manager = event_manager
        
        self.center_x = frame_width // 2
        self.center_zone_width = frame_width * config.center_zone_width_ratio
        
        # 狀態追蹤
        self.ball_on_left_of_center = False
        self.last_ball_x_global: Optional[int] = None
        
    def detect_crossing(self, ball_x_global: int, ball_y_global: int, 
                       current_timestamp: float, trajectory: List[Tuple[int, int, float]], 
                       current_speed_kmh: float) -> None:
        """偵測中線穿越"""
        if self.config.net_crossing_direction not in ['right_to_left', 'both']:
            self._update_last_position(ball_x_global)
            return
            
        # 檢查冷卻時間
        if self._is_in_cooldown(current_timestamp):
            self._update_last_position(ball_x_global)
            return
            
        # 偵測實際穿越
        if self._detect_actual_crossing(ball_x_global, current_speed_kmh, current_timestamp):
            return
            
        # 偵測預測穿越
        self._detect_predicted_crossing(
            ball_x_global, current_timestamp, trajectory, current_speed_kmh
        )
        
        # 更新球體狀態
        self._update_ball_state(ball_x_global)
        self._update_last_position(ball_x_global)
        
    def _is_in_cooldown(self, current_timestamp: float) -> bool:
        """檢查是否在冷卻期間"""
        return (current_timestamp - self.event_manager.last_committed_crossing_time < 
                self.config.crossing_cooldown_s)
    
    def _detect_actual_crossing(self, ball_x_global: int, current_speed_kmh: float, 
                               current_timestamp: float) -> bool:
        """偵測實際穿越事件"""
        if (self.last_ball_x_global is not None and 
            self.last_ball_x_global >= self.center_x and 
            ball_x_global < self.center_x and 
            not self.ball_on_left_of_center and 
            current_speed_kmh > 0.1):
            
            event = CrossingEvent(
                ball_x_global=ball_x_global,
                timestamp=current_timestamp,
                speed_kmh=current_speed_kmh,
                predicted=False
            )
            self.event_manager.add_crossing_event(event)
            return True
        return False
    
    def _detect_predicted_crossing(self, ball_x_global: int, current_timestamp: float,
                                  trajectory: List[Tuple[int, int, float]], 
                                  current_speed_kmh: float) -> None:
        """偵測預測穿越事件"""
        if (self.ball_on_left_of_center or len(trajectory) < 2 or 
            current_speed_kmh <= 0.1):
            return
            
        pt1_x, _, pt1_t = trajectory[-2]
        pt2_x, _, pt2_t = trajectory[-1]
        
        # 確保前一點在中線右側
        if pt1_x < self.center_x:
            return
            
        delta_t = pt2_t - pt1_t
        if delta_t <= 0:
            return
            
        # 計算速度向量
        vx_pixels_per_time = (pt2_x - pt1_x) / delta_t
        
        # 計算最小速度閾值
        fps = self.display_fps if self.display_fps > 1 else 60
        min_vx_threshold = -(self.frame_width * 0.005) * (delta_t / (1.0 / fps))
        
        if vx_pixels_per_time >= min_vx_threshold:
            return
            
        # 嘗試不同的預測幀數
        for lookahead_frames in [1, 2, 3]:
            time_to_predict = lookahead_frames / fps
            predicted_x = ball_x_global + vx_pixels_per_time * time_to_predict
            predicted_timestamp = current_timestamp + time_to_predict
            
            if predicted_x < self.center_x:
                # 檢查是否已有相近的預測事件
                if not self._has_similar_prediction(predicted_timestamp, fps):
                    event = CrossingEvent(
                        ball_x_global=predicted_x,
                        timestamp=predicted_timestamp,
                        speed_kmh=current_speed_kmh,
                        predicted=True
                    )
                    self.event_manager.add_crossing_event(event)
                break
    
    def _has_similar_prediction(self, predicted_timestamp: float, fps: float) -> bool:
        """檢查是否已有相似的預測事件"""
        time_threshold = 1.0 / fps
        for event in self.event_manager.crossing_events:
            if (event.predicted and 
                abs(event.timestamp - predicted_timestamp) < time_threshold):
                return True
        return False
    
    def _update_ball_state(self, ball_x_global: int) -> None:
        """更新球體位置狀態"""
        if ball_x_global < self.center_x - self.center_zone_width:
            self.ball_on_left_of_center = True
        elif ball_x_global > self.center_x + self.center_zone_width:
            self.ball_on_left_of_center = False
    
    def _update_last_position(self, ball_x_global: int) -> None:
        """更新最後位置"""
        self.last_ball_x_global = ball_x_global
    
    def reset_state(self) -> None:
        """重置狀態"""
        self.ball_on_left_of_center = False
        self.last_ball_x_global = None