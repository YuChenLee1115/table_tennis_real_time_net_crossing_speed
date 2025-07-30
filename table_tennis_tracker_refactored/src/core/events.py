"""
事件處理模組
負責記錄和處理各種事件
"""

from dataclasses import dataclass
from typing import List, Optional
from collections import deque
import time


@dataclass
class BallPosition:
    """球體位置數據"""
    x_global: int
    y_global: int
    x_roi: int
    y_roi: int
    timestamp: float
    
    
@dataclass
class BallDetectionEvent:
    """球體偵測事件"""
    position: BallPosition
    area: float
    circularity: float
    contour: Optional[object] = None


@dataclass
class CrossingEvent:
    """中線穿越事件"""
    ball_x_global: float
    timestamp: float
    speed_kmh: float
    predicted: bool = False
    processed: bool = False


@dataclass
class FrameData:
    """單幀數據包"""
    frame: Optional[object] = None
    roi_sub_frame: Optional[object] = None
    ball_position_in_roi: Optional[tuple] = None
    ball_contour_in_roi: Optional[object] = None
    current_ball_speed_kmh: float = 0.0
    display_fps: float = 0.0
    is_counting_active: bool = False
    collected_net_speeds: List[float] = None
    last_recorded_net_speed_kmh: float = 0.0
    collected_relative_times: List[float] = None
    debug_display_text: Optional[str] = None
    frame_counter: int = 0
    trajectory_points_global: List[tuple] = None
    
    def __post_init__(self):
        if self.collected_net_speeds is None:
            self.collected_net_speeds = []
        if self.collected_relative_times is None:
            self.collected_relative_times = []
        if self.trajectory_points_global is None:
            self.trajectory_points_global = []


class EventManager:
    """事件管理器"""
    
    def __init__(self, buffer_size: int = 200):
        self.crossing_events = deque(maxlen=buffer_size)
        self.last_committed_crossing_time = 0.0
        self.first_ball_crossing_timestamp: Optional[float] = None
        self.timing_started = False
        
    def add_crossing_event(self, event: CrossingEvent) -> None:
        """添加穿越事件"""
        self.crossing_events.append(event)
    
    def get_pending_events(self) -> List[CrossingEvent]:
        """獲取待處理事件"""
        return [event for event in self.crossing_events if not event.processed]
    
    def process_events(self, current_time: float, cooldown_s: float) -> Optional[CrossingEvent]:
        """處理事件並返回可提交的事件"""
        pending = self.get_pending_events()
        if not pending:
            return None
            
        # 優先處理實際穿越事件
        for event in sorted(pending, key=lambda e: e.timestamp):
            if not event.predicted:
                if event.timestamp - self.last_committed_crossing_time >= cooldown_s:
                    event.processed = True
                    self.last_committed_crossing_time = event.timestamp
                    return event
        
        # 若無實際事件，考慮預測事件
        for event in sorted(pending, key=lambda e: e.timestamp):
            if event.predicted and current_time >= event.timestamp:
                if event.timestamp - self.last_committed_crossing_time >= cooldown_s:
                    event.processed = True
                    self.last_committed_crossing_time = event.timestamp
                    return event
        
        return None
    
    def cleanup_old_events(self, current_time: float, max_age_s: float = 2.0) -> None:
        """清理過舊的事件"""
        new_deque = deque(maxlen=self.crossing_events.maxlen)
        for event in self.crossing_events:
            if current_time - event.timestamp < max_age_s:
                new_deque.append(event)
        self.crossing_events = new_deque
    
    def reset_session(self) -> None:
        """重置會話"""
        self.crossing_events.clear()
        self.last_committed_crossing_time = 0.0
        self.first_ball_crossing_timestamp = None
        self.timing_started = False