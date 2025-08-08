"""
軌跡管理器
管理球體軌跡的連續性，處理軌跡中斷和恢復
包含預測機制以減少球體遺漏
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class TrajectoryPoint:
    """軌跡點 - 包含完整的狀態信息"""
    x: float                    # 全局座標 X
    y: float                    # 全局座標 Y
    timestamp: float           # 時間戳
    velocity_x: float = 0.0    # X方向速度 (像素/秒)
    velocity_y: float = 0.0    # Y方向速度 (像素/秒)
    confidence: float = 1.0    # 檢測信心度 (0-1)
    predicted: bool = False    # 是否為預測點
    interpolated: bool = False # 是否為插值點
    detection_id: int = 0      # 檢測ID，用於追蹤
    metadata: Dict[str, Any] = None  # 額外元數據
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TrajectoryManager:
    """軌跡管理器 - 處理球體軌跡的連續性和預測
    
    主要功能:
    - 軌跡平滑和插值
    - 短期遺漏預測
    - 軌跡品質評估
    - 異常點檢測和修正
    - 軌跡分割和重組
    """
    
    def __init__(self, max_points: int = 200):
        self.max_points = max_points
        
        # 主軌跡存儲
        self.trajectory = deque(maxlen=max_points)
        self.smoothed_trajectory = deque(maxlen=max_points)
        
        # 軌跡狀態
        self.current_detection_id = 0
        self.last_detection_time = 0.0
        self.trajectory_active = False
        self.missing_detection_count = 0
        
        # 預測參數
        self.max_missing_detections = 5  # 最大允許連續遺漏檢測數
        self.prediction_horizon_s = 0.1   # 預測時間範圍（秒）
        self.min_points_for_prediction = 3  # 預測所需的最小軌跡點數
        
        # 軌跡品質參數
        self.quality_window_size = 10
        self.smoothing_window_size = 5
        self.outlier_threshold_factor = 2.5
        
        # 插值參數
        self.interpolation_max_gap_s = 0.05  # 最大插值時間間隔
        self.interpolation_enabled = True
        
        # 統計信息
        self.stats = {
            'total_points_added': 0,
            'predictions_made': 0,
            'interpolations_made': 0,
            'outliers_filtered': 0,
            'average_velocity': 0.0,
            'trajectory_quality': 0.0
        }
    
    def add_detection(self, x: float, y: float, timestamp: float, 
                     confidence: float = 1.0, metadata: Dict = None) -> bool:
        """添加新的檢測點到軌跡
        
        Args:
            x, y: 球體位置（全局座標）
            timestamp: 檢測時間戳
            confidence: 檢測信心度
            metadata: 額外元數據
            
        Returns:
            是否成功添加到軌跡
        """
        new_point = TrajectoryPoint(
            x=x, y=y, timestamp=timestamp, confidence=confidence,
            detection_id=self.current_detection_id, metadata=metadata or {}
        )
        
        # 計算速度（如果有前一點）
        if len(self.trajectory) > 0:
            prev_point = self.trajectory[-1]
            dt = timestamp - prev_point.timestamp
            if dt > 0:
                new_point.velocity_x = (x - prev_point.x) / dt
                new_point.velocity_y = (y - prev_point.y) / dt
        
        # 異常值檢測
        if self._is_outlier(new_point):
            self.stats['outliers_filtered'] += 1
            return False
        
        # 處理軌跡中斷後的恢復
        if self.missing_detection_count > 0:
            self._handle_trajectory_recovery(new_point)
        
        # 添加到軌跡
        self.trajectory.append(new_point)
        self.stats['total_points_added'] += 1
        
        # 更新平滑軌跡
        self._update_smoothed_trajectory(new_point)
        
        # 重置遺漏計數器
        self.missing_detection_count = 0
        self.last_detection_time = timestamp
        self.trajectory_active = True
        self.current_detection_id += 1
        
        return True
    
    def handle_missing_detection(self, current_timestamp: float) -> Optional[TrajectoryPoint]:
        """處理檢測遺漏，嘗試預測軌跡點
        
        Args:
            current_timestamp: 當前時間戳
            
        Returns:
            預測的軌跡點，如果無法預測則返回 None
        """
        self.missing_detection_count += 1
        
        # 如果遺漏太多，停用軌跡
        if self.missing_detection_count > self.max_missing_detections:
            self.trajectory_active = False
            return None
        
        # 嘗試預測下一個位置
        predicted_point = self._predict_next_position(current_timestamp)
        
        if predicted_point:
            self.stats['predictions_made'] += 1
            
            # 添加預測點到軌跡（但標記為預測）
            self.trajectory.append(predicted_point)
            self._update_smoothed_trajectory(predicted_point)
            
        return predicted_point
    
    def get_current_velocity(self) -> Tuple[float, float]:
        """獲取當前速度向量
        
        Returns:
            (vx, vy) 速度向量 (像素/秒)
        """
        if len(self.smoothed_trajectory) < 2:
            return (0.0, 0.0)
        
        recent_points = list(self.smoothed_trajectory)[-3:]
        
        # 計算平均速度
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_points)):
            dt = recent_points[i].timestamp - recent_points[i-1].timestamp
            if dt > 0:
                vx = (recent_points[i].x - recent_points[i-1].x) / dt
                vy = (recent_points[i].y - recent_points[i-1].y) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        if velocities_x:
            avg_vx = np.mean(velocities_x)
            avg_vy = np.mean(velocities_y)
            self.stats['average_velocity'] = math.hypot(avg_vx, avg_vy)
            return (avg_vx, avg_vy)
        
        return (0.0, 0.0)
    
    def get_trajectory_quality(self) -> float:
        """評估軌跡品質
        
        Returns:
            品質分數 (0-1)，1表示最佳品質
        """
        if len(self.trajectory) < self.quality_window_size:
            return 0.5  # 數據不足時返回中等品質
        
        recent_points = list(self.trajectory)[-self.quality_window_size:]
        
        # 評估因子
        confidence_score = np.mean([p.confidence for p in recent_points])
        
        # 軌跡平滑度評估
        smoothness_score = self._calculate_smoothness_score(recent_points)
        
        # 速度一致性評估
        velocity_consistency_score = self._calculate_velocity_consistency(recent_points)
        
        # 預測準確性評估（基於預測點的比例）
        prediction_ratio = sum(1 for p in recent_points if p.predicted) / len(recent_points)
        prediction_score = max(0.0, 1.0 - prediction_ratio)
        
        # 綜合品質分數
        quality = (
            confidence_score * 0.3 +
            smoothness_score * 0.3 +
            velocity_consistency_score * 0.25 +
            prediction_score * 0.15
        )
        
        self.stats['trajectory_quality'] = quality
        return quality
    
    def get_trajectory_points(self, smoothed: bool = True, 
                            max_age_s: Optional[float] = None) -> List[TrajectoryPoint]:
        """獲取軌跡點列表
        
        Args:
            smoothed: 是否返回平滑後的軌跡
            max_age_s: 最大年齡限制（秒）
            
        Returns:
            軌跡點列表
        """
        source_trajectory = self.smoothed_trajectory if smoothed else self.trajectory
        
        if max_age_s is None:
            return list(source_trajectory)
        
        current_time = time.time()
        return [p for p in source_trajectory 
                if (current_time - p.timestamp) <= max_age_s]
    
    def get_recent_positions(self, count: int = 10) -> List[Tuple[float, float, float]]:
        """獲取最近的位置點（用於相容性）
        
        Returns:
            [(x, y, timestamp), ...] 格式的位置列表
        """
        points = list(self.smoothed_trajectory)[-count:]
        return [(p.x, p.y, p.timestamp) for p in points]
    
    def interpolate_missing_segment(self, start_point: TrajectoryPoint, 
                                  end_point: TrajectoryPoint, 
                                  num_interpolations: int = 1) -> List[TrajectoryPoint]:
        """在兩個軌跡點之間插值
        
        Args:
            start_point: 起始點
            end_point: 結束點
            num_interpolations: 插值點數量
            
        Returns:
            插值點列表
        """
        if num_interpolations <= 0:
            return []
        
        interpolated_points = []
        dt_total = end_point.timestamp - start_point.timestamp
        
        for i in range(1, num_interpolations + 1):
            t = i / (num_interpolations + 1)  # 插值參數 (0, 1)
            
            # 線性插值位置
            interp_x = start_point.x + t * (end_point.x - start_point.x)
            interp_y = start_point.y + t * (end_point.y - start_point.y)
            interp_timestamp = start_point.timestamp + t * dt_total
            
            # 插值速度
            interp_vx = start_point.velocity_x + t * (end_point.velocity_x - start_point.velocity_x)
            interp_vy = start_point.velocity_y + t * (end_point.velocity_y - start_point.velocity_y)
            
            # 插值信心度（線性衰減）
            interp_confidence = min(start_point.confidence, end_point.confidence) * 0.8
            
            interpolated_point = TrajectoryPoint(
                x=interp_x, y=interp_y, timestamp=interp_timestamp,
                velocity_x=interp_vx, velocity_y=interp_vy,
                confidence=interp_confidence, interpolated=True,
                detection_id=start_point.detection_id
            )
            
            interpolated_points.append(interpolated_point)
        
        self.stats['interpolations_made'] += len(interpolated_points)
        return interpolated_points
    
    def _is_outlier(self, new_point: TrajectoryPoint) -> bool:
        """檢測新點是否為異常值"""
        if len(self.trajectory) < 3:
            return False
        
        recent_points = list(self.trajectory)[-5:]
        
        # 位置異常值檢測
        positions = [(p.x, p.y) for p in recent_points]
        if len(positions) >= 2:
            distances = [math.hypot(positions[i][0] - positions[i-1][0],
                                  positions[i][1] - positions[i-1][1])
                        for i in range(1, len(positions))]
            
            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                
                current_distance = math.hypot(
                    new_point.x - recent_points[-1].x,
                    new_point.y - recent_points[-1].y
                )
                
                threshold = avg_distance + self.outlier_threshold_factor * std_distance
                if current_distance > threshold:
                    return True
        
        # 速度異常值檢測
        if len(recent_points) >= 2:
            recent_speeds = [math.hypot(p.velocity_x, p.velocity_y) for p in recent_points[-3:]]
            current_speed = math.hypot(new_point.velocity_x, new_point.velocity_y)
            
            if recent_speeds:
                avg_speed = np.mean(recent_speeds)
                std_speed = np.std(recent_speeds)
                
                if std_speed > 0:
                    z_score = abs(current_speed - avg_speed) / std_speed
                    if z_score > self.outlier_threshold_factor:
                        return True
        
        return False
    
    def _predict_next_position(self, timestamp: float) -> Optional[TrajectoryPoint]:
        """預測下一個軌跡點位置"""
        if len(self.smoothed_trajectory) < self.min_points_for_prediction:
            return None
        
        # 使用最近的幾個點進行預測
        recent_points = list(self.smoothed_trajectory)[-5:]
        
        # 計算平均速度和加速度
        velocities = []
        for i in range(1, len(recent_points)):
            dt = recent_points[i].timestamp - recent_points[i-1].timestamp
            if dt > 0:
                vx = (recent_points[i].x - recent_points[i-1].x) / dt
                vy = (recent_points[i].y - recent_points[i-1].y) / dt
                velocities.append((vx, vy, recent_points[i].timestamp))
        
        if len(velocities) < 2:
            return None
        
        # 使用線性外推進行預測
        last_point = recent_points[-1]
        dt_prediction = timestamp - last_point.timestamp
        
        if dt_prediction > self.prediction_horizon_s:
            return None  # 超出預測範圍
        
        # 計算平均速度
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # 預測位置
        predicted_x = last_point.x + avg_vx * dt_prediction
        predicted_y = last_point.y + avg_vy * dt_prediction
        
        # 計算預測信心度（隨時間衰減）
        time_factor = max(0.1, 1.0 - dt_prediction / self.prediction_horizon_s)
        predicted_confidence = last_point.confidence * 0.7 * time_factor
        
        return TrajectoryPoint(
            x=predicted_x, y=predicted_y, timestamp=timestamp,
            velocity_x=avg_vx, velocity_y=avg_vy,
            confidence=predicted_confidence, predicted=True,
            detection_id=self.current_detection_id
        )
    
    def _handle_trajectory_recovery(self, new_point: TrajectoryPoint):
        """處理軌跡中斷後的恢復"""
        if len(self.trajectory) == 0:
            return
        
        last_point = self.trajectory[-1]
        gap_time = new_point.timestamp - last_point.timestamp
        
        # 如果間隔時間適中且啟用插值，添加插值點
        if (self.interpolation_enabled and 
            gap_time <= self.interpolation_max_gap_s and
            gap_time > 0.01):  # 避免插值過於頻繁
            
            num_interpolations = min(3, int(gap_time * 100))  # 基於時間間隔決定插值點數
            interpolated_points = self.interpolate_missing_segment(
                last_point, new_point, num_interpolations
            )
            
            # 添加插值點到軌跡
            for interp_point in interpolated_points:
                self.trajectory.append(interp_point)
                self._update_smoothed_trajectory(interp_point)
    
    def _update_smoothed_trajectory(self, new_point: TrajectoryPoint):
        """更新平滑軌跡"""
        if len(self.trajectory) < self.smoothing_window_size:
            self.smoothed_trajectory.append(new_point)
            return
        
        # 使用移動平均進行平滑
        recent_points = list(self.trajectory)[-self.smoothing_window_size:]
        
        # 加權平均，新點權重較高
        weights = np.linspace(0.5, 1.0, len(recent_points))
        weights = weights / np.sum(weights)
        
        smoothed_x = np.sum([p.x * w for p, w in zip(recent_points, weights)])
        smoothed_y = np.sum([p.y * w for p, w in zip(recent_points, weights)])
        
        # 保持其他屬性
        smoothed_point = TrajectoryPoint(
            x=smoothed_x, y=smoothed_y, timestamp=new_point.timestamp,
            velocity_x=new_point.velocity_x, velocity_y=new_point.velocity_y,
            confidence=new_point.confidence, predicted=new_point.predicted,
            interpolated=new_point.interpolated, detection_id=new_point.detection_id,
            metadata=new_point.metadata
        )
        
        self.smoothed_trajectory.append(smoothed_point)
    
    def _calculate_smoothness_score(self, points: List[TrajectoryPoint]) -> float:
        """計算軌跡平滑度分數"""
        if len(points) < 3:
            return 0.5
        
        # 計算方向變化的平滑度
        direction_changes = []
        for i in range(2, len(points)):
            p1, p2, p3 = points[i-2], points[i-1], points[i]
            
            # 計算兩個向量的夾角
            v1 = (p2.x - p1.x, p2.y - p1.y)
            v2 = (p3.x - p2.x, p3.y - p2.y)
            
            v1_mag = math.hypot(*v1)
            v2_mag = math.hypot(*v2)
            
            if v1_mag > 0 and v2_mag > 0:
                cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (v1_mag * v2_mag)
                cos_angle = max(-1, min(1, cos_angle))  # 限制範圍
                angle_change = math.acos(cos_angle)
                direction_changes.append(angle_change)
        
        if not direction_changes:
            return 0.5
        
        # 平滑度與平均角度變化成反比
        avg_angle_change = np.mean(direction_changes)
        smoothness = max(0.0, 1.0 - avg_angle_change / math.pi)
        
        return smoothness
    
    def _calculate_velocity_consistency(self, points: List[TrajectoryPoint]) -> float:
        """計算速度一致性分數"""
        if len(points) < 2:
            return 0.5
        
        speeds = []
        for point in points:
            speed = math.hypot(point.velocity_x, point.velocity_y)
            speeds.append(speed)
        
        if not speeds:
            return 0.5
        
        speed_std = np.std(speeds)
        speed_mean = np.mean(speeds)
        
        if speed_mean > 0:
            consistency = max(0.0, 1.0 - speed_std / speed_mean)
        else:
            consistency = 0.5
        
        return consistency
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取軌跡管理統計信息"""
        return {
            **self.stats,
            'trajectory_length': len(self.trajectory),
            'smoothed_trajectory_length': len(self.smoothed_trajectory),
            'missing_detection_count': self.missing_detection_count,
            'trajectory_active': self.trajectory_active,
            'current_detection_id': self.current_detection_id
        }
    
    def reset(self):
        """重置軌跡管理器"""
        self.trajectory.clear()
        self.smoothed_trajectory.clear()
        self.current_detection_id = 0
        self.last_detection_time = 0.0
        self.trajectory_active = False
        self.missing_detection_count = 0
        
        # 重置統計
        self.stats = {
            'total_points_added': 0,
            'predictions_made': 0,
            'interpolations_made': 0,
            'outliers_filtered': 0,
            'average_velocity': 0.0,
            'trajectory_quality': 0.0
        }