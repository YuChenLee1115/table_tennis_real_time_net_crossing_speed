"""
改進的速度計算器
提供更準確的實時速度計算，包含多點平均、異常值過濾和透視校正
針對桌球高速運動優化
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from ..core.config import TrackingConfig
from ..utils.perspective import PerspectiveCorrector


class SpeedCalculator:
    """改進的速度計算器 - 針對高速桌球運動優化
    
    主要改進:
    - 多點速度計算和平滑濾波
    - 異常值檢測和過濾
    - 自適應濾波參數
    - 透視校正的精確距離計算
    - 速度置信度評估
    """
    
    def __init__(self, config: TrackingConfig, perspective_corrector: PerspectiveCorrector):
        self.config = config
        self.perspective_corrector = perspective_corrector
        
        # 當前速度狀態
        self.current_speed_kmh = 0.0
        self.current_speed_confidence = 0.0  # 速度可信度 (0-1)
        
        # 速度歷史，用於平滑濾波和統計分析
        self.speed_history = deque(maxlen=20)  # 存儲更多歷史點
        self.instantaneous_speeds = deque(maxlen=10)  # 瞬時速度
        
        # 多點速度計算的窗口大小
        self.velocity_window_sizes = [2, 3, 4, 5]  # 不同窗口大小
        
        # 異常值檢測參數
        self.outlier_threshold_factor = 2.5  # 標準差倍數
        self.min_valid_speed = 0.5  # km/h - 最小有效速度
        self.max_valid_speed = 200.0  # km/h - 最大有效速度
        
        # 自適應濾波參數
        self.base_smoothing_factor = config.speed_smoothing_factor
        self.adaptive_smoothing_factor = config.speed_smoothing_factor
        
        # 速度計算品質統計
        self.calculation_stats = {
            'total_calculations': 0,
            'valid_calculations': 0,
            'outliers_filtered': 0,
            'average_confidence': 0.0
        }
        
        # 透視校正缓存，提高性能
        self.perspective_cache = {}
        self.cache_max_size = 100
    
    def calculate_speed(self, trajectory: List[Tuple[int, int, float]]) -> float:
        """計算當前速度 - 改進版本使用多種方法
        
        Args:
            trajectory: 軌跡點列表 [(x, y, timestamp), ...]
            
        Returns:
            當前速度 (km/h)
        """
        self.calculation_stats['total_calculations'] += 1
        
        if len(trajectory) < 2:
            self.current_speed_kmh = 0.0
            self.current_speed_confidence = 0.0
            return self.current_speed_kmh
        
        # 計算多種速度並評估可信度
        speed_estimates = self._calculate_multiple_speed_estimates(trajectory)
        
        if not speed_estimates:
            self._decay_current_speed()
            return self.current_speed_kmh
        
        # 選擇最佳速度估計
        best_speed, confidence = self._select_best_speed_estimate(speed_estimates)
        
        # 異常值檢測和過濾
        if self._is_speed_outlier(best_speed):
            self.calculation_stats['outliers_filtered'] += 1
            self._decay_current_speed()
            return self.current_speed_kmh
        
        # 更新速度歷史
        self.speed_history.append(best_speed)
        self.current_speed_confidence = confidence
        
        # 自適應濾波
        self._update_adaptive_smoothing()
        filtered_speed = self._apply_adaptive_filter(best_speed)
        
        # 更新當前速度
        self.current_speed_kmh = filtered_speed
        self.calculation_stats['valid_calculations'] += 1
        self.calculation_stats['average_confidence'] = (
            (self.calculation_stats['average_confidence'] * 0.9) + (confidence * 0.1)
        )
        
        return self.current_speed_kmh
    
    def _calculate_multiple_speed_estimates(self, trajectory: List[Tuple[int, int, float]]) -> List[Tuple[float, float]]:
        """使用不同窗口大小計算多種速度估計
        
        Returns:
            [(speed, confidence), ...] 速度和可信度對列表
        """
        estimates = []
        
        for window_size in self.velocity_window_sizes:
            if len(trajectory) >= window_size:
                speed_estimate = self._calculate_windowed_speed(trajectory, window_size)
                if speed_estimate is not None:
                    speed, confidence = speed_estimate
                    if self.min_valid_speed <= speed <= self.max_valid_speed:
                        estimates.append((speed, confidence))
        
        return estimates
    
    def _calculate_windowed_speed(self, trajectory: List[Tuple[int, int, float]], 
                                 window_size: int) -> Optional[Tuple[float, float]]:
        """使用指定窗口大小計算速度
        
        Returns:
            (speed_kmh, confidence) 或 None
        """
        if len(trajectory) < window_size:
            return None
        
        # 取最近的點
        points = trajectory[-window_size:]
        
        # 計算總位移和總時間
        total_distance_cm = 0.0
        total_time = 0.0
        displacement_segments = []
        
        for i in range(1, len(points)):
            p1_x, p1_y, t1 = points[i-1]
            p2_x, p2_y, t2 = points[i]
            
            # 計算考慮透視校正的實際距離
            real_distance_cm = self._calculate_real_distance_cm_cached(
                p1_x, p1_y, p2_x, p2_y
            )
            
            delta_t = t2 - t1
            
            if delta_t > 0:
                displacement_segments.append((real_distance_cm, delta_t))
                total_distance_cm += real_distance_cm
                total_time += delta_t
        
        if total_time <= 0.0001:
            return None
        
        # 計算平均速度
        avg_speed_cm_per_s = total_distance_cm / total_time
        avg_speed_kmh = avg_speed_cm_per_s * self.config.kmh_conversion_factor
        
        # 計算可信度基於時間跨度和位移一致性
        confidence = self._calculate_speed_confidence(
            displacement_segments, window_size, total_time
        )
        
        return avg_speed_kmh, confidence
    
    def _calculate_real_distance_cm_cached(self, x1_global: int, y1_global: int, 
                                          x2_global: int, y2_global: int) -> float:
        """帶緩存的實際距離計算，提高性能"""
        # 創建缓存鍵
        cache_key = (x1_global // 10, y1_global // 10, x2_global // 10, y2_global // 10)
        
        if cache_key in self.perspective_cache:
            return self.perspective_cache[cache_key]
        
        # 計算實際距離
        distance = self._calculate_real_distance_cm(x1_global, y1_global, x2_global, y2_global)
        
        # 更新緩存
        if len(self.perspective_cache) >= self.cache_max_size:
            # 移除最老的緩存項
            oldest_key = next(iter(self.perspective_cache))
            del self.perspective_cache[oldest_key]
        
        self.perspective_cache[cache_key] = distance
        return distance
    
    def _calculate_real_distance_cm(self, x1_global: int, y1_global: int, 
                                   x2_global: int, y2_global: int) -> float:
        """計算考慮透視校正的實際距離"""
        # 獲取兩點的像素到實際距離比例
        ratio1 = self.perspective_corrector.get_pixel_to_cm_ratio(y1_global)
        ratio2 = self.perspective_corrector.get_pixel_to_cm_ratio(y2_global)
        
        # 使用更精確的加權平均，考慮距離權重
        pixel_distance = math.hypot(x2_global - x1_global, y2_global - y1_global)
        
        if pixel_distance == 0:
            return 0.0
        
        # 對於長距離移動，使用積分方法進行更精確的透視校正
        if pixel_distance > 50:  # 像素
            return self._integrate_perspective_correction(
                x1_global, y1_global, x2_global, y2_global
            )
        else:
            # 短距離使用簡單平均
            avg_ratio = (ratio1 + ratio2) / 2.0
            return pixel_distance * avg_ratio
    
    def _integrate_perspective_correction(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """對長距離移動使用積分方法進行透視校正"""
        # 將路徑分割成多個小段進行積分
        num_segments = max(5, int(math.hypot(x2 - x1, y2 - y1) // 20))
        
        total_distance = 0.0
        
        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            
            # 線性插值計算段點
            seg_x1 = x1 + t1 * (x2 - x1)
            seg_y1 = y1 + t1 * (y2 - y1)
            seg_x2 = x1 + t2 * (x2 - x1)
            seg_y2 = y1 + t2 * (y2 - y1)
            
            # 計算段中點的透視比例
            mid_y = (seg_y1 + seg_y2) / 2
            ratio = self.perspective_corrector.get_pixel_to_cm_ratio(int(mid_y))
            
            # 段長度
            seg_pixel_length = math.hypot(seg_x2 - seg_x1, seg_y2 - seg_y1)
            seg_real_length = seg_pixel_length * ratio
            
            total_distance += seg_real_length
        
        return total_distance
    
    def _calculate_speed_confidence(self, displacement_segments: List[Tuple[float, float]], 
                                   window_size: int, total_time: float) -> float:
        """計算速度估計的可信度"""
        if not displacement_segments:
            return 0.0
        
        # 基於時間跨度的可信度 - 更長的時間跨度更可信
        time_confidence = min(1.0, total_time / 0.1)  # 0.1秒為滿分
        
        # 基於窗口大小的可信度 - 更多點更可信
        window_confidence = min(1.0, window_size / 5.0)
        
        # 基於位移一致性的可信度
        if len(displacement_segments) > 1:
            speeds = [dist / time for dist, time in displacement_segments if time > 0]
            if speeds:
                speed_std = np.std(speeds)
                speed_mean = np.mean(speeds)
                consistency_confidence = max(0.0, 1.0 - (speed_std / max(speed_mean, 1.0)))
            else:
                consistency_confidence = 0.0
        else:
            consistency_confidence = 0.7
        
        # 綜合可信度
        overall_confidence = (
            time_confidence * 0.4 + 
            window_confidence * 0.3 + 
            consistency_confidence * 0.3
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _select_best_speed_estimate(self, estimates: List[Tuple[float, float]]) -> Tuple[float, float]:
        """選擇最佳速度估計 - 基於可信度和一致性"""
        if not estimates:
            return 0.0, 0.0
        
        if len(estimates) == 1:
            return estimates[0]
        
        # 根據可信度排序
        estimates.sort(key=lambda x: x[1], reverse=True)
        
        # 如果最高可信度的估計顯著高於其他，直接使用
        best_estimate = estimates[0]
        if best_estimate[1] > 0.8:
            return best_estimate
        
        # 否則使用加權平均，權重為可信度
        total_weight = sum(conf for _, conf in estimates)
        if total_weight == 0:
            return estimates[0]
        
        weighted_speed = sum(speed * conf for speed, conf in estimates) / total_weight
        avg_confidence = total_weight / len(estimates)
        
        return weighted_speed, avg_confidence
    
    def _is_speed_outlier(self, speed: float) -> bool:
        """檢測速度是否為異常值"""
        # 基本範圍檢查
        if speed < self.min_valid_speed or speed > self.max_valid_speed:
            return True
        
        # 如果沒有歷史數據，不認為是異常值
        if len(self.speed_history) < 3:
            return False
        
        # 統計學異常值檢測
        recent_speeds = list(self.speed_history)[-10:]
        speed_mean = np.mean(recent_speeds)
        speed_std = np.std(recent_speeds)
        
        # Z-score 檢測
        if speed_std > 0:
            z_score = abs(speed - speed_mean) / speed_std
            return z_score > self.outlier_threshold_factor
        
        return False
    
    def _update_adaptive_smoothing(self):
        """根據速度變化更新自適應濾波參數"""
        if len(self.speed_history) < 5:
            self.adaptive_smoothing_factor = self.base_smoothing_factor
            return
        
        # 計算最近速度的變化率
        recent_speeds = list(self.speed_history)[-5:]
        speed_changes = [abs(recent_speeds[i] - recent_speeds[i-1]) 
                        for i in range(1, len(recent_speeds))]
        
        if speed_changes:
            avg_change = np.mean(speed_changes)
            
            # 根據變化率調整平滑係數
            # 變化大時減少平滑（更快響應），變化小時增加平滑（更穩定）
            change_factor = min(2.0, avg_change / 10.0)  # 標準化變化率
            
            self.adaptive_smoothing_factor = max(
                0.1, min(0.8, self.base_smoothing_factor * (1.0 + change_factor))
            )
    
    def _apply_adaptive_filter(self, new_speed: float) -> float:
        """應用自適應濾波器"""
        if self.current_speed_kmh == 0.0:
            return new_speed
        
        # 根據可信度調整濾波強度
        confidence_factor = max(0.1, self.current_speed_confidence)
        effective_smoothing = self.adaptive_smoothing_factor * confidence_factor
        
        # 指數移動平均濾波
        filtered_speed = (
            (1 - effective_smoothing) * self.current_speed_kmh +
            effective_smoothing * new_speed
        )
        
        return filtered_speed
    
    def _decay_current_speed(self):
        """當無法計算新速度時衰減當前速度"""
        self.current_speed_kmh *= (1 - self.adaptive_smoothing_factor * 2)
        self.current_speed_confidence *= 0.9
    
    def get_current_speed(self) -> float:
        """獲取當前速度"""
        return self.current_speed_kmh
    
    def get_speed_confidence(self) -> float:
        """獲取當前速度的可信度"""
        return self.current_speed_confidence
    
    def get_speed_statistics(self) -> Dict:
        """獲取速度計算統計資訊"""
        if self.calculation_stats['total_calculations'] > 0:
            success_rate = (
                self.calculation_stats['valid_calculations'] / 
                self.calculation_stats['total_calculations']
            )
        else:
            success_rate = 0.0
        
        return {
            'current_speed': self.current_speed_kmh,
            'speed_confidence': self.current_speed_confidence,
            'success_rate': success_rate,
            'outliers_filtered': self.calculation_stats['outliers_filtered'],
            'average_confidence': self.calculation_stats['average_confidence'],
            'adaptive_smoothing_factor': self.adaptive_smoothing_factor,
            'speed_history_length': len(self.speed_history)
        }
    
    def reset(self) -> None:
        """重置速度計算器狀態"""
        self.current_speed_kmh = 0.0
        self.current_speed_confidence = 0.0
        self.speed_history.clear()
        self.instantaneous_speeds.clear()
        self.adaptive_smoothing_factor = self.base_smoothing_factor
        
        # 重置統計
        self.calculation_stats = {
            'total_calculations': 0,
            'valid_calculations': 0,
            'outliers_filtered': 0,
            'average_confidence': 0.0
        }
        
        # 清除緩存
        self.perspective_cache.clear()