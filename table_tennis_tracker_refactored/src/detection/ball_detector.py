"""
球體偵測器
從運動遮罩中識別和篩選球體候選
"""

import cv2
import numpy as np
import math
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from ..core.config import DetectionConfig
from ..core.events import BallDetectionEvent, BallPosition


class BallCandidate:
    """球體候選"""
    def __init__(self, position_roi: Tuple[int, int], area: float, 
                 circularity: float, contour: Optional[np.ndarray] = None):
        self.position_roi = position_roi
        self.area = area
        self.circularity = circularity
        self.contour = contour
        self.distance_from_last = float('inf')
        self.consistency = 0.0
        self.score = 0.0


class BallDetector:
    """球體偵測器"""
    
    def __init__(self, config: DetectionConfig, roi_start_x: int, roi_top_y: int, frame_width: int):
        self.config = config
        self.roi_start_x = roi_start_x
        self.roi_top_y = roi_top_y
        self.frame_width = frame_width
        self.trajectory = deque(maxlen=config.max_trajectory_points)
        self.last_detection_timestamp = 0.0
        
    def detect_from_motion_mask(self, motion_mask: np.ndarray, current_timestamp: float, 
                               use_video_timing: bool = False) -> Optional[BallDetectionEvent]:
        """從運動遮罩中偵測球體"""
        candidates = self._find_candidates(motion_mask)
        if not candidates:
            return None
            
        best_candidate = self._select_best_candidate(candidates)
        if not best_candidate:
            return None
            
        # 建立球體位置
        cx_roi, cy_roi = best_candidate.position_roi
        cx_global = cx_roi + self.roi_start_x
        cy_global = cy_roi + self.roi_top_y
        
        timestamp = current_timestamp if not use_video_timing else len(self.trajectory) / 60.0
        
        position = BallPosition(
            x_global=cx_global,
            y_global=cy_global,
            x_roi=cx_roi,
            y_roi=cy_roi,
            timestamp=timestamp
        )
        
        # 更新軌跡
        self.trajectory.append((cx_global, cy_global, timestamp))
        self.last_detection_timestamp = current_timestamp
        
        return BallDetectionEvent(
            position=position,
            area=best_candidate.area,
            circularity=best_candidate.circularity,
            contour=best_candidate.contour
        )
    
    def _find_candidates(self, motion_mask: np.ndarray) -> List[BallCandidate]:
        """在運動遮罩中尋找球體候選"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=8
        )
        
        candidates = []
        
        for i in range(1, num_labels):  # 跳過背景標籤0
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 面積篩選
            if not (self.config.min_ball_area_px < area < self.config.max_ball_area_px):
                continue
                
            x_roi = stats[i, cv2.CC_STAT_LEFT]
            y_roi = stats[i, cv2.CC_STAT_TOP]
            w_roi = stats[i, cv2.CC_STAT_WIDTH]
            h_roi = stats[i, cv2.CC_STAT_HEIGHT]
            cx_roi, cy_roi = centroids[i]
            
            # 計算圓形度
            circularity = 0.0
            contour = None
            
            if max(w_roi, h_roi) > 0:
                component_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    contour = contours[0]
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            candidates.append(BallCandidate(
                position_roi=(int(cx_roi), int(cy_roi)),
                area=area,
                circularity=circularity,
                contour=contour
            ))
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[BallCandidate]) -> Optional[BallCandidate]:
        """選擇最佳球體候選"""
        if not candidates:
            return None
            
        # 如果沒有軌跡歷史，基於圓形度選擇
        if not self.trajectory:
            highly_circular = [
                c for c in candidates 
                if c.circularity > self.config.min_ball_circularity
            ]
            if highly_circular:
                return max(highly_circular, key=lambda c: c.circularity)
            return max(candidates, key=lambda c: c.area)
        
        # 有軌跡歷史時，計算距離和一致性分數
        last_x_global, last_y_global, _ = self.trajectory[-1]
        
        for candidate in candidates:
            cx_roi, cy_roi = candidate.position_roi
            cx_global = cx_roi + self.roi_start_x
            cy_global = cy_roi + self.roi_top_y
            
            # 計算距離
            distance = math.hypot(cx_global - last_x_global, cy_global - last_y_global)
            candidate.distance_from_last = distance
            
            # 距離過大則排除
            if distance > self.frame_width * 0.4:
                candidate.distance_from_last = float('inf')
            
            # 計算運動一致性
            candidate.consistency = self._calculate_consistency(
                cx_global, cy_global, last_x_global, last_y_global
            )
        
        # 計算綜合分數
        for candidate in candidates:
            candidate.score = (
                0.3 / (1.0 + candidate.distance_from_last) +
                0.5 * candidate.consistency +
                0.2 * candidate.circularity
            )
        
        return max(candidates, key=lambda c: c.score)
    
    def _calculate_consistency(self, cx_global: int, cy_global: int, 
                             last_x_global: int, last_y_global: int) -> float:
        """計算運動一致性分數"""
        if len(self.trajectory) < 2:
            return 0.0
            
        prev_x_global, prev_y_global, _ = self.trajectory[-2]
        
        # 歷史向量
        vec_hist_dx = last_x_global - prev_x_global
        vec_hist_dy = last_y_global - prev_y_global
        
        # 當前向量
        vec_curr_dx = cx_global - last_x_global
        vec_curr_dy = cy_global - last_y_global
        
        # 計算餘弦相似度
        dot_product = vec_hist_dx * vec_curr_dx + vec_hist_dy * vec_curr_dy
        mag_hist = math.sqrt(vec_hist_dx**2 + vec_hist_dy**2)
        mag_curr = math.sqrt(vec_curr_dx**2 + vec_curr_dy**2)
        
        if mag_hist > 0 and mag_curr > 0:
            cosine_similarity = dot_product / (mag_hist * mag_curr)
            return max(0, cosine_similarity)
        
        return 0.0
    
    def is_detection_timeout(self, current_time: float) -> bool:
        """檢查是否偵測超時"""
        return current_time - self.last_detection_timestamp > self.config.timeout_s
    
    def reset_trajectory(self) -> None:
        """重置軌跡"""
        self.trajectory.clear()
    
    def get_trajectory(self) -> List[Tuple[int, int, float]]:
        """獲取當前軌跡"""
        return list(self.trajectory)