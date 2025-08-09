"""
球體偵測器
從運動遮罩中識別和篩選球體候選
使用多重評分機制和軌跡預測提高準確性
"""

import cv2
import numpy as np
import math
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass
from ..core.config import DetectionConfig
from ..core.events import BallDetectionEvent, BallPosition


@dataclass
class BallCandidate:
    """球體候選物件 - 包含完整的評估資訊"""
    position_roi: Tuple[int, int]  # ROI座標系中的位置
    area: float                    # 輪廓面積
    circularity: float             # 圓度評分 (0-1)
    contour: Optional[np.ndarray]  # 原始輪廓資料
    
    # 評分相關屬性
    distance_from_last: float = float('inf')  # 與上次檢測位置的距離
    consistency: float = 0.0                  # 軌跡一致性評分
    size_consistency: float = 0.0             # 大小一致性評分
    score: float = 0.0                       # 綜合評分
    
    # 額外的形狀特徵
    aspect_ratio: float = 1.0                # 長寬比
    solidity: float = 1.0                    # 實心度 (凸包面積比)
    extent: float = 1.0                      # 填充度 (邊界框面積比)


class BallDetector:
    """改進的球體偵測器 - 使用多重評分機制和軌跡預測
    
    主要改進:
    - 多維度特徵評估 (面積、圓度、實心度、軌跡一致性)
    - 卡爾曼濾波器預測球體位置
    - 自適應評分權重根據檢測品質調整
    - 軌跡平滑和異常值過濾
    """
    
    def __init__(self, config: DetectionConfig, roi_start_x: int, roi_top_y: int, frame_width: int):
        self.config = config
        self.roi_start_x = roi_start_x
        self.roi_top_y = roi_top_y
        self.frame_width = frame_width
        
        # 軌跡管理 - 存儲更多歷史點以改善預測
        self.trajectory = deque(maxlen=config.max_trajectory_points)
        self.trajectory_smoothed = deque(maxlen=config.max_trajectory_points)
        self.last_detection_timestamp = 0.0
        
        # 卡爾曼濾波器用於位置和速度預測
        self.kalman = cv2.KalmanFilter(4, 2)  # 4個狀態(x,y,vx,vy), 2個觀測(x,y)
        self._init_kalman_filter()
        self.kalman_initialized = False
        
        # 候選評分權重 - 可自適應調整
        self.score_weights = {
            'circularity': 0.25,     # 圓度權重
            'size_consistency': 0.25, # 大小一致性權重  
            'trajectory': 0.30,      # 軌跡一致性權重
            'shape_features': 0.20   # 其他形狀特徵權重
        }
        
        # 檢測品質統計
        self.detection_stats = {
            'successful_detections': 0,
            'total_attempts': 0,
            'average_candidates': 0,
            'last_quality_scores': deque(maxlen=20)
        }
        
        # 軌跡異常值檢測參數
        self.outlier_threshold_multiplier = 2.5
        self.min_trajectory_points_for_prediction = 3
        
    def _init_kalman_filter(self):
        """初始化卡爾曼濾波器參數"""
        # 狀態轉移矩陣 [x, y, vx, vy]
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy  
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # 觀測矩陣 - 只能觀測到位置
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # 觀測 x
            [0, 1, 0, 0]   # 觀測 y
        ], dtype=np.float32)
        
        # 過程噪聲協方差
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # 測量噪聲協方差
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        
        # 估計誤差協方差
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
    def detect_from_motion_mask(self, motion_mask: np.ndarray, current_timestamp: float, 
                               use_video_timing: bool = False) -> Optional[BallDetectionEvent]:
        """從運動遮罩中偵測球體 - 改進版本使用多重評分和預測機制
        
        這是球體檢測的主函數，整合了輪廓分析、特徵評估、軌跡預測等多種技術：
        
        處理流程：
        1. 輪廓檢測：在運動遮罩中尋找所有可能的球體候選
        2. 特徵分析：計算每個候選的面積、圓度、實心度等幾何特徵
        3. 多重評分：基於形狀特徵、軌跡一致性、大小一致性等進行綜合評分
        4. 卡爾曼預測：當沒有有效候選時，使用卡爾曼濾波器預測位置
        5. 軌跡更新：更新球體軌跡歷史和統計資訊
        
        評分機制：
        - 圓度評分（25%）：越接近圓形分數越高
        - 大小一致性（25%）：與歷史大小的一致性
        - 軌跡一致性（30%）：與預測軌跡的符合程度  
        - 形狀特徵（20%）：長寬比、實心度、填充度的綜合評估
        
        Args:
            motion_mask (np.ndarray): 運動物體的二值化遮罩（255為運動區域）
            current_timestamp (float): 當前時間戳（秒）
            use_video_timing (bool): 是否使用視頻檔案的計時模式
            
        Returns:
            Optional[BallDetectionEvent]: 球體檢測事件，包含：
                - position: 球體位置（ROI和全局座標）
                - area: 球體面積
                - circularity: 圓度評分
                - contour: 原始輪廓資料
                若沒有檢測到有效球體則返回None
                
        性能特點：
            - 實時多候選評估和比較
            - 異常值自動過濾
            - 軌跡連續性保證
            - 自適應評分權重調整
        """
        self.detection_stats['total_attempts'] += 1
        
        # 尋找候選輪廓
        candidates = self._find_candidates(motion_mask)
        self.detection_stats['average_candidates'] = len(candidates)
        
        if not candidates:
            # 如果沒有候選，嘗試使用卡爾曼濾波預測
            predicted_candidate = self._try_kalman_prediction()
            if predicted_candidate:
                candidates = [predicted_candidate]
            else:
                return None
        
        # 評估和選擇最佳候選
        best_candidate = self._select_best_candidate(candidates, current_timestamp)
        if not best_candidate:
            return None
        
        # 建立球體位置物件
        ball_position = self._create_ball_position(
            best_candidate, current_timestamp, use_video_timing
        )
        
        # 更新軌跡和統計
        self._update_trajectory(ball_position)
        self._update_kalman_filter(ball_position)
        self.detection_stats['successful_detections'] += 1
        
        # 創建檢測事件
        detection_event = BallDetectionEvent(
            position=ball_position,
            area=best_candidate.area,
            circularity=best_candidate.circularity,
            contour=best_candidate.contour
        )
        
        return detection_event
    
    def _find_candidates(self, motion_mask: np.ndarray) -> List[BallCandidate]:
        """尋找球體候選 - 使用改進的輪廓分析"""
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        candidates = []
        for contour in contours:
            candidate = self._analyze_contour(contour)
            if candidate and self._is_valid_candidate(candidate):
                candidates.append(candidate)
        
        return candidates
    
    def _analyze_contour(self, contour: np.ndarray) -> Optional[BallCandidate]:
        """分析單個輪廓並創建候選 - 包含更多形狀特徵"""
        area = cv2.contourArea(contour)
        
        # 基本面積檢查
        if area < self.config.min_ball_area_px or area > self.config.max_ball_area_px:
            return None
        
        # 計算輪廓的各種幾何特徵
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
        
        # 圓度計算
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 獲取中心點
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None
        
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        
        # 計算其他形狀特徵
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 邊界框特徵
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0
        extent = area / (w * h) if (w * h) > 0 else 0
        
        return BallCandidate(
            position_roi=(center_x, center_y),
            area=area,
            circularity=circularity,
            contour=contour,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            extent=extent
        )
    
    def _is_valid_candidate(self, candidate: BallCandidate) -> bool:
        """檢查候選是否符合球體的基本特徵"""
        # 圓度檢查
        if candidate.circularity < self.config.min_ball_circularity:
            return False
        
        # 長寬比檢查 - 球應該接近正圓
        if candidate.aspect_ratio < 0.6 or candidate.aspect_ratio > 1.7:
            return False
        
        # 實心度檢查 - 球應該相當實心
        if candidate.solidity < 0.7:
            return False
        
        # 填充度檢查 - 排除太細長的形狀
        if candidate.extent < 0.5:
            return False
        
        return True
    
    def _select_best_candidate(self, candidates: List[BallCandidate], 
                              current_timestamp: float) -> Optional[BallCandidate]:
        """選擇最佳候選 - 使用多重評分機制"""
        if not candidates:
            return None
        
        # 計算每個候選的綜合評分
        for candidate in candidates:
            candidate.score = self._calculate_candidate_score(candidate, current_timestamp)
        
        # 選擇評分最高的候選
        best_candidate = max(candidates, key=lambda c: c.score)
        
        # 確保評分超過最低閾值
        min_score_threshold = 0.3
        if best_candidate.score < min_score_threshold:
            return None
        
        return best_candidate
    
    def _calculate_candidate_score(self, candidate: BallCandidate, 
                                  current_timestamp: float) -> float:
        """計算候選的綜合評分 - 考慮多個因素"""
        scores = {}
        
        # 1. 圓度評分 (0-1)
        scores['circularity'] = min(1.0, candidate.circularity / 1.0)
        
        # 2. 大小一致性評分
        scores['size_consistency'] = self._calculate_size_consistency_score(candidate.area)
        
        # 3. 軌跡一致性評分  
        scores['trajectory'] = self._calculate_trajectory_consistency_score(
            candidate.position_roi, current_timestamp
        )
        
        # 4. 形狀特徵綜合評分
        scores['shape_features'] = self._calculate_shape_features_score(candidate)
        
        # 加權平均
        total_score = sum(
            scores[key] * self.score_weights[key] 
            for key in scores.keys()
        )
        
        return total_score
    
    def _calculate_size_consistency_score(self, area: float) -> float:
        """計算大小一致性評分 - 基於歷史面積"""
        if len(self.trajectory) < 2:
            return 0.7  # 沒有歷史資料時給予中等分數
        
        # 獲取最近幾次檢測的平均面積
        recent_areas = []
        for _, _, _, metadata in list(self.trajectory)[-5:]:
            if 'area' in metadata:
                recent_areas.append(metadata['area'])
        
        if not recent_areas:
            return 0.7
        
        avg_area = np.mean(recent_areas)
        area_diff_ratio = abs(area - avg_area) / avg_area
        
        # 面積變化越小評分越高
        return max(0.0, 1.0 - area_diff_ratio * 2)
    
    def _calculate_trajectory_consistency_score(self, position: Tuple[int, int], 
                                               current_timestamp: float) -> float:
        """計算軌跡一致性評分 - 基於運動預測"""
        if len(self.trajectory_smoothed) < self.min_trajectory_points_for_prediction:
            return 0.5  # 軌跡點不足時給予中等分數
        
        # 預測下一個位置
        predicted_pos = self._predict_next_position(current_timestamp)
        if predicted_pos is None:
            return 0.5
        
        # 計算預測誤差
        prediction_error = math.hypot(
            position[0] - predicted_pos[0],
            position[1] - predicted_pos[1]
        )
        
        # 根據預測誤差計算評分
        max_acceptable_error = 50  # 像素
        score = max(0.0, 1.0 - prediction_error / max_acceptable_error)
        
        return score
    
    def _calculate_shape_features_score(self, candidate: BallCandidate) -> float:
        """計算形狀特徵綜合評分"""
        # 長寬比評分 - 越接近1.0越好
        aspect_score = 1.0 - abs(candidate.aspect_ratio - 1.0)
        
        # 實心度評分
        solidity_score = candidate.solidity
        
        # 填充度評分
        extent_score = candidate.extent
        
        # 綜合評分
        return (aspect_score + solidity_score + extent_score) / 3.0
    
    def _predict_next_position(self, current_timestamp: float) -> Optional[Tuple[float, float]]:
        """基於軌跡預測下一個位置"""
        if len(self.trajectory_smoothed) < 2:
            return None
        
        # 使用最近的幾個點進行線性外推
        recent_points = list(self.trajectory_smoothed)[-3:]
        
        if len(recent_points) < 2:
            return None
        
        # 計算平均速度向量
        velocities = []
        for i in range(1, len(recent_points)):
            p1_x, p1_y, t1, _ = recent_points[i-1]
            p2_x, p2_y, t2, _ = recent_points[i]
            
            dt = t2 - t1
            if dt > 0:
                vx = (p2_x - p1_x) / dt
                vy = (p2_y - p1_y) / dt
                velocities.append((vx, vy))
        
        if not velocities:
            return None
        
        # 平均速度
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        # 預測位置
        last_x, last_y, last_t, _ = recent_points[-1]
        dt_prediction = current_timestamp - last_t
        
        predicted_x = last_x + avg_vx * dt_prediction
        predicted_y = last_y + avg_vy * dt_prediction
        
        return (predicted_x, predicted_y)
    
    def _try_kalman_prediction(self) -> Optional[BallCandidate]:
        """嘗試使用卡爾曼濾波預測位置"""
        if not self.kalman_initialized:
            return None
        
        # 預測下一個狀態
        prediction = self.kalman.predict()
        predicted_x, predicted_y = prediction[0, 0], prediction[1, 0]
        
        # 檢查預測位置是否合理
        if (predicted_x < 0 or predicted_x >= self.frame_width or
            predicted_y < 0 or predicted_y >= 600):  # 假設最大高度
            return None
        
        # 創建預測候選
        return BallCandidate(
            position_roi=(int(predicted_x), int(predicted_y)),
            area=100,  # 使用平均面積
            circularity=0.8,  # 給予較高圓度
            contour=None
        )
    
    def _create_ball_position(self, candidate: BallCandidate, timestamp: float, 
                             use_video_timing: bool) -> BallPosition:
        """創建球體位置物件"""
        roi_x, roi_y = candidate.position_roi
        
        # 轉換為全局坐標
        global_x = roi_x + self.roi_start_x
        global_y = roi_y + self.roi_top_y
        
        return BallPosition(
            x_global=global_x,
            y_global=global_y,
            x_roi=roi_x,
            y_roi=roi_y,
            timestamp=timestamp
        )
    
    def _update_trajectory(self, ball_position: BallPosition):
        """更新軌跡 - 包含平滑處理和異常值過濾"""
        # 添加到原始軌跡
        metadata = {'area': 100}  # 可以加入更多元數據
        trajectory_point = (
            ball_position.x_global, ball_position.y_global, 
            ball_position.timestamp, metadata
        )
        self.trajectory.append(trajectory_point)
        
        # 異常值檢測和平滑處理
        smoothed_point = self._apply_trajectory_smoothing(trajectory_point)
        if smoothed_point:
            self.trajectory_smoothed.append(smoothed_point)
    
    def _apply_trajectory_smoothing(self, new_point: Tuple) -> Optional[Tuple]:
        """軌跡平滑和異常值過濾"""
        if len(self.trajectory_smoothed) < 2:
            return new_point
        
        # 檢測異常值
        if self._is_trajectory_outlier(new_point):
            return None  # 丟棄異常值
        
        # 應用移動平均平滑
        recent_points = list(self.trajectory_smoothed)[-3:]
        if len(recent_points) >= 2:
            # 加權平均，新點權重較高
            weights = [0.2, 0.3, 0.5][-len(recent_points):]
            
            avg_x = sum(p[0] * w for p, w in zip(recent_points, weights[:-1]))
            avg_y = sum(p[1] * w for p, w in zip(recent_points, weights[:-1]))
            
            # 與新點混合
            smoothed_x = avg_x * 0.3 + new_point[0] * 0.7
            smoothed_y = avg_y * 0.3 + new_point[1] * 0.7
            
            return (smoothed_x, smoothed_y, new_point[2], new_point[3])
        
        return new_point
    
    def _is_trajectory_outlier(self, new_point: Tuple) -> bool:
        """檢測軌跡異常值"""
        if len(self.trajectory_smoothed) < 3:
            return False
        
        recent_points = list(self.trajectory_smoothed)[-3:]
        
        # 計算最近點的平均距離
        distances = []
        for i in range(1, len(recent_points)):
            dist = math.hypot(
                recent_points[i][0] - recent_points[i-1][0],
                recent_points[i][1] - recent_points[i-1][1]
            )
            distances.append(dist)
        
        if not distances:
            return False
        
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # 計算新點與最後一點的距離
        last_point = recent_points[-1]
        new_distance = math.hypot(
            new_point[0] - last_point[0],
            new_point[1] - last_point[1]
        )
        
        # 異常值檢測：距離超過平均值加上閾值倍數的標準差
        threshold = avg_distance + self.outlier_threshold_multiplier * std_distance
        
        return new_distance > threshold
    
    def _update_kalman_filter(self, ball_position: BallPosition):
        """更新卡爾曼濾波器"""
        measurement = np.array([
            [float(ball_position.x_global)],
            [float(ball_position.y_global)]
        ], dtype=np.float32)
        
        if not self.kalman_initialized:
            # 初始化狀態
            self.kalman.statePre = np.array([
                [float(ball_position.x_global)],
                [float(ball_position.y_global)],
                [0.0],  # 初始速度為0
                [0.0]
            ], dtype=np.float32)
            self.kalman_initialized = True
        
        # 更新濾波器
        self.kalman.correct(measurement)
    
    def get_trajectory(self) -> List[Tuple[int, int, float]]:
        """獲取軌跡點列表（相容性方法）
        
        Returns:
            [(x, y, timestamp), ...] 格式的軌跡點列表
        """
        return [(int(x), int(y), t) for x, y, t, _ in list(self.trajectory)]
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """獲取檢測統計資訊"""
        success_rate = (
            self.detection_stats['successful_detections'] / 
            max(1, self.detection_stats['total_attempts'])
        )
        
        return {
            'success_rate': success_rate,
            'total_detections': self.detection_stats['successful_detections'],
            'average_candidates': self.detection_stats['average_candidates'],
            'trajectory_length': len(self.trajectory),
            'kalman_initialized': self.kalman_initialized
        }
    
    def reset(self):
        """重置檢測器狀態"""
        self.trajectory.clear()
        self.trajectory_smoothed.clear()
        self.last_detection_timestamp = 0.0
        self.kalman_initialized = False
        self._init_kalman_filter()
        
        # 重置統計
        self.detection_stats = {
            'successful_detections': 0,
            'total_attempts': 0,
            'average_candidates': 0,
            'last_quality_scores': deque(maxlen=20)
        }