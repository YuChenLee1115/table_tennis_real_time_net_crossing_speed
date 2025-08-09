#!/usr/bin/env python3
"""
桌球速度追蹤系統 - 重構版本
主入口點
"""

import argparse
import sys
import os

# 添加src路徑到Python路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import create_default_config
from src.core.tracker import TableTennisTracker
from src.utils.performance_optimizer import initialize_performance_optimization, cleanup_performance_optimization


def parse_arguments():
    """解析命令行參數
    
    解析並驗證用戶輸入的命令行參數，包括視頻源、相機設定、檢測參數、
    追蹤參數和系統設定。提供詳細的幫助信息和使用範例。
    
    Returns:
        argparse.Namespace: 包含所有解析後參數的命名空間對象
        
    參數說明:
        --video: 視頻檔案路徑，若未指定則使用攝像頭
        --camera_idx: 攝像頭索引號（預設：0）
        --fps: 目標幀率（預設：60）
        --width/--height: 影像解析度（預設：1280x720）
        --timeout: 球體檢測超時時間（預設：0.2秒）
        --table_len: 桌子長度，用於像素/公分比例計算（預設：70公分）
        --near_width/--far_width: 近端/遠端寬度，用於透視校正（預設：29/72公分）
        --direction: 網線穿越方向（right_to_left/left_to_right/both）
        --count: 每次收集的網線穿越速度數量（預設：30）
        --cooldown: 穿越事件冷卻時間（預設：0.2秒）
        --debug: 開啟除錯模式
    """
    parser = argparse.ArgumentParser(
        description='Table Tennis Speed Tracker (Refactored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用攝像頭
  python main.py
  
  # 使用影片檔案
  python main.py --video path/to/video.mp4
  
  # 開啟除錯模式
  python main.py --debug
  
  # 自定義參數
  python main.py --fps 60 --width 1280 --height 720 --count 20
        """
    )
    
    # 輸入源
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file. If None, uses webcam.')
    parser.add_argument('--camera_idx', type=int, default=0,
                       help='Webcam index (default: 0).')
    
    # 攝像頭設置
    parser.add_argument('--fps', type=int, default=60,
                       help='Target FPS for webcam (default: 60).')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280).')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720).')
    
    # 偵測設置
    parser.add_argument('--timeout', type=float, default=0.2,
                       help='Ball detection timeout in seconds (default: 0.2).')
    
    # 追蹤設置
    parser.add_argument('--table_len', type=float, default=70.0,
                       help='Table length in cm for nominal px/cm ratio (default: 70.0).')
    parser.add_argument('--near_width', type=float, default=29.0,
                       help='Real width (cm) of ROI at near side (default: 29.0).')
    parser.add_argument('--far_width', type=float, default=72.0,
                       help='Real width (cm) of ROI at far side (default: 72.0).')
    parser.add_argument('--direction', type=str, default='right_to_left',
                       choices=['left_to_right', 'right_to_left', 'both'],
                       help='Net crossing direction to record (default: right_to_left).')
    parser.add_argument('--count', type=int, default=30,
                       help='Number of net speeds to collect per session (default: 30).')
    parser.add_argument('--cooldown', type=float, default=0.2,
                       help='Crossing event cooldown in seconds (default: 0.2).')
    
    # 系統設置
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug printouts.')
    
    return parser.parse_args()


def create_config_from_args(args):
    """根據命令行參數創建系統配置
    
    將解析後的命令行參數轉換為系統配置對象，覆蓋預設配置中的相應設定。
    這允許用戶在不修改配置文件的情況下，通過命令行參數調整系統行為。
    
    Args:
        args (argparse.Namespace): parse_arguments()返回的參數對象
        
    Returns:
        SystemConfig: 更新後的系統配置對象
        
    配置更新項目:
        - 相機配置：索引、幀率、解析度
        - 檢測配置：超時時間
        - 追蹤配置：桌子尺寸、透視校正參數、穿越檢測設定
        - 系統配置：除錯模式
    """
    config = create_default_config()
    
    # 更新攝像頭配置
    config.camera.default_index = args.camera_idx
    config.camera.target_fps = args.fps
    config.camera.frame_width = args.width
    config.camera.frame_height = args.height
    
    # 更新偵測配置
    config.detection.timeout_s = args.timeout
    
    # 更新追蹤配置
    config.tracking.table_length_cm = args.table_len
    config.tracking.near_side_width_cm = args.near_width
    config.tracking.far_side_width_cm = args.far_width
    config.tracking.net_crossing_direction = args.direction
    config.tracking.max_net_speeds_to_collect = args.count
    config.tracking.crossing_cooldown_s = args.cooldown
    
    # 更新系統配置
    config.debug_mode = args.debug
    
    return config


def main():
    """主函數 - 系統入口點
    
    程序的主要執行流程，負責：
    1. 初始化Apple Silicon性能優化
    2. 解析命令行參數並創建配置
    3. 確定視頻源（攝像頭或影片檔案）
    4. 創建並運行桌球追蹤器
    5. 處理異常情況和用戶中斷
    6. 清理系統資源
    
    Returns:
        int: 程序退出碼（0：成功，1：錯誤）
        
    異常處理:
        - KeyboardInterrupt: 用戶按下Ctrl+C中斷程序
        - 其他Exception: 程序運行中的其他錯誤
        
    資源管理:
        - 自動清理性能優化器資源
        - 確保在任何情況下都能正確清理
    """
    optimizer = None
    try:
        # 初始化性能優化
        print("Initializing performance optimizations for Apple Silicon...")
        optimizer = initialize_performance_optimization()
        
        # 解析參數
        args = parse_arguments()
        
        # 創建配置
        config = create_config_from_args(args)
        
        # 確定視頻源
        video_source = args.video if args.video else args.camera_idx
        use_video_file = args.video is not None
        
        # 創建追蹤器
        tracker = TableTennisTracker(
            config=config,
            video_source=video_source,
            use_video_file=use_video_file,
            video_file_path=args.video
        )
        
        # 運行追蹤器
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # 清理性能優化資源
        if optimizer:
            print("Cleaning up performance optimizations...")
            cleanup_performance_optimization()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())