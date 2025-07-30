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


def parse_arguments():
    """解析命令行參數"""
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
    """根據命令行參數創建配置"""
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
    """主函數"""
    try:
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
    
    return 0


if __name__ == '__main__':
    sys.exit(main())