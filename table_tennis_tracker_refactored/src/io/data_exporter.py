"""
數據輸出器
負責將速度數據輸出為圖表和文件
"""

import os
import csv
import datetime
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import concurrent.futures
from ..core.config import IOConfig


class DataExporter:
    """數據輸出器"""
    
    def __init__(self, config: IOConfig):
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    def export_async(self, speeds: List[float], times: List[float], 
                    session_id: int, use_video_file: bool = False, 
                    video_file_path: Optional[str] = None) -> None:
        """異步輸出數據"""
        speeds_copy = list(speeds)
        times_copy = list(times)
        
        self.executor.submit(
            self._export_files, speeds_copy, times_copy, session_id, 
            use_video_file, video_file_path
        )
    
    def _export_files(self, speeds: List[float], times: List[float], 
                     session_id: int, use_video_file: bool = False, 
                     video_file_path: Optional[str] = None) -> None:
        """實際執行文件輸出"""
        if not speeds:
            print("No speed data to export.")
            return
            
        output_dir, base_filename = self._determine_output_path(
            use_video_file, video_file_path
        )
        
        # 計算統計數據
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        min_speed = min(speeds)
        
        # 生成文件路徑
        chart_path = os.path.join(output_dir, f'{base_filename}_chart.png')
        txt_path = os.path.join(output_dir, f'{base_filename}_data.txt')
        csv_path = os.path.join(output_dir, f'{base_filename}_data.csv')
        
        # 輸出文件
        self._create_chart(chart_path, speeds, times, session_id, base_filename, 
                          avg_speed, max_speed, min_speed)
        self._create_text_file(txt_path, speeds, times, session_id, base_filename, 
                              avg_speed, max_speed, min_speed)
        self._create_csv_file(csv_path, speeds, times, session_id, base_filename, 
                             avg_speed, max_speed, min_speed)
        
        print(f"Export completed for session {session_id}.")
    
    def _determine_output_path(self, use_video_file: bool, 
                              video_file_path: Optional[str]) -> tuple:
        """確定輸出路徑和文件名"""
        if use_video_file and video_file_path:
            try:
                # 使用影片所在目錄
                output_dir = os.path.dirname(video_file_path)
                video_name = os.path.splitext(os.path.basename(video_file_path))[0]
                
                # 解析檔名格式
                parts = video_name.split('_')
                if len(parts) >= 2:
                    base_filename = f"{parts[0]}_{parts[1]}"
                else:
                    base_filename = video_name
                    
                print(f"Video mode: Using prefix '{base_filename}' in '{output_dir}'")
                return output_dir, base_filename
                
            except Exception as e:
                print(f"Error parsing video path: {e}. Using timestamp mode.")
        
        # 即時模式或回退模式
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.config.output_data_folder}/{timestamp_str}"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = f"speed_data_{timestamp_str}"
        
        print(f"Real-time mode: Using directory '{output_dir}'")
        return output_dir, base_filename
    
    def _create_chart(self, chart_path: str, speeds: List[float], times: List[float],
                     session_id: int, base_filename: str, avg_speed: float, 
                     max_speed: float, min_speed: float) -> None:
        """創建速度圖表"""
        plt.figure(figsize=(12, 7))
        plt.plot(times, speeds, 'o-', linewidth=2, markersize=6, label='Speed (km/h)')
        plt.axhline(y=avg_speed, color='r', linestyle='--', 
                   label=f'Avg: {avg_speed:.1f} km/h')
        
        # 添加數值標註
        for t, s in zip(times, speeds):
            plt.annotate(f"{s:.1f}", (t, s), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8)
        
        plt.title(f'Net Crossing Speeds - Session {session_id} - File: {base_filename}', 
                 fontsize=16)
        plt.xlabel('Relative Time (s)', fontsize=12)
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        
        # 設置座標軸範圍
        if times:
            x_margin = (max(times) - min(times)) * 0.05 if len(times) > 1 else 0.5
            plt.xlim(min(times) - x_margin, max(times) + x_margin)
        
        if speeds:
            y_range = max_speed - min_speed if max_speed > min_speed else 10
            plt.ylim(max(0, min_speed - y_range * 0.1), max_speed + y_range * 0.1)
        
        # 添加統計信息
        plt.figtext(0.02, 0.02, 
                   f"Count: {len(speeds)}, Max: {max_speed:.1f}, Min: {min_speed:.1f} km/h", 
                   fontsize=9)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_path, dpi=150)
        plt.close()
    
    def _create_text_file(self, txt_path: str, speeds: List[float], times: List[float],
                         session_id: int, base_filename: str, avg_speed: float, 
                         max_speed: float, min_speed: float) -> None:
        """創建文本文件"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Net Speeds - Session {session_id} - File: {base_filename}\n")  
            f.write("---------------------------------------\n")
            
            for t, s in zip(times, speeds):
                f.write(f"{t:.2f}s: {s:.1f} km/h\n")
            
            f.write("---------------------------------------\n")
            f.write(f"Total Points: {len(speeds)}\n")
            f.write(f"Average Speed: {avg_speed:.1f} km/h\n")
            f.write(f"Maximum Speed: {max_speed:.1f} km/h\n")
            f.write(f"Minimum Speed: {min_speed:.1f} km/h\n")
    
    def _create_csv_file(self, csv_path: str, speeds: List[float], times: List[float],
                        session_id: int, base_filename: str, avg_speed: float, 
                        max_speed: float, min_speed: float) -> None:
        """創建CSV文件"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 寫入數據
            writer.writerow(['Session ID', 'File Prefix', 'Point Number', 
                           'Relative Time (s)', 'Speed (km/h)'])
            
            for i, (t, s) in enumerate(zip(times, speeds)):
                writer.writerow([session_id, base_filename, i + 1, 
                               f"{t:.2f}", f"{s:.1f}"])
            
            # 寫入統計信息
            writer.writerow([])
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Total Points', len(speeds)])
            writer.writerow(['Average Speed (km/h)', f"{avg_speed:.1f}"])
            writer.writerow(['Maximum Speed (km/h)', f"{max_speed:.1f}"])
            writer.writerow(['Minimum Speed (km/h)', f"{min_speed:.1f}"])
    
    def shutdown(self) -> None:
        """關閉輸出器"""
        self.executor.shutdown(wait=True)