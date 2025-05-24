#!/usr/bin/env python3
# 乒乓球速度追蹤系統 GUI版本
# 專為M2 Pro MacBook Pro優化

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
import time
import datetime
from collections import deque
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import csv
import threading
import queue
import concurrent.futures
from PIL import Image, ImageTk
import json

# 導入原有的追蹤器類
from real_time_v14 import PingPongSpeedTracker, FrameData

class PingPongGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("乒乓球速度追蹤系統 - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # 設置macOS風格
        self.setup_macos_style()
        
        # 初始化變量
        self.tracker = None
        self.is_running = False
        self.is_paused = False
        self.current_frame = None
        self.video_label = None
        
        # 創建UI組件
        self.create_widgets()
        self.load_default_settings()
        
        # 開始UI更新循環
        self.update_display()
        
    def setup_macos_style(self):
        """設置macOS風格的主題"""
        style = ttk.Style()
        style.theme_use('aqua')  # macOS專用主題
        
        # 自定義顏色
        self.colors = {
            'bg_primary': '#2b2b2b',
            'bg_secondary': '#3b3b3b',
            'bg_accent': '#0078d4',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        }
        
    def create_widgets(self):
        """創建所有UI組件"""
        # 主容器
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側面板 - 視頻顯示
        self.create_video_panel(main_frame)
        
        # 右側面板 - 控制和數據
        self.create_control_panel(main_frame)
        
        # 底部狀態欄
        self.create_status_bar(main_frame)
        
    def create_video_panel(self, parent):
        """創建視頻顯示面板"""
        video_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 視頻標題
        video_title = tk.Label(video_frame, text="實時視頻監控", 
                              bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                              font=('SF Pro Display', 16, 'bold'))
        video_title.pack(pady=10)
        
        # 視頻顯示區域
        self.video_label = tk.Label(video_frame, bg='black', text="等待視頻輸入...",
                                   fg='white', font=('SF Pro Display', 14))
        self.video_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # 視頻控制按鈕
        video_controls = tk.Frame(video_frame, bg=self.colors['bg_secondary'])
        video_controls.pack(pady=10)
        
        self.start_btn = tk.Button(video_controls, text="▶ 開始", 
                                  command=self.toggle_tracking,
                                  bg=self.colors['success'], fg='black',
                                  font=('SF Pro Display', 12, 'bold'),
                                  padx=20, pady=8)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(video_controls, text="⏸ 暫停", 
                                  command=self.pause_tracking,
                                  bg=self.colors['warning'], fg='black',
                                  font=('SF Pro Display', 12, 'bold'),
                                  padx=20, pady=8, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(video_controls, text="⏹ 停止", 
                                 command=self.stop_tracking,
                                 bg=self.colors['danger'], fg='black',
                                 font=('SF Pro Display', 12, 'bold'),
                                 padx=20, pady=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 幫助按鈕
        help_btn = tk.Button(video_controls, text="❓ 幫助", 
                            command=self.show_usage_guide,
                            bg='#6c757d', fg='black',
                            font=('SF Pro Display', 12, 'bold'),
                            padx=20, pady=8)
        help_btn.pack(side=tk.LEFT, padx=5)
        
    def create_control_panel(self, parent):
        """創建控制面板"""
        control_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        control_frame.configure(width=400)
        
        # 使用Notebook創建標籤頁
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 設置標籤頁
        self.create_settings_tab(notebook)
        self.create_monitor_tab(notebook)
        self.create_data_tab(notebook)
        self.create_output_tab(notebook)
        
    def create_settings_tab(self, notebook):
        """創建設置標籤頁"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="設置")
        
        # 視頻源設置
        video_group = ttk.LabelFrame(settings_frame, text="視頻源設置", padding=10)
        video_group.pack(fill=tk.X, pady=5)
        
        # 攝像頭/文件選擇
        tk.Label(video_group, text="視頻源類型:").pack(anchor=tk.W)
        self.source_var = tk.StringVar(value="camera")
        tk.Radiobutton(video_group, text="攝像頭", variable=self.source_var, 
                      value="camera").pack(anchor=tk.W)
        tk.Radiobutton(video_group, text="視頻文件", variable=self.source_var, 
                      value="file").pack(anchor=tk.W)
        
        # 攝像頭索引
        cam_frame = tk.Frame(video_group)
        cam_frame.pack(fill=tk.X, pady=5)
        tk.Label(cam_frame, text="攝像頭索引:").pack(side=tk.LEFT)
        self.camera_idx_var = tk.StringVar(value="0")
        tk.Entry(cam_frame, textvariable=self.camera_idx_var, width=10).pack(side=tk.RIGHT)
        
        # 視頻文件路徑
        file_frame = tk.Frame(video_group)
        file_frame.pack(fill=tk.X, pady=5)
        tk.Label(file_frame, text="視頻文件:").pack(anchor=tk.W)
        self.video_path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.video_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(file_frame, text="瀏覽", command=self.browse_video_file).pack(side=tk.RIGHT)
        
        # 視頻參數設置
        video_params = ttk.LabelFrame(settings_frame, text="視頻參數", padding=10)
        video_params.pack(fill=tk.X, pady=5)
        
        # FPS設置
        fps_frame = tk.Frame(video_params)
        fps_frame.pack(fill=tk.X, pady=2)
        tk.Label(fps_frame, text="目標FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="60")
        tk.Entry(fps_frame, textvariable=self.fps_var, width=10).pack(side=tk.RIGHT)
        
        # 解析度設置
        res_frame = tk.Frame(video_params)
        res_frame.pack(fill=tk.X, pady=2)
        tk.Label(res_frame, text="解析度:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="1280")
        self.height_var = tk.StringVar(value="720")
        tk.Entry(res_frame, textvariable=self.width_var, width=8).pack(side=tk.RIGHT)
        tk.Label(res_frame, text="x").pack(side=tk.RIGHT)
        tk.Entry(res_frame, textvariable=self.height_var, width=8).pack(side=tk.RIGHT)
        
        # 追蹤參數設置
        tracking_params = ttk.LabelFrame(settings_frame, text="追蹤參數", padding=10)
        tracking_params.pack(fill=tk.X, pady=5)
        
        # 桌子長度
        table_frame = tk.Frame(tracking_params)
        table_frame.pack(fill=tk.X, pady=2)
        tk.Label(table_frame, text="桌子長度(cm):").pack(side=tk.LEFT)
        self.table_length_var = tk.StringVar(value="94")
        tk.Entry(table_frame, textvariable=self.table_length_var, width=10).pack(side=tk.RIGHT)
        
        # 檢測超時
        timeout_frame = tk.Frame(tracking_params)
        timeout_frame.pack(fill=tk.X, pady=2)
        tk.Label(timeout_frame, text="檢測超時(s):").pack(side=tk.LEFT)
        self.timeout_var = tk.StringVar(value="0.3")
        tk.Entry(timeout_frame, textvariable=self.timeout_var, width=10).pack(side=tk.RIGHT)
        
        # 穿越方向
        direction_frame = tk.Frame(tracking_params)
        direction_frame.pack(fill=tk.X, pady=2)
        tk.Label(direction_frame, text="穿越方向:").pack(side=tk.LEFT)
        self.direction_var = tk.StringVar(value="right_to_left")
        direction_combo = ttk.Combobox(direction_frame, textvariable=self.direction_var,
                                      values=["left_to_right", "right_to_left", "both"],
                                      width=12, state="readonly")
        direction_combo.pack(side=tk.RIGHT)
        
        # 收集數量
        count_frame = tk.Frame(tracking_params)
        count_frame.pack(fill=tk.X, pady=2)
        tk.Label(count_frame, text="收集速度數量:").pack(side=tk.LEFT)
        self.max_speeds_var = tk.StringVar(value="30")
        tk.Entry(count_frame, textvariable=self.max_speeds_var, width=10).pack(side=tk.RIGHT)
        
        # 調試模式
        self.debug_var = tk.BooleanVar()
        tk.Checkbutton(tracking_params, text="啟用調試模式", 
                      variable=self.debug_var).pack(anchor=tk.W, pady=5)
        
    def create_monitor_tab(self, notebook):
        """創建監控標籤頁"""
        monitor_frame = ttk.Frame(notebook)
        notebook.add(monitor_frame, text="監控")
        
        # 實時數據顯示
        data_group = ttk.LabelFrame(monitor_frame, text="實時數據", padding=10)
        data_group.pack(fill=tk.X, pady=5)
        
        # 當前速度
        speed_frame = tk.Frame(data_group)
        speed_frame.pack(fill=tk.X, pady=2)
        tk.Label(speed_frame, text="當前速度:", font=('SF Pro Display', 12)).pack(side=tk.LEFT)
        self.current_speed_label = tk.Label(speed_frame, text="0.0 km/h", 
                                           font=('SF Pro Display', 12, 'bold'),
                                           fg=self.colors['success'])
        self.current_speed_label.pack(side=tk.RIGHT)
        
        # FPS
        fps_frame = tk.Frame(data_group)
        fps_frame.pack(fill=tk.X, pady=2)
        tk.Label(fps_frame, text="FPS:", font=('SF Pro Display', 12)).pack(side=tk.LEFT)
        self.fps_label = tk.Label(fps_frame, text="0.0", 
                                 font=('SF Pro Display', 12, 'bold'))
        self.fps_label.pack(side=tk.RIGHT)
        
        # 計數狀態
        counting_frame = tk.Frame(data_group)
        counting_frame.pack(fill=tk.X, pady=2)
        tk.Label(counting_frame, text="計數狀態:", font=('SF Pro Display', 12)).pack(side=tk.LEFT)
        self.counting_label = tk.Label(counting_frame, text="關閉", 
                                      font=('SF Pro Display', 12, 'bold'),
                                      fg=self.colors['danger'])
        self.counting_label.pack(side=tk.RIGHT)
        
        # 已記錄數量
        recorded_frame = tk.Frame(data_group)
        recorded_frame.pack(fill=tk.X, pady=2)
        tk.Label(recorded_frame, text="已記錄:", font=('SF Pro Display', 12)).pack(side=tk.LEFT)
        self.recorded_label = tk.Label(recorded_frame, text="0/30", 
                                      font=('SF Pro Display', 12, 'bold'))
        self.recorded_label.pack(side=tk.RIGHT)
        
        # 最後記錄速度
        last_speed_frame = tk.Frame(data_group)
        last_speed_frame.pack(fill=tk.X, pady=2)
        tk.Label(last_speed_frame, text="最後記錄:", font=('SF Pro Display', 12)).pack(side=tk.LEFT)
        self.last_speed_label = tk.Label(last_speed_frame, text="0.0 km/h", 
                                        font=('SF Pro Display', 12, 'bold'))
        self.last_speed_label.pack(side=tk.RIGHT)
        
        # 控制按鈕
        control_group = ttk.LabelFrame(monitor_frame, text="計數控制", padding=10)
        control_group.pack(fill=tk.X, pady=5)
        
        # 操作提示
        instruction_label = tk.Label(control_group, text="操作提示：先點擊左側'開始'啟動追蹤，再點擊下方'開始計數'記錄速度", 
                                   font=('SF Pro Display', 9), fg='gray', wraplength=300)
        instruction_label.pack(pady=(0, 10))
        
        self.toggle_counting_btn = tk.Button(control_group, text="開始計數", 
                                           command=self.toggle_counting,
                                           bg=self.colors['success'], fg='black',
                                           font=('SF Pro Display', 12, 'bold'),
                                           width=15)
        self.toggle_counting_btn.pack(pady=5)
        
        # 調試信息
        debug_group = ttk.LabelFrame(monitor_frame, text="調試信息", padding=10)
        debug_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.debug_text = scrolledtext.ScrolledText(debug_group, height=8, 
                                                   font=('Monaco', 10))
        self.debug_text.pack(fill=tk.BOTH, expand=True)
        
    def create_data_tab(self, notebook):
        """創建數據標籤頁"""
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="數據")
        
        # 實時圖表
        chart_group = ttk.LabelFrame(data_frame, text="速度圖表", padding=10)
        chart_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 創建matplotlib圖表
        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-time Speed Records")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Speed (km/h)")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_group)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 統計信息
        stats_group = ttk.LabelFrame(data_frame, text="統計信息", padding=10)
        stats_group.pack(fill=tk.X, pady=5)
        
        stats_frame = tk.Frame(stats_group)
        stats_frame.pack(fill=tk.X)
        
        # 左列
        left_stats = tk.Frame(stats_frame)
        left_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_stats, text="平均速度:").pack(anchor=tk.W)
        self.avg_speed_label = tk.Label(left_stats, text="0.0 km/h", font=('SF Pro Display', 10, 'bold'))
        self.avg_speed_label.pack(anchor=tk.W)
        
        tk.Label(left_stats, text="最大速度:").pack(anchor=tk.W)
        self.max_speed_label = tk.Label(left_stats, text="0.0 km/h", font=('SF Pro Display', 10, 'bold'))
        self.max_speed_label.pack(anchor=tk.W)
        
        # 右列
        right_stats = tk.Frame(stats_frame)
        right_stats.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_stats, text="最小速度:").pack(anchor=tk.W)
        self.min_speed_label = tk.Label(right_stats, text="0.0 km/h", font=('SF Pro Display', 10, 'bold'))
        self.min_speed_label.pack(anchor=tk.W)
        
        tk.Label(right_stats, text="標準差:").pack(anchor=tk.W)
        self.std_speed_label = tk.Label(right_stats, text="0.0 km/h", font=('SF Pro Display', 10, 'bold'))
        self.std_speed_label.pack(anchor=tk.W)
        
    def create_output_tab(self, notebook):
        """創建輸出標籤頁"""
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="輸出")
        
        # 輸出設置
        output_group = ttk.LabelFrame(output_frame, text="輸出設置", padding=10)
        output_group.pack(fill=tk.X, pady=5)
        
        # 輸出路徑
        path_frame = tk.Frame(output_group)
        path_frame.pack(fill=tk.X, pady=5)
        tk.Label(path_frame, text="輸出路徑:").pack(anchor=tk.W)
        self.output_path_var = tk.StringVar(value="./real_time_output")
        tk.Entry(path_frame, textvariable=self.output_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(path_frame, text="瀏覽", command=self.browse_output_path).pack(side=tk.RIGHT)
        
        # 輸出格式
        format_frame = tk.Frame(output_group)
        format_frame.pack(fill=tk.X, pady=5)
        tk.Label(format_frame, text="輸出格式:").pack(anchor=tk.W)
        
        self.export_png_var = tk.BooleanVar(value=True)
        self.export_csv_var = tk.BooleanVar(value=True)
        self.export_txt_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(format_frame, text="PNG圖表", variable=self.export_png_var).pack(anchor=tk.W)
        tk.Checkbutton(format_frame, text="CSV數據", variable=self.export_csv_var).pack(anchor=tk.W)
        tk.Checkbutton(format_frame, text="TXT報告", variable=self.export_txt_var).pack(anchor=tk.W)
        
        # 手動輸出按鈕
        manual_output_group = ttk.LabelFrame(output_frame, text="手動輸出", padding=10)
        manual_output_group.pack(fill=tk.X, pady=5)
        
        tk.Button(manual_output_group, text="立即輸出當前數據", 
                 command=self.manual_export,
                 bg=self.colors['bg_accent'], fg='black',
                 font=('SF Pro Display', 12, 'bold'),
                 width=20).pack(pady=10)
        
        # 輸出歷史
        history_group = ttk.LabelFrame(output_frame, text="輸出歷史", padding=10)
        history_group.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 輸出歷史列表
        self.history_listbox = tk.Listbox(history_group, height=6)
        self.history_listbox.pack(fill=tk.BOTH, expand=True)
        
        # 打開輸出文件夾按鈕
        tk.Button(history_group, text="打開輸出文件夾", 
                 command=self.open_output_folder,
                 font=('SF Pro Display', 10)).pack(pady=5)
        
    def create_status_bar(self, parent):
        """創建狀態欄"""
        self.status_bar = tk.Frame(parent, bg=self.colors['bg_secondary'], height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        self.status_label = tk.Label(self.status_bar, text="就緒", 
                                    bg=self.colors['bg_secondary'], 
                                    fg=self.colors['text_secondary'],
                                    font=('SF Pro Display', 10))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 時間顯示
        self.time_label = tk.Label(self.status_bar, text="", 
                                  bg=self.colors['bg_secondary'], 
                                  fg=self.colors['text_secondary'],
                                  font=('SF Pro Display', 10))
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
    def browse_video_file(self):
        """瀏覽視頻文件"""
        file_path = filedialog.askopenfilename(
            title="選擇視頻文件",
            filetypes=[("視頻文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            
    def browse_output_path(self):
        """瀏覽輸出路徑"""
        folder_path = filedialog.askdirectory(title="選擇輸出文件夾")
        if folder_path:
            self.output_path_var.set(folder_path)
            
    def load_default_settings(self):
        """載入默認設置"""
        try:
            if os.path.exists("pingpong_settings.json"):
                with open("pingpong_settings.json", "r") as f:
                    settings = json.load(f)
                    self.apply_settings(settings)
        except Exception as e:
            self.debug_print(f"載入設置失敗: {e}")
            
    def save_settings(self):
        """保存設置"""
        settings = {
            "source_type": self.source_var.get(),
            "camera_idx": self.camera_idx_var.get(),
            "video_path": self.video_path_var.get(),
            "fps": self.fps_var.get(),
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "table_length": self.table_length_var.get(),
            "timeout": self.timeout_var.get(),
            "direction": self.direction_var.get(),
            "max_speeds": self.max_speeds_var.get(),
            "debug": self.debug_var.get(),
            "output_path": self.output_path_var.get()
        }
        try:
            with open("pingpong_settings.json", "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            self.debug_print(f"保存設置失敗: {e}")
            
    def apply_settings(self, settings):
        """應用設置"""
        try:
            self.source_var.set(settings.get("source_type", "camera"))
            self.camera_idx_var.set(settings.get("camera_idx", "0"))
            self.video_path_var.set(settings.get("video_path", ""))
            self.fps_var.set(settings.get("fps", "60"))
            self.width_var.set(settings.get("width", "1280"))
            self.height_var.set(settings.get("height", "720"))
            self.table_length_var.set(settings.get("table_length", "94"))
            self.timeout_var.set(settings.get("timeout", "0.3"))
            self.direction_var.set(settings.get("direction", "right_to_left"))
            self.max_speeds_var.set(settings.get("max_speeds", "30"))
            self.debug_var.set(settings.get("debug", False))
            self.output_path_var.set(settings.get("output_path", "./real_time_output"))
        except Exception as e:
            self.debug_print(f"應用設置失敗: {e}")
            
    def toggle_tracking(self):
        """開始/停止追蹤"""
        if not self.is_running:
            self.start_tracking()
        else:
            self.stop_tracking()
            
    def start_tracking(self):
        """開始追蹤"""
        try:
            # 創建追蹤器
            self.create_tracker()
            
            if self.tracker is None:
                messagebox.showerror("錯誤", "無法創建追蹤器，請檢查設置")
                return
                
            # 啟動追蹤線程
            self.is_running = True
            self.is_paused = False
            self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
            self.tracking_thread.start()
            
            # 更新按鈕狀態
            self.start_btn.config(text="🔄 運行中", state=tk.DISABLED, fg='black')
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text="追蹤已開始", fg=self.colors['success'])
            self.debug_print("追蹤系統已啟動")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"啟動追蹤失敗: {e}")
            self.debug_print(f"啟動失敗: {e}")
            
    def pause_tracking(self):
        """暫停/恢復追蹤"""
        if self.is_running:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.pause_btn.config(text="▶ 恢復", fg='black')
                self.status_label.config(text="已暫停", fg=self.colors['warning'])
            else:
                self.pause_btn.config(text="⏸ 暫停", fg='black')
                self.status_label.config(text="追蹤中", fg=self.colors['success'])
                
    def stop_tracking(self):
        """停止追蹤"""
        try:
            self.is_running = False
            self.is_paused = False
            
            if self.tracker:
                self.tracker.running = False
                self.tracker.reader.stop()
                
            # 更新按鈕狀態
            self.start_btn.config(text="▶ 開始", state=tk.NORMAL, fg='black')
            self.pause_btn.config(text="⏸ 暫停", state=tk.DISABLED, fg='black')
            self.stop_btn.config(state=tk.DISABLED)
            
            # 清除視頻顯示
            self.video_label.config(image="", text="等待視頻輸入...")
            
            self.status_label.config(text="已停止", fg=self.colors['danger'])
            self.debug_print("追蹤系統已停止")
            
        except Exception as e:
            self.debug_print(f"停止失敗: {e}")
            
    def create_tracker(self):
        """創建追蹤器實例"""
        try:
            # 獲取設置參數
            if self.source_var.get() == "camera":
                video_source = int(self.camera_idx_var.get())
                use_video_file = False
            else:
                video_source = self.video_path_var.get()
                use_video_file = True
                if not os.path.exists(video_source):
                    raise ValueError("視頻文件不存在")
                    
            # 創建追蹤器
            self.tracker = PingPongSpeedTracker(
                video_source=video_source,
                table_length_cm=float(self.table_length_var.get()),
                detection_timeout_s=float(self.timeout_var.get()),
                use_video_file=use_video_file,
                target_fps=int(self.fps_var.get()),
                frame_width=int(self.width_var.get()),
                frame_height=int(self.height_var.get()),
                debug_mode=self.debug_var.get(),
                net_crossing_direction=self.direction_var.get(),
                max_net_speeds=int(self.max_speeds_var.get())
            )
            
            # 啟動追蹤器的讀取器
            self.tracker.reader.start()
            
        except Exception as e:
            self.tracker = None
            raise e
            
    def tracking_loop(self):
        """追蹤主循環"""
        try:
            while self.is_running and self.tracker:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                    
                # 讀取幀
                ret, frame = self.tracker.reader.read()
                if not ret or frame is None:
                    self.debug_print("視頻結束或讀取失敗")
                    break
                    
                # 處理幀
                frame_data = self.tracker.process_single_frame(frame)
                
                # 保存當前幀用於顯示
                self.current_frame = frame_data
                
                # 添加小延遲避免CPU過載
                time.sleep(0.01)
                
        except Exception as e:
            self.debug_print(f"追蹤循環錯誤: {e}")
        finally:
            self.root.after(0, self.stop_tracking)
            
    def toggle_counting(self):
        """開始/停止計數"""
        if self.tracker and self.is_running:
            self.tracker.toggle_counting()
            
            if self.tracker.is_counting_active:
                self.toggle_counting_btn.config(text="停止計數", bg=self.colors['danger'], fg='black')
                self.counting_label.config(text="開啟", fg=self.colors['success'])
                self.debug_print("計數已開始")
            else:
                self.toggle_counting_btn.config(text="開始計數", bg=self.colors['success'], fg='black')
                self.counting_label.config(text="關閉", fg=self.colors['danger'])
                self.debug_print("計數已停止")
        else:
            messagebox.showwarning("警告", "請先啟動追蹤系統")
            
    def manual_export(self):
        """手動導出數據 - 使用原有追蹤器的輸出格式"""
        if self.tracker and hasattr(self.tracker, 'collected_net_speeds') and self.tracker.collected_net_speeds:
            try:
                # 直接調用原追蹤器的輸出方法，保持文件格式完全一致
                self.tracker._generate_outputs_async()
                self.debug_print("手動導出已啟動")
                
                # 添加到歷史記錄
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                export_info = f"{timestamp} - 手動導出 ({len(self.tracker.collected_net_speeds)} 個數據點)"
                self.history_listbox.insert(0, export_info)
                
                messagebox.showinfo("成功", 
                    f"數據導出已開始，將生成以下文件：\n"
                    f"• PNG圖表文件 (speed_chart_*.png)\n"
                    f"• CSV數據文件 (speed_data_*.csv)\n" 
                    f"• TXT報告文件 (speed_data_*.txt)\n\n"
                    f"請查看輸出文件夾：{self.output_path_var.get()}")
            except Exception as e:
                messagebox.showerror("錯誤", f"導出失敗: {e}")
                self.debug_print(f"導出失敗: {e}")
        else:
            messagebox.showwarning("警告", "沒有可導出的數據\n請先啟動追蹤並開始計數")
            
    def open_output_folder(self):
        """打開輸出文件夾"""
        output_path = self.output_path_var.get()
        if os.path.exists(output_path):
            os.system(f"open '{output_path}'")  # macOS命令
        else:
            messagebox.showwarning("警告", "輸出文件夾不存在")
            
    def show_usage_guide(self):
        """顯示使用說明"""
        guide_text = """🏓 乒乓球速度追蹤系統 - 使用指南

📋 基本操作流程：

1️⃣ 設置視頻源
   • 選擇攝像頭或視頻文件
   • 調整追蹤參數（如需要）

2️⃣ 啟動追蹤系統  
   • 點擊左側視頻區域的 "▶ 開始" 按鈕
   • 確認視頻畫面正常顯示

3️⃣ 開始記錄速度
   • 切換到 "監控" 標籤頁
   • 點擊 "開始計數" 按鈕
   • 系統會自動記錄穿越中線的球速

4️⃣ 監控和輸出
   • 在 "數據" 頁面查看實時圖表
   • 達到設定數量會自動輸出文件
   • 也可在 "輸出" 頁面手動導出

⚠️ 注意事項：
   • 確保乒乓球在ROI區域內
   • 建議使用高幀率攝像頭（60fps+）
   • 輸出文件格式：PNG圖表 + CSV數據 + TXT報告"""
        
        messagebox.showinfo("使用指南", guide_text)
            
    def update_display(self):
        """更新顯示"""
        try:
            # 更新時間
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.config(text=current_time)
            
            # 更新視頻顯示和數據
            if self.current_frame and self.is_running and not self.is_paused:
                self.update_video_display()
                self.update_data_display()
                
        except Exception as e:
            self.debug_print(f"顯示更新錯誤: {e}")
        finally:
            # 每33ms更新一次 (約30fps)
            self.root.after(33, self.update_display)
            
    def update_video_display(self):
        """更新視頻顯示"""
        try:
            if self.current_frame and self.current_frame.frame is not None:
                # 繪製可視化
                display_frame = self.tracker._draw_visualizations(
                    self.current_frame.frame.copy(), self.current_frame
                )
                
                # 調整大小適應顯示區域
                display_height = 480
                aspect_ratio = display_frame.shape[1] / display_frame.shape[0]
                display_width = int(display_height * aspect_ratio)
                
                resized_frame = cv2.resize(display_frame, (display_width, display_height))
                
                # 轉換為PIL格式
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新顯示
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # 保持引用
                
        except Exception as e:
            self.debug_print(f"視頻顯示更新錯誤: {e}")
            
    def update_data_display(self):
        """更新數據顯示"""
        try:
            if self.current_frame:
                # 更新實時數據標籤
                self.current_speed_label.config(text=f"{self.current_frame.current_ball_speed_kmh:.1f} km/h")
                self.fps_label.config(text=f"{self.current_frame.display_fps:.1f}")
                
                max_speeds = int(self.max_speeds_var.get())
                recorded_count = len(self.current_frame.collected_net_speeds)
                self.recorded_label.config(text=f"{recorded_count}/{max_speeds}")
                
                if self.current_frame.last_recorded_net_speed_kmh > 0:
                    self.last_speed_label.config(text=f"{self.current_frame.last_recorded_net_speed_kmh:.1f} km/h")
                
                # 更新統計信息
                if self.current_frame.collected_net_speeds:
                    speeds = self.current_frame.collected_net_speeds
                    avg_speed = sum(speeds) / len(speeds)
                    max_speed = max(speeds)
                    min_speed = min(speeds)
                    std_speed = np.std(speeds) if len(speeds) > 1 else 0
                    
                    self.avg_speed_label.config(text=f"{avg_speed:.1f} km/h")
                    self.max_speed_label.config(text=f"{max_speed:.1f} km/h")
                    self.min_speed_label.config(text=f"{min_speed:.1f} km/h")
                    self.std_speed_label.config(text=f"{std_speed:.1f} km/h")
                    
                    # 更新圖表
                    self.update_chart()
                    
        except Exception as e:
            self.debug_print(f"數據顯示更新錯誤: {e}")
            
    def update_chart(self):
        """更新速度圖表"""
        try:
            if self.current_frame and self.current_frame.collected_net_speeds:
                speeds = self.current_frame.collected_net_speeds
                times = self.current_frame.collected_relative_times
                
                self.ax.clear()
                self.ax.plot(times, speeds, 'o-', linewidth=2, markersize=6, color='#0078d4')
                
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    self.ax.axhline(y=avg_speed, color='red', linestyle='--', 
                                   label=f'Avg: {avg_speed:.1f} km/h', alpha=0.7)
                
                self.ax.set_title("Real-time Speed Records", fontsize=12)
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Speed (km/h)")
                self.ax.grid(True, alpha=0.3)
                self.ax.legend()
                
                # 設置坐標軸範圍
                if times:
                    x_margin = (max(times) - min(times)) * 0.05 if len(times) > 1 else 0.5
                    self.ax.set_xlim(min(times) - x_margin, max(times) + x_margin)
                
                if speeds:
                    y_range = max(speeds) - min(speeds) if len(speeds) > 1 else 10
                    self.ax.set_ylim(max(0, min(speeds) - y_range * 0.1), 
                                    max(speeds) + y_range * 0.1)
                
                self.canvas.draw()
                
        except Exception as e:
            self.debug_print(f"圖表更新錯誤: {e}")
            
    def debug_print(self, message):
        """在調試區域顯示消息"""
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            self.debug_text.insert(tk.END, formatted_message)
            self.debug_text.see(tk.END)
            
            # 限制調試文本長度
            lines = self.debug_text.get("1.0", tk.END).split("\n")
            if len(lines) > 100:
                self.debug_text.delete("1.0", f"{len(lines)-100}.0")
                
        except Exception as e:
            print(f"調試輸出錯誤: {e}")
            
    def on_closing(self):
        """關閉程序時的清理工作"""
        try:
            # 保存設置
            self.save_settings()
            
            # 停止追蹤
            if self.is_running:
                self.stop_tracking()
                
            # 等待線程結束
            if hasattr(self, 'tracking_thread') and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=2)
                
            self.root.destroy()
            
        except Exception as e:
            print(f"關閉時錯誤: {e}")
            self.root.destroy()

def main():
    """主函數"""
    # 檢查是否可以導入原始追蹤器
    try:
        from real_time_v14 import PingPongSpeedTracker
    except ImportError:
        print("錯誤: 無法導入 real_time_v14.py")
        print("請確保 real_time_v14.py 文件在同一目錄下")
        return
        
    # 創建主窗口
    root = tk.Tk()
    
    # 設置macOS特定選項
    try:
        # 設置程序圖標（如果有的話）
        # root.iconbitmap('icon.icns')
        
        # 設置macOS窗口樣式
        root.tk.call('tk', 'scaling', 2.0)  # 適應高DPI顯示器
    except:
        pass
    
    # 創建GUI應用
    app = PingPongGUI(root)
    
    # 設置關閉事件
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 運行主循環
    root.mainloop()

if __name__ == "__main__":
    main()