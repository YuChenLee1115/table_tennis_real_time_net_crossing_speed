#!/usr/bin/env python3
# 乒乓球速度追蹤系統 GUI 介面
# 基於 real_time_v14.py 的圖形化使用者介面

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import os
import time
import json
from datetime import datetime
import numpy as np

# 嘗試導入 customtkinter 以獲得更好的外觀
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    USE_CTK = True
except ImportError:
    USE_CTK = False
    print("CustomTkinter not found. Using standard tkinter.")

# 導入原始追蹤系統
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 假設原始程式碼在同一目錄下，命名為 real_time_v14.py
try:
    from real_time_v14 import PingPongSpeedTracker, FrameData
    from real_time_v14 import (DEFAULT_CAMERA_INDEX, DEFAULT_TARGET_FPS, 
                               DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT,
                               DEFAULT_TABLE_LENGTH_CM, DEFAULT_DETECTION_TIMEOUT,
                               MAX_NET_SPEEDS_TO_COLLECT, NET_CROSSING_DIRECTION_DEFAULT,
                               NEAR_SIDE_WIDTH_CM_DEFAULT, FAR_SIDE_WIDTH_CM_DEFAULT)
except ImportError:
    messagebox.showerror("錯誤", "無法導入 real_time_v14.py\n請確保檔案在同一目錄下")
    sys.exit(1)

class PingPongTrackerGUI:
    def __init__(self):
        # 建立主視窗
        if USE_CTK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()
        
        self.root.title("乒乓球速度追蹤系統 v14 GUI")
        self.root.geometry("1400x900")
        
        # 設定變數
        self.setup_variables()
        
        # 建立 GUI 元件
        self.create_widgets()
        
        # 追蹤器相關
        self.tracker = None
        self.tracking_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.is_tracking = False
        
        # 設定視窗關閉處理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 開始 GUI 更新循環
        self.update_gui()
        
    def setup_variables(self):
        """設定所有 GUI 變數"""
        # 輸入源選擇
        self.source_type = tk.StringVar(value="camera")
        self.video_path = tk.StringVar(value="")
        self.camera_index = tk.IntVar(value=DEFAULT_CAMERA_INDEX)
        
        # 基本參數
        self.target_fps = tk.IntVar(value=DEFAULT_TARGET_FPS)
        self.frame_width = tk.IntVar(value=DEFAULT_FRAME_WIDTH)
        self.frame_height = tk.IntVar(value=DEFAULT_FRAME_HEIGHT)
        self.table_length = tk.DoubleVar(value=DEFAULT_TABLE_LENGTH_CM)
        self.detection_timeout = tk.DoubleVar(value=DEFAULT_DETECTION_TIMEOUT)
        
        # 進階參數
        self.net_direction = tk.StringVar(value=NET_CROSSING_DIRECTION_DEFAULT)
        self.max_speeds = tk.IntVar(value=MAX_NET_SPEEDS_TO_COLLECT)
        self.near_width = tk.DoubleVar(value=NEAR_SIDE_WIDTH_CM_DEFAULT)
        self.far_width = tk.DoubleVar(value=FAR_SIDE_WIDTH_CM_DEFAULT)
        self.debug_mode = tk.BooleanVar(value=False)
        
        # 狀態變數
        self.is_counting = tk.BooleanVar(value=False)
        self.current_speed = tk.StringVar(value="0.0")
        self.current_fps = tk.StringVar(value="0.0")
        self.speeds_collected = tk.StringVar(value="0/0")
        self.last_net_speed = tk.StringVar(value="0.0")
        
        # 統計數據
        self.stats_text = tk.StringVar(value="尚無數據")
        
    def create_widgets(self):
        """建立所有 GUI 元件"""
        # 主要容器
        if USE_CTK:
            main_container = ctk.CTkFrame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            main_container = ttk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側控制面板
        self.create_control_panel(main_container)
        
        # 右側視訊顯示
        self.create_video_panel(main_container)
        
        # 底部狀態列
        self.create_status_bar()
        
    def create_control_panel(self, parent):
        """建立控制面板"""
        if USE_CTK:
            control_frame = ctk.CTkFrame(parent)
            control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            
            # 標題
            title_label = ctk.CTkLabel(control_frame, text="控制面板", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
            title_label.pack(pady=10)
            
            # 使用 CTkTabview
            tabview = ctk.CTkTabview(control_frame, width=350)
            tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        else:
            control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
            control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            
            # 使用標準 Notebook
            tabview = ttk.Notebook(control_frame)
            tabview.pack(fill=tk.BOTH, expand=True)
        
        # 建立分頁
        if USE_CTK:
            tab1 = tabview.add("輸入源")
            tab2 = tabview.add("參數設定")
            tab3 = tabview.add("統計數據")
        else:
            tab1 = ttk.Frame(tabview)
            tab2 = ttk.Frame(tabview)
            tab3 = ttk.Frame(tabview)
            tabview.add(tab1, text="輸入源")
            tabview.add(tab2, text="參數設定")
            tabview.add(tab3, text="統計數據")
        
        # 填充分頁內容
        self.create_source_tab(tab1)
        self.create_params_tab(tab2)
        self.create_stats_tab(tab3)
        
        # 控制按鈕
        self.create_control_buttons(control_frame)
        
    def create_source_tab(self, parent):
        """建立輸入源選擇分頁"""
        # 輸入源類型選擇
        if USE_CTK:
            source_label = ctk.CTkLabel(parent, text="選擇輸入源：")
            source_label.pack(pady=(10, 5))
            
            camera_radio = ctk.CTkRadioButton(parent, text="即時攝影機", 
                                            variable=self.source_type, 
                                            value="camera",
                                            command=self.on_source_change)
            camera_radio.pack(pady=5)
            
            video_radio = ctk.CTkRadioButton(parent, text="影片檔案", 
                                           variable=self.source_type, 
                                           value="video",
                                           command=self.on_source_change)
            video_radio.pack(pady=5)
            
            # 攝影機設定
            self.camera_frame = ctk.CTkFrame(parent)
            self.camera_frame.pack(fill=tk.X, pady=10, padx=20)
            
            camera_idx_label = ctk.CTkLabel(self.camera_frame, text="攝影機編號：")
            camera_idx_label.grid(row=0, column=0, sticky=tk.W, pady=5)
            
            camera_idx_spin = ctk.CTkEntry(self.camera_frame, textvariable=self.camera_index, width=100)
            camera_idx_spin.grid(row=0, column=1, pady=5)
            
            # 影片檔案選擇
            self.video_frame = ctk.CTkFrame(parent)
            
            video_path_label = ctk.CTkLabel(self.video_frame, text="影片路徑：")
            video_path_label.grid(row=0, column=0, sticky=tk.W, pady=5)
            
            video_path_entry = ctk.CTkEntry(self.video_frame, textvariable=self.video_path, width=200)
            video_path_entry.grid(row=1, column=0, pady=5, padx=(0, 5))
            
            browse_btn = ctk.CTkButton(self.video_frame, text="瀏覽", 
                                      command=self.browse_video,
                                      width=60)
            browse_btn.grid(row=1, column=1, pady=5)
            
        else:
            source_label = ttk.Label(parent, text="選擇輸入源：")
            source_label.pack(pady=(10, 5))
            
            camera_radio = ttk.Radiobutton(parent, text="即時攝影機", 
                                         variable=self.source_type, 
                                         value="camera",
                                         command=self.on_source_change)
            camera_radio.pack(pady=5)
            
            video_radio = ttk.Radiobutton(parent, text="影片檔案", 
                                        variable=self.source_type, 
                                        value="video",
                                        command=self.on_source_change)
            video_radio.pack(pady=5)
            
            # 攝影機設定
            self.camera_frame = ttk.LabelFrame(parent, text="攝影機設定", padding=10)
            self.camera_frame.pack(fill=tk.X, pady=10, padx=20)
            
            ttk.Label(self.camera_frame, text="攝影機編號：").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Spinbox(self.camera_frame, from_=0, to=5, textvariable=self.camera_index, 
                       width=10).grid(row=0, column=1, pady=5)
            
            # 影片檔案選擇
            self.video_frame = ttk.LabelFrame(parent, text="影片檔案", padding=10)
            
            ttk.Label(self.video_frame, text="影片路徑：").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Entry(self.video_frame, textvariable=self.video_path, 
                     width=25).grid(row=1, column=0, pady=5, padx=(0, 5))
            ttk.Button(self.video_frame, text="瀏覽", 
                      command=self.browse_video).grid(row=1, column=1, pady=5)
        
        # 初始顯示
        self.on_source_change()
        
    def create_params_tab(self, parent):
        """建立參數設定分頁"""
        # 使用 ScrolledFrame 以容納所有參數
        if USE_CTK:
            # CustomTkinter 沒有內建的 ScrolledFrame，使用 Frame + Scrollbar
            canvas = tk.Canvas(parent, highlightthickness=0)
            scrollbar = ctk.CTkScrollbar(parent, command=canvas.yview)
            scrollable_frame = ctk.CTkFrame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            params_frame = scrollable_frame
        else:
            # 標準 tkinter 使用 Canvas + Scrollbar
            canvas = tk.Canvas(parent)
            scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            params_frame = scrollable_frame
        
        # 參數群組
        self.create_basic_params(params_frame)
        self.create_advanced_params(params_frame)
        
    def create_basic_params(self, parent):
        """建立基本參數設定"""
        if USE_CTK:
            basic_frame = ctk.CTkFrame(parent)
            basic_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ctk.CTkLabel(basic_frame, text="基本參數", 
                        font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
            
            # FPS
            fps_frame = ctk.CTkFrame(basic_frame)
            fps_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(fps_frame, text="目標 FPS：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(fps_frame, textvariable=self.target_fps, width=80).pack(side=tk.LEFT)
            
            # 解析度
            res_frame = ctk.CTkFrame(basic_frame)
            res_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(res_frame, text="解析度：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(res_frame, textvariable=self.frame_width, width=60).pack(side=tk.LEFT, padx=2)
            ctk.CTkLabel(res_frame, text="x").pack(side=tk.LEFT)
            ctk.CTkEntry(res_frame, textvariable=self.frame_height, width=60).pack(side=tk.LEFT, padx=2)
            
            # 球桌長度
            table_frame = ctk.CTkFrame(basic_frame)
            table_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(table_frame, text="球桌長度 (cm)：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(table_frame, textvariable=self.table_length, width=80).pack(side=tk.LEFT)
            
            # 檢測超時
            timeout_frame = ctk.CTkFrame(basic_frame)
            timeout_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(timeout_frame, text="檢測超時 (秒)：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(timeout_frame, textvariable=self.detection_timeout, width=80).pack(side=tk.LEFT)
            
        else:
            basic_frame = ttk.LabelFrame(parent, text="基本參數", padding=10)
            basic_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # FPS
            ttk.Label(basic_frame, text="目標 FPS：").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Spinbox(basic_frame, from_=30, to=120, textvariable=self.target_fps, 
                       width=10).grid(row=0, column=1, pady=5)
            
            # 解析度
            ttk.Label(basic_frame, text="解析度：").grid(row=1, column=0, sticky=tk.W, pady=5)
            res_frame = ttk.Frame(basic_frame)
            res_frame.grid(row=1, column=1, pady=5)
            ttk.Entry(res_frame, textvariable=self.frame_width, width=6).pack(side=tk.LEFT)
            ttk.Label(res_frame, text=" x ").pack(side=tk.LEFT)
            ttk.Entry(res_frame, textvariable=self.frame_height, width=6).pack(side=tk.LEFT)
            
            # 球桌長度
            ttk.Label(basic_frame, text="球桌長度 (cm)：").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Entry(basic_frame, textvariable=self.table_length, width=10).grid(row=2, column=1, pady=5)
            
            # 檢測超時
            ttk.Label(basic_frame, text="檢測超時 (秒)：").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Entry(basic_frame, textvariable=self.detection_timeout, width=10).grid(row=3, column=1, pady=5)
    
    def create_advanced_params(self, parent):
        """建立進階參數設定"""
        if USE_CTK:
            adv_frame = ctk.CTkFrame(parent)
            adv_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ctk.CTkLabel(adv_frame, text="進階參數", 
                        font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
            
            # 過網方向
            dir_frame = ctk.CTkFrame(adv_frame)
            dir_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(dir_frame, text="過網方向：").pack(side=tk.LEFT, padx=5)
            ctk.CTkOptionMenu(dir_frame, variable=self.net_direction,
                             values=["right_to_left", "left_to_right", "both"]).pack(side=tk.LEFT)
            
            # 最大記錄數
            max_frame = ctk.CTkFrame(adv_frame)
            max_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(max_frame, text="最大記錄數：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(max_frame, textvariable=self.max_speeds, width=80).pack(side=tk.LEFT)
            
            # 透視校正
            persp_frame = ctk.CTkFrame(adv_frame)
            persp_frame.pack(fill=tk.X, pady=5)
            ctk.CTkLabel(persp_frame, text="近端寬度 (cm)：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(persp_frame, textvariable=self.near_width, width=60).pack(side=tk.LEFT, padx=5)
            ctk.CTkLabel(persp_frame, text="遠端寬度 (cm)：").pack(side=tk.LEFT, padx=5)
            ctk.CTkEntry(persp_frame, textvariable=self.far_width, width=60).pack(side=tk.LEFT)
            
            # 除錯模式
            ctk.CTkCheckBox(adv_frame, text="除錯模式", 
                           variable=self.debug_mode).pack(pady=10)
            
        else:
            adv_frame = ttk.LabelFrame(parent, text="進階參數", padding=10)
            adv_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # 過網方向
            ttk.Label(adv_frame, text="過網方向：").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Combobox(adv_frame, textvariable=self.net_direction,
                        values=["right_to_left", "left_to_right", "both"],
                        width=15).grid(row=0, column=1, pady=5)
            
            # 最大記錄數
            ttk.Label(adv_frame, text="最大記錄數：").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Entry(adv_frame, textvariable=self.max_speeds, width=10).grid(row=1, column=1, pady=5)
            
            # 透視校正
            ttk.Label(adv_frame, text="近端寬度 (cm)：").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Entry(adv_frame, textvariable=self.near_width, width=10).grid(row=2, column=1, pady=5)
            
            ttk.Label(adv_frame, text="遠端寬度 (cm)：").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Entry(adv_frame, textvariable=self.far_width, width=10).grid(row=3, column=1, pady=5)
            
            # 除錯模式
            ttk.Checkbutton(adv_frame, text="除錯模式", 
                           variable=self.debug_mode).grid(row=4, column=0, columnspan=2, pady=10)
    
    def create_stats_tab(self, parent):
        """建立統計數據分頁"""
        if USE_CTK:
            stats_frame = ctk.CTkFrame(parent)
            stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(stats_frame, text="即時統計", 
                        font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
            
            # 統計文字顯示
            self.stats_display = ctk.CTkTextbox(stats_frame, height=300)
            self.stats_display.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # 匯出按鈕
            export_btn = ctk.CTkButton(stats_frame, text="匯出統計數據", 
                                      command=self.export_stats)
            export_btn.pack(pady=10)
            
        else:
            stats_frame = ttk.Frame(parent)
            stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(stats_frame, text="即時統計", 
                     font=("Arial", 14, "bold")).pack(pady=10)
            
            # 統計文字顯示
            stats_text_frame = ttk.Frame(stats_frame)
            stats_text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            self.stats_display = tk.Text(stats_text_frame, height=15, width=40)
            self.stats_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(stats_text_frame, command=self.stats_display.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.stats_display.config(yscrollcommand=scrollbar.set)
            
            # 匯出按鈕
            ttk.Button(stats_frame, text="匯出統計數據", 
                      command=self.export_stats).pack(pady=10)
    
    def create_control_buttons(self, parent):
        """建立控制按鈕"""
        if USE_CTK:
            btn_frame = ctk.CTkFrame(parent)
            btn_frame.pack(fill=tk.X, padx=10, pady=20)
            
            # 開始/停止追蹤
            self.track_btn = ctk.CTkButton(btn_frame, text="開始追蹤", 
                                          command=self.toggle_tracking,
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          height=40)
            self.track_btn.pack(fill=tk.X, pady=5)
            
            # 開始/停止計數
            self.count_btn = ctk.CTkButton(btn_frame, text="開始計數", 
                                          command=self.toggle_counting,
                                          state="disabled",
                                          height=35)
            self.count_btn.pack(fill=tk.X, pady=5)
            
            # 重置按鈕
            reset_btn = ctk.CTkButton(btn_frame, text="重置", 
                                     command=self.reset_session,
                                     fg_color="orange",
                                     hover_color="darkorange",
                                     height=35)
            reset_btn.pack(fill=tk.X, pady=5)
            
        else:
            btn_frame = ttk.Frame(parent)
            btn_frame.pack(fill=tk.X, padx=10, pady=20)
            
            # 開始/停止追蹤
            self.track_btn = ttk.Button(btn_frame, text="開始追蹤", 
                                       command=self.toggle_tracking,
                                       style="Accent.TButton")
            self.track_btn.pack(fill=tk.X, pady=5)
            
            # 開始/停止計數
            self.count_btn = ttk.Button(btn_frame, text="開始計數", 
                                       command=self.toggle_counting,
                                       state="disabled")
            self.count_btn.pack(fill=tk.X, pady=5)
            
            # 重置按鈕
            ttk.Button(btn_frame, text="重置", 
                      command=self.reset_session).pack(fill=tk.X, pady=5)
    
    def create_video_panel(self, parent):
        """建立視訊顯示面板"""
        if USE_CTK:
            video_frame = ctk.CTkFrame(parent)
            video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # 視訊標題
            video_title = ctk.CTkLabel(video_frame, text="即時追蹤畫面", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
            video_title.pack(pady=10)
            
            # 視訊顯示區域
            self.video_label = tk.Label(video_frame, bg="black")
            self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 即時數據顯示
            data_frame = ctk.CTkFrame(video_frame)
            data_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            # 使用網格佈局顯示數據
            ctk.CTkLabel(data_frame, text="目前速度：").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, textvariable=self.current_speed).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, text="km/h").grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
            
            ctk.CTkLabel(data_frame, text="FPS：").grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, textvariable=self.current_fps).grid(row=0, column=4, padx=10, pady=5, sticky=tk.W)
            
            ctk.CTkLabel(data_frame, text="最後過網速度：").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, textvariable=self.last_net_speed).grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, text="km/h").grid(row=1, column=2, padx=10, pady=5, sticky=tk.W)
            
            ctk.CTkLabel(data_frame, text="已記錄：").grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)
            ctk.CTkLabel(data_frame, textvariable=self.speeds_collected).grid(row=1, column=4, padx=10, pady=5, sticky=tk.W)
            
        else:
            video_frame = ttk.LabelFrame(parent, text="即時追蹤畫面", padding=10)
            video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # 視訊顯示區域
            self.video_label = tk.Label(video_frame, bg="black")
            self.video_label.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # 即時數據顯示
            data_frame = ttk.Frame(video_frame)
            data_frame.pack(fill=tk.X, pady=10)
            
            # 使用網格佈局顯示數據
            ttk.Label(data_frame, text="目前速度：").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, textvariable=self.current_speed).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, text="km/h").grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(data_frame, text="FPS：").grid(row=0, column=3, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, textvariable=self.current_fps).grid(row=0, column=4, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(data_frame, text="最後過網速度：").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, textvariable=self.last_net_speed).grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, text="km/h").grid(row=1, column=2, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(data_frame, text="已記錄：").grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)
            ttk.Label(data_frame, textvariable=self.speeds_collected).grid(row=1, column=4, padx=10, pady=5, sticky=tk.W)
    
    def create_status_bar(self):
        """建立狀態列"""
        if USE_CTK:
            status_frame = ctk.CTkFrame(self.root)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            self.status_label = ctk.CTkLabel(status_frame, text="準備就緒")
            self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
            
            # 時間顯示
            self.time_label = ctk.CTkLabel(status_frame, text="")
            self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
            
        else:
            status_frame = ttk.Frame(self.root)
            status_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            self.status_label = ttk.Label(status_frame, text="準備就緒", relief=tk.SUNKEN)
            self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
            
            # 時間顯示
            self.time_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
            self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.update_time()
    
    def on_source_change(self):
        """處理輸入源改變"""
        if self.source_type.get() == "camera":
            self.camera_frame.pack(fill=tk.X, pady=10, padx=20)
            self.video_frame.pack_forget()
        else:
            self.video_frame.pack(fill=tk.X, pady=10, padx=20)
            self.camera_frame.pack_forget()
    
    def browse_video(self):
        """瀏覽選擇影片檔案"""
        filename = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"), ("所有檔案", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def toggle_tracking(self):
        """開始/停止追蹤"""
        if not self.is_tracking:
            self.start_tracking()
        else:
            self.stop_tracking()
    
    def start_tracking(self):
        """開始追蹤"""
        try:
            # 準備參數
            if self.source_type.get() == "camera":
                video_source = self.camera_index.get()
                use_video_file = False
            else:
                video_source = self.video_path.get()
                if not video_source or not os.path.exists(video_source):
                    messagebox.showerror("錯誤", "請選擇有效的影片檔案")
                    return
                use_video_file = True
            
            # 建立追蹤器
            self.tracker = PingPongSpeedTracker(
                video_source=video_source,
                table_length_cm=self.table_length.get(),
                detection_timeout_s=self.detection_timeout.get(),
                use_video_file=use_video_file,
                target_fps=self.target_fps.get(),
                frame_width=self.frame_width.get(),
                frame_height=self.frame_height.get(),
                debug_mode=self.debug_mode.get(),
                net_crossing_direction=self.net_direction.get(),
                max_net_speeds=self.max_speeds.get(),
                near_width_cm=self.near_width.get(),
                far_width_cm=self.far_width.get()
            )
            
            # 啟動追蹤執行緒
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
            self.tracking_thread.start()
            
            # 更新 UI
            if USE_CTK:
                self.track_btn.configure(text="停止追蹤", fg_color="red")
                self.count_btn.configure(state="normal")
            else:
                self.track_btn.config(text="停止追蹤")
                self.count_btn.config(state="normal")
            
            self.update_status("追蹤中...")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"無法啟動追蹤：{str(e)}")
            self.stop_tracking()
    
    def stop_tracking(self):
        """停止追蹤"""
        self.is_tracking = False
        
        if self.tracker:
            self.tracker.running = False
            self.tracker.reader.stop()
            
            # 等待執行緒結束
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=2.0)
            
            # 清理
            cv2.destroyAllWindows()
            self.tracker = None
        
        # 更新 UI
        if USE_CTK:
            self.track_btn.configure(text="開始追蹤", fg_color=["#3B8ED0", "#1F6AA5"])
            self.count_btn.configure(state="disabled", text="開始計數")
        else:
            self.track_btn.config(text="開始追蹤")
            self.count_btn.config(state="disabled", text="開始計數")
        
        self.is_counting.set(False)
        self.update_status("已停止")
        
        # 清空視訊顯示
        self.video_label.configure(image='')
    
    def tracking_loop(self):
        """追蹤執行緒主迴圈"""
        self.tracker.reader.start()
        
        while self.is_tracking and self.tracker.running:
            ret, frame = self.tracker.reader.read()
            if not ret or frame is None:
                if self.tracker.use_video_file:
                    self.update_status("影片播放結束")
                    if self.tracker.is_counting_active and self.tracker.collected_net_speeds:
                        self.tracker._generate_outputs_async()
                break
            
            # 處理框架
            frame_data = self.tracker.process_single_frame(frame)
            
            # 繪製視覺化
            display_frame = self.tracker._draw_visualizations(frame_data.frame, frame_data)
            
            # 將框架放入佇列
            if not self.frame_queue.full():
                self.frame_queue.put((display_frame, frame_data))
            
            # 檢查鍵盤輸入（在執行緒中處理）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                self.is_tracking = False
                break
    
    def toggle_counting(self):
        """開始/停止計數"""
        if self.tracker:
            self.tracker.toggle_counting()
            self.is_counting.set(self.tracker.is_counting_active)
            
            if self.is_counting.get():
                if USE_CTK:
                    self.count_btn.configure(text="停止計數", fg_color="orange")
                else:
                    self.count_btn.config(text="停止計數")
                self.update_status("計數中...")
            else:
                if USE_CTK:
                    self.count_btn.configure(text="開始計數", fg_color=["#3B8ED0", "#1F6AA5"])
                else:
                    self.count_btn.config(text="開始計數")
                self.update_status("計數已停止")
    
    def reset_session(self):
        """重置當前會話"""
        if self.tracker and self.is_counting.get():
            response = messagebox.askyesno("確認", "確定要重置當前會話嗎？未儲存的數據將會遺失。")
            if response:
                self.tracker.collected_net_speeds = []
                self.tracker.collected_relative_times = []
                self.tracker.event_buffer_center_cross.clear()
                self.tracker.output_generated_for_session = False
                self.update_stats_display()
                self.update_status("會話已重置")
    
    def export_stats(self):
        """匯出統計數據"""
        if not self.tracker or not self.tracker.collected_net_speeds:
            messagebox.showwarning("警告", "沒有可匯出的數據")
            return
        
        # 使用追蹤器的匯出功能
        self.tracker._generate_outputs_async()
        messagebox.showinfo("成功", "統計數據已匯出")
    
    def update_gui(self):
        """更新 GUI 顯示"""
        # 更新視訊顯示
        try:
            if not self.frame_queue.empty():
                display_frame, frame_data = self.frame_queue.get()
                
                # 轉換為 PIL 影像並調整大小
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # 調整影像大小以適應顯示區域
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                if label_width > 1 and label_height > 1:
                    image = self.resize_image(image, label_width, label_height)
                
                photo = ImageTk.PhotoImage(image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # 保持參考
                
                # 更新數據顯示
                self.current_speed.set(f"{frame_data.current_ball_speed_kmh:.1f}")
                self.current_fps.set(f"{frame_data.display_fps:.1f}")
                self.last_net_speed.set(f"{frame_data.last_recorded_net_speed_kmh:.1f}")
                self.speeds_collected.set(f"{len(frame_data.collected_net_speeds)}/{self.max_speeds.get()}")
                
                # 更新統計顯示
                self.update_stats_display()
                
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # 繼續更新
        self.root.after(30, self.update_gui)
    
    def resize_image(self, image, max_width, max_height):
        """調整影像大小以適應顯示區域"""
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        
        if img_width > max_width or img_height > max_height:
            if aspect_ratio > max_width / max_height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def update_stats_display(self):
        """更新統計顯示"""
        if self.tracker and self.tracker.collected_net_speeds:
            speeds = self.tracker.collected_net_speeds
            times = self.tracker.collected_relative_times
            
            stats_text = "即時統計數據\n" + "="*30 + "\n\n"
            stats_text += f"已記錄速度數: {len(speeds)}\n"
            stats_text += f"平均速度: {sum(speeds)/len(speeds):.1f} km/h\n"
            stats_text += f"最高速度: {max(speeds):.1f} km/h\n"
            stats_text += f"最低速度: {min(speeds):.1f} km/h\n\n"
            
            stats_text += "詳細記錄:\n" + "-"*20 + "\n"
            for i, (t, s) in enumerate(zip(times, speeds)):
                stats_text += f"{i+1:3d}. {t:6.2f}s: {s:5.1f} km/h\n"
            
            if USE_CTK:
                self.stats_display.delete("0.0", tk.END)
                self.stats_display.insert("0.0", stats_text)
            else:
                self.stats_display.delete(1.0, tk.END)
                self.stats_display.insert(1.0, stats_text)
        else:
            empty_text = "即時統計數據\n" + "="*30 + "\n\n尚無數據記錄"
            if USE_CTK:
                self.stats_display.delete("0.0", tk.END)
                self.stats_display.insert("0.0", empty_text)
            else:
                self.stats_display.delete(1.0, tk.END)
                self.stats_display.insert(1.0, empty_text)
    
    def update_status(self, message):
        """更新狀態列訊息"""
        self.status_label.configure(text=message)
    
    def update_time(self):
        """更新時間顯示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)
    
    def on_closing(self):
        """處理視窗關閉事件"""
        if self.is_tracking:
            response = messagebox.askyesno("確認", "追蹤仍在進行中，確定要退出嗎？")
            if not response:
                return
            self.stop_tracking()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """啟動 GUI"""
        self.root.mainloop()


def main():
    """主程式進入點"""
    # 建立並執行 GUI
    app = PingPongTrackerGUI()
    app.run()


if __name__ == "__main__":
    main()