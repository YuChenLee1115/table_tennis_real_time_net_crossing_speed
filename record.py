#!/usr/bin/env python3
# high_fps_recorder.py
#
# macOS 專用高幀率視頻錄製程式
# ------------------------------------------------------

import cv2
import time
import os
import numpy as np
from datetime import datetime

def record_video(name, device_index=0, resolution=(1920, 1080), target_fps=120, codec='mp4v'):
    """
    使用 AVFoundation 後端錄製高幀率視頻
    按空白鍵開始/停止錄製
    
    參數:
    - name: 使用者姓名，用於檔案和資料夾命名
    - device_index: 攝像頭/擷取卡索引
    - resolution: 影片解析度，默認1920x1080 (1080p)
    - target_fps: 目標每秒幀數，默認120fps
    - codec: 輸出視頻編碼格式，默認'mp4v'
    """
    # 建立時間戳記 (格式: YYYYMMDD_HHMMSS)
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建資料夾名稱和路徑
    folder_name = f"{name}_{base_timestamp}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    
    # 確保資料夾存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"已創建資料夾: {folder_path}")
    
    print("=== macOS 高幀率視頻錄製程式 ===")
    print(f"使用者: {name}")
    print(f"所有錄製的影片將保存在: {folder_path}")
    print("初始化 AVFoundation 後端...")
    
    # 使用 AVFoundation 後端初始化攝像頭/擷取卡
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    
    # 檢查是否成功打開
    if not cap.isOpened():
        print(f"錯誤：無法打開索引為 {device_index} 的影像裝置")
        return
    
    # 顯示後端資訊
    print(f"使用後端: AVFoundation")
    
    # 設定 MJPG 格式 (有助於提高幀率)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 設置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # 設置幀率
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # 檢查實際設置的參數
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"相機初始化完成")
    print(f"報告的分辨率: {actual_width}x{actual_height}")
    print(f"報告的幀率: {actual_fps}fps")
    
    # 為輸出視頻設定編解碼器
    output_fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # 變數初始化
    out = None
    recording = False
    start_time = None
    frames_recorded = 0
    frame_times = []
    show_fps = True
    recording_count = 1  # 記錄當前是第幾段錄製
    
    print("\n準備就緒!")
    print("按 '空白鍵' 開始/停止錄製")
    print("按 'f' 鍵顯示/隱藏FPS計數器")
    print("按 'q' 鍵退出程式")
    
    # FPS 測量變數
    fps_start = time.time()
    fps_count = 0
    current_fps = 0
    
    # 記錄每秒的 FPS
    fps_history = []
    
    while True:
        # 擷取影像
        ret, frame = cap.read()
        
        if not ret:
            print("無法讀取視頻幀")
            time.sleep(0.1)
            continue
        
        # 計算即時FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count
            fps_history.append(current_fps)
            if len(fps_history) > 5:  # 保持最近5秒的數據
                fps_history.pop(0)
            fps_count = 0
            fps_start = time.time()
        
        # 在畫面上顯示資訊 (使用英文)
        # 1. 錄製狀態
        status_text = "RECORDING..." if recording else "READY (Press SPACE to start)"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255) if recording else (0, 255, 0), 2)
        
        # 2. 顯示FPS
        if show_fps:
            cv2.putText(frame, f"Real-time FPS: {current_fps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 0), 2)
            
            if len(fps_history) > 0:
                avg_fps = sum(fps_history) / len(fps_history)
                cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (255, 255, 0), 2)
        
        # 3. 如果正在錄製，顯示錄製時間
        if recording and start_time:
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Recording Time: {elapsed:.1f}s", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 165, 255), 2)
            
            # 計算錄製中的平均FPS
            if len(frame_times) > 10:
                record_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                cv2.putText(frame, f"Recording FPS: {record_fps:.1f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 165, 255), 2)
        
        # 顯示使用者姓名
        cv2.putText(frame, f"User: {name}", (actual_width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        
        # 顯示預覽
        cv2.imshow('High FPS Camera', frame)
        
        # 如果正在錄製，寫入視頻
        if recording:
            out.write(frame)
            frame_times.append(time.time())
            frames_recorded += 1
        
        # 處理按鍵
        key = cv2.waitKey(1) & 0xFF
        
        # 按q鍵退出
        if key == ord('q'):
            print("程式結束")
            break
        
        # 按f鍵切換FPS顯示
        if key == ord('f'):
            show_fps = not show_fps
            print(f"FPS顯示: {'開啟' if show_fps else '關閉'}")
        
        # 按空白鍵開始/停止錄製
        if key == 32:  # 空白鍵的ASCII碼是32
            if not recording:
                # 開始錄製
                recording = True
                start_time = time.time()
                frames_recorded = 0
                frame_times = []
                
                # 創建新的視頻文件
                recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{name}_{recording_timestamp}.mp4"
                output_path = os.path.join(folder_path, output_filename)
                
                out = cv2.VideoWriter(
                    output_path, 
                    output_fourcc, 
                    target_fps, 
                    (actual_width, actual_height)
                )
                
                print(f"\n開始錄製 #{recording_count}...")
                print(f"輸出文件: {output_path}")
                print(f"目標幀率: {target_fps}fps")
                
            else:
                # 停止錄製
                recording = False
                elapsed_time = time.time() - start_time
                
                # 計算實際幀率
                if len(frame_times) > 1:
                    actual_recorded_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                else:
                    actual_recorded_fps = 0
                
                # 關閉視頻寫入器
                if out:
                    out.release()
                    out = None
                
                print(f"\n錄製 #{recording_count} 完成!")
                print(f"錄製時間: {elapsed_time:.2f}秒")
                print(f"錄製幀數: {frames_recorded}幀")
                print(f"實際平均幀率: {actual_recorded_fps:.2f}fps")
                print(f"視頻已保存為: {output_path}")
                print("\n按 '空白鍵' 開始新的錄製")
                
                # 增加錄製計數
                recording_count += 1
    
    # 釋放資源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print("\n程式已退出")
    print(f"所有錄製的影片已保存在: {folder_path}")

if __name__ == "__main__":
    # 提示使用者輸入姓名
    print("=== 高幀率視頻錄製程式 ===")
    user_name = input("請輸入您的姓名: ")
    
    # 確保姓名是有效的檔案名稱
    # 移除可能導致檔案命名問題的字符
    user_name = "".join(c for c in user_name if c.isalnum() or c in "_ -")
    
    if not user_name:
        user_name = "User"  # 提供默認名稱
        print("使用默認使用者名稱: User")
    
    # 啟動錄製
    record_video(
        name=user_name,            # 使用者姓名
        device_index=0,            # 預設相機/擷取卡索引
        resolution=(1920, 1080),   # 1080p解析度
        target_fps=120,            # 目標120fps
        codec='mp4v'               # 輸出編碼器
    )