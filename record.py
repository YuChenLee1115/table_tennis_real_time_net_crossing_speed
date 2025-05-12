import cv2
import time
import os

def record_video(resolution=(1920, 1080), fps=120, filename='recorded_video.mp4'):
    """
    使用網絡攝像頭錄製高清視頻
    按空白鍵開始/停止錄製
    
    參數:
    - resolution: 影片解析度，默認1920x1080 (1080p)
    - fps: 每秒幀數，默認120fps
    - filename: 輸出文件名
    """
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    
    # 檢查攝像頭是否成功打開
    if not cap.isOpened():
        print("無法打開攝像頭")
        return
    
    # 嘗試設置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # 嘗試設置幀率
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # 檢查實際設置的分辨率和幀率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"相機初始化完成")
    print(f"實際相機分辨率: {actual_width}x{actual_height}")
    print(f"實際相機幀率: {actual_fps}fps")
    
    # 視頻編解碼器設置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264編碼
    out = None  # 先不初始化VideoWriter
    
    recording = False
    start_time = None
    frames_recorded = 0
    
    print("\n準備就緒!")
    print("按 '空白鍵' 開始/停止錄製")
    print("按 'q' 鍵退出程式")
    
    # 主循環
    while True:
        # 捕獲視頻幀
        ret, frame = cap.read()
        
        if not ret:
            print("無法讀取視頻幀")
            break
        
        # 顯示當前狀態
        status_text = "Recording..." if recording else "等待開始 (按空白鍵開始錄製)"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
        
        # 如果正在錄製，顯示已錄製時間
        if recording and start_time:
            elapsed_time = time.time() - start_time
            time_text = f"Record Time: {elapsed_time:.1f}s"
            cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2)
        
        # 顯示預覽
        cv2.imshow('Camera', frame)
        
        # 如果正在錄製，寫入視頻
        if recording:
            out.write(frame)
            frames_recorded += 1
        
        # 鍵盤事件處理
        key = cv2.waitKey(1) & 0xFF
        
        # 按q鍵退出
        if key == ord('q'):
            print("用戶終止程式")
            break
        
        # 按空白鍵開始/停止錄製
        if key == 32:  # 空白鍵的ASCII碼是32
            if not recording:
                # 開始錄製
                recording = True
                start_time = time.time()
                frames_recorded = 0
                
                # 創建新的視頻文件
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'recorded_video_{timestamp}.mp4'
                out = cv2.VideoWriter(filename, fourcc, actual_fps, (actual_width, actual_height))
                
                print(f"\n開始錄製...")
                print(f"輸出文件: {filename}")
            else:
                # 停止錄製
                recording = False
                elapsed_time = time.time() - start_time
                actual_recorded_fps = frames_recorded / elapsed_time if elapsed_time > 0 else 0
                
                # 關閉視頻寫入器
                if out:
                    out.release()
                    out = None
                
                print(f"\n錄製完成!")
                print(f"錄製時間: {elapsed_time:.2f}秒")
                print(f"錄製幀數: {frames_recorded}幀")
                print(f"實際平均幀率: {actual_recorded_fps:.2f}fps")
                print(f"視頻已保存為: {os.path.abspath(filename)}")
                print("\n按 '空白鍵' 開始新的錄製")
                print("按 'q' 鍵退出程式")
    
    # 完成後釋放所有資源
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print("\n程式已退出")

if __name__ == "__main__":
    record_video(
        resolution=(1920, 1080),   # 1080p解析度
        fps=120                    # 120fps
    )