import cv2
import numpy as np
import time
import math
import argparse
from collections import deque

# Vimba API
from vmbpy import VmbSystem, PixelFormat

class PingPongSpeedTracker:
    def __init__(self,
                 video_source=0,
                 table_length_cm=274,
                 detection_timeout=1,
                 use_video_file=False,
                 target_fps=60,
                 use_vimba=False):
        self.use_vimba = use_vimba
        self.use_video_file = use_video_file
        self.table_length_cm = table_length_cm
        self.detection_timeout = detection_timeout

        if self.use_vimba:
            # ---- Vimba 初始化 ----
            self.vmb = VmbSystem.get_instance()
            self.vmb.__enter__()  # 啟動 Vimba 系統
            cams = self.vmb.get_all_cameras()
            if not cams:
                raise RuntimeError("未偵測到任何 Allied Vision 相機")
            self.cam = cams[0]
            self.cam.__enter__()  # 開啟相機

            # 診斷輸出 - 相機資訊
            print("\n===== 相機資訊 =====")
            print(f"相機 ID: {self.cam.get_id()}")
            print(f"相機型號: {self.cam.get_name()}")
            print(f"相機序號: {self.cam.get_serial()}")
            
            # 檢查解析度範圍
            try:
                width_min = self.cam.Width.get_range()[0]
                width_max = self.cam.Width.get_range()[1]
                height_min = self.cam.Height.get_range()[0]
                height_max = self.cam.Height.get_range()[1]
                print(f"\n解析度範圍: 寬度 {width_min} - {width_max}, 高度 {height_min} - {height_max}")
                
                # 檢查目前解析度
                current_width = self.cam.Width.get()
                current_height = self.cam.Height.get()
                print(f"目前解析度: {current_width}x{current_height}")
                
                # 嘗試設定解析度
                target_width = min(1920, width_max)
                target_height = min(1080, height_max)
                
                # 設定寬度前關閉自動設定（如果有）
                try:
                    if hasattr(self.cam, 'WidthAuto'):
                        self.cam.WidthAuto.set('Off')
                    if hasattr(self.cam, 'HeightAuto'):
                        self.cam.HeightAuto.set('Off')
                except Exception as e:
                    print(f"無法關閉自動解析度: {e}")
                
                # 有些相機需要先設定 Offset 為 0
                try:
                    if hasattr(self.cam, 'OffsetX'):
                        self.cam.OffsetX.set(0)
                    if hasattr(self.cam, 'OffsetY'):
                        self.cam.OffsetY.set(0)
                except Exception as e:
                    print(f"無法設定偏移: {e}")
                
                # 設定解析度 - 嘗試不同方法
                try:
                    # 方法1: 直接設定
                    self.cam.Width.set(target_width)
                    self.cam.Height.set(target_height)
                except Exception as e:
                    print(f"直接設定解析度失敗: {e}")
                    try:
                        # 方法2: 設定偶數解析度（有些相機需要）
                        if target_width % 2 != 0:
                            target_width -= 1
                        if target_height % 2 != 0:
                            target_height -= 1
                        self.cam.Width.set(target_width)
                        self.cam.Height.set(target_height)
                    except Exception as e2:
                        print(f"設定偶數解析度失敗: {e2}")
                        
                # 檢查設定後的解析度
                actual_width = self.cam.Width.get()
                actual_height = self.cam.Height.get()
                print(f"設定後解析度: {actual_width}x{actual_height}")
                
                # 檢查幀率範圍和設定
                try:
                    if hasattr(self.cam, 'AcquisitionFrameRate'):
                        fps_min, fps_max = self.cam.AcquisitionFrameRate.get_range()
                        print(f"\n幀率範圍: {fps_min} - {fps_max} fps")
                        
                        # 嘗試設定幀率
                        if hasattr(self.cam, 'AcquisitionFrameRateEnable'):
                            self.cam.AcquisitionFrameRateEnable.set(True)
                            
                        if hasattr(self.cam, 'AcquisitionFrameRateMode'):
                            self.cam.AcquisitionFrameRateMode.set('Basic')
                            
                        target_fps = min(target_fps, fps_max)
                        self.cam.AcquisitionFrameRate.set(target_fps)
                        actual_fps = self.cam.AcquisitionFrameRate.get()
                        print(f"設定幀率: 目標 {target_fps}, 實際 {actual_fps} fps")
                except Exception as e:
                    print(f"無法取得或設定幀率範圍: {e}")
                    
                # 效能優化設定
                print("\n===== 效能優化設定 =====")
                
                # 1. 檢查並設定傳輸層控制
                try:
                    if hasattr(self.cam, 'DeviceLinkThroughputLimit'):
                        max_throughput = self.cam.DeviceLinkThroughputLimit.get_range()[1]
                        self.cam.DeviceLinkThroughputLimit.set(max_throughput)
                        print(f"設定最大傳輸速率: {max_throughput}")
                except Exception as e:
                    print(f"無法設定傳輸速率: {e}")
                
                # 2. 增加傳輸緩衝區大小
                try:
                    if hasattr(self.cam, 'StreamBufferCountMax'):
                        max_buffer = self.cam.StreamBufferCountMax.get()
                        print(f"最大緩衝區數量: {max_buffer}")
                except Exception as e:
                    print(f"無法取得緩衝區資訊: {e}")
                    
                # 3. 檢查和優化封包大小
                try:
                    streams = self.cam.get_streams()
                    if streams:
                        stream = streams[0]
                        if hasattr(stream, 'GVSPAdjustPacketSize'):
                            print("正在優化 GigE 封包大小...")
                            stream.GVSPAdjustPacketSize.run()
                            while not stream.GVSPAdjustPacketSize.is_done():
                                pass
                            print("GigE 封包大小優化完成")
                except Exception as e:
                    print(f"無法優化 GigE 封包大小: {e}")
                    
                # 4. 降低影像處理費用（如果需要提高速度）
                # 例如減少 ROI 大小或調整解析度
                print("\n若影像處理速度仍然過慢，建議：")
                print("1. 降低解析度")
                print("2. 減小 ROI 範圍")
                print("3. 降低幀率")
                print("4. 檢查 USB/網路連接")
                print("============================\n")
                
            except Exception as e:
                print(f"檢查和設定相機參數時出錯: {e}")

            # 列印相機支援的 PixelFormat
            supported_formats = self.cam.get_pixel_formats()
            print("Supported PixelFormats:", supported_formats)
            
            # 列印所有可用的顏色像素格式
            from vmbpy import COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
            color_formats = [fmt for fmt in supported_formats if fmt in COLOR_PIXEL_FORMATS]
            mono_formats = [fmt for fmt in supported_formats if fmt in MONO_PIXEL_FORMATS]
            print("支援的彩色格式:", color_formats)
            print("支援的單色格式:", mono_formats)

            # 設定目標格式為 BGR8（OpenCV 兼容格式）
            target_format = PixelFormat.Bgr8

            # 優先選擇彩色格式
            if color_formats:
                # 檢查哪些彩色格式可以轉換為 BGR8
                for fmt in color_formats:
                    try:
                        convertible_formats = fmt.get_convertible_formats()
                        if target_format in convertible_formats:
                            self.cam.set_pixel_format(fmt)
                            print(f"使用彩色格式 {fmt} (將轉換為 {target_format})")
                            break
                    except Exception as e:
                        print(f"檢查格式 {fmt} 時出錯: {e}")
                else:
                    # 如果沒有找到可轉換為BGR8的彩色格式，使用第一個彩色格式
                    self.cam.set_pixel_format(color_formats[0])
                    print(f"使用彩色格式 {color_formats[0]}")
            elif PixelFormat.Mono8 in supported_formats:
                # 如果沒有彩色格式，退回到 Mono8
                self.cam.set_pixel_format(PixelFormat.Mono8)
                print("只能使用 Mono8 格式（黑白圖像）")
            else:
                # 使用第一個可用格式
                self.cam.set_pixel_format(supported_formats[0])
                print(f"使用 {supported_formats[0]} 格式 (可能無法正確顯示)")

            # 啟用自動曝光（如果相機支持）
            try:
                self.cam.ExposureAuto.set('Continuous')
                print("已啟用自動曝光")
            except (AttributeError, VmbFeatureError):
                print("相機不支持自動曝光設置")

            # 啟用自動白平衡（如果相機支持）- 對彩色圖像很重要
            try:
                self.cam.BalanceWhiteAuto.set('Continuous')
                print("已啟用自動白平衡")
            except (AttributeError, VmbFeatureError):
                print("相機不支持自動白平衡")

            # 不使用 cv2.VideoCapture
            self.cap = None
            self.fps = target_fps
            self.manual_fps_calc = False
            print(f"使用 Vimba API 串流，相機目標 FPS：{self.fps}")

            # 抓一張 Frame 取得解析度
            frame = self.cam.get_frame()
            print(f"Frame pixel format: {frame.get_pixel_format()}")
            
            # 嘗試轉換為 OpenCV 兼容格式
            try:
                # 如果相機使用的是彩色格式
                if frame.get_pixel_format() in COLOR_PIXEL_FORMATS:
                    # 嘗試轉換為 BGR8
                    if frame.get_pixel_format() == target_format:
                        img = frame.as_opencv_image()
                    else:
                        # 轉換為目標格式
                        converted_frame = frame.convert_pixel_format(target_format)
                        img = converted_frame.as_opencv_image()
                # 如果相機使用的是單色格式
                elif frame.get_pixel_format() in MONO_PIXEL_FORMATS:
                    if frame.get_pixel_format() == PixelFormat.Mono8:
                        # 單色圖像轉為三通道顯示
                        mono_img = frame.as_opencv_image()
                        img = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
                    else:
                        # 先轉換為 Mono8，再轉為 BGR
                        mono_frame = frame.convert_pixel_format(PixelFormat.Mono8)
                        mono_img = mono_frame.as_opencv_image()
                        img = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
                else:
                    # 嘗試直接轉換
                    try:
                        converted_frame = frame.convert_pixel_format(target_format)
                        img = converted_frame.as_opencv_image()
                    except Exception as e:
                        print(f"無法轉換為 BGR8: {e}")
                        raw_img = frame.as_numpy_array()
                        if len(raw_img.shape) == 2:
                            img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                        else:
                            img = raw_img
                
                self.frame_height, self.frame_width = img.shape[:2]
                print(f"成功獲取圖像，尺寸: {self.frame_width}x{self.frame_height}")
            except Exception as e:
                print(f"圖像轉換錯誤: {e}")
                # 使用相機報告的尺寸
                self.frame_width = frame.get_width()
                self.frame_height = frame.get_height()
                print(f"使用相機報告的尺寸: {self.frame_width}x{self.frame_height}")
            self.frame_height, self.frame_width = img.shape[:2]
        else:
            # ---- OpenCV VideoCapture 初始化 ----
            self.cap = cv2.VideoCapture(video_source)
            if not self.use_video_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 1000:
                    self.fps = target_fps
                    self.manual_fps_calc = True
                    self.frame_times = deque(maxlen=20)
                    print(f"使用手動 FPS 計算，初始假設 FPS：{self.fps}")
                else:
                    self.manual_fps_calc = False
                print(f"Webcam FPS set to: {self.fps}")
            else:
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.manual_fps_calc = False
                print(f"Video FPS: {self.fps}")

            # 取得解析度
            self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 公用屬性
        self.pixels_per_cm    = self.frame_width / table_length_cm
        self.roi_start_x      = int(self.frame_width * 0.4)
        self.roi_end_x        = int(self.frame_width * 0.6)
        self.last_detection_time = time.time()
        self.trajectory       = deque(maxlen=30)
        self.ball_speed       = 0.0
        self.frame_count      = 0
        self.prev_frames      = deque(maxlen=3)
        self.opening_kernel   = np.ones((3, 3), np.uint8)
        self.closing_kernel   = np.ones((5, 5), np.uint8)

        print(f"解析度：{self.frame_width}×{self.frame_height}，ROI X 範圍：[{self.roi_start_x}, {self.roi_end_x}]")

    def update_fps(self):
        if self.manual_fps_calc:
            now = time.time()
            self.frame_times.append(now)
            if len(self.frame_times) >= 10:
                dt = self.frame_times[-1] - self.frame_times[0]
                if dt > 0:
                    measured = (len(self.frame_times)-1) / dt
                    self.fps = 0.7*self.fps + 0.3*measured

    def preprocess_frame(self, frame):
        roi = frame[:, self.roi_start_x:self.roi_end_x].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.prev_frames.append(gray)
        return roi, gray

    def detect_fmo(self):
        if len(self.prev_frames) < 3:
            return None
        
        # 修正: 使用列表中最後三個元素，但不使用切片
        if len(self.prev_frames) >= 3:
            f1 = self.prev_frames[len(self.prev_frames)-3]
            f2 = self.prev_frames[len(self.prev_frames)-2]
            f3 = self.prev_frames[len(self.prev_frames)-1]
            
            d1 = cv2.absdiff(f1, f2)
            d2 = cv2.absdiff(f2, f3)
            mask = cv2.bitwise_and(d1, d2)
            _, thr = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
            open_ = cv2.morphologyEx(thr, cv2.MORPH_OPEN, self.opening_kernel)
            return cv2.morphologyEx(open_, cv2.MORPH_CLOSE, self.closing_kernel)
        return None

    def detect_ball(self, roi, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            if 20 < area < 500:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cx0 = cx + self.roi_start_x
                self.last_detection_time = time.time()
                ts = (self.frame_count / self.fps) if self.use_video_file else time.time()
                self.trajectory.append((cx0, cy, ts))
                return (cx, cy), c
        return None, None

    def calculate_speed(self):
        if len(self.trajectory) >= 2:
            x1, y1, t1 = self.trajectory[-2]
            x2, y2, t2 = self.trajectory[-1]
            dp = math.hypot(x2-x1, y2-y1)
            dc = dp / self.pixels_per_cm
            dt = t2 - t1
            if dt > 0:
                kmh = (dc / dt) * 0.036
                self.ball_speed = 0.7*self.ball_speed + 0.3*kmh if self.ball_speed > 0 else kmh

    def draw_visualizations(self, frame, roi, pos=None, cnt=None):
        # ROI 邊界
        cv2.line(frame, (self.roi_start_x,0),(self.roi_start_x,self.frame_height),(0,255,0),2)
        cv2.line(frame, (self.roi_end_x,0),(self.roi_end_x,self.frame_height),(0,255,0),2)
        # 軌跡
        for i in range(1, len(self.trajectory)):
            x1,y1,_ = self.trajectory[i-1]
            x2,y2,_ = self.trajectory[i]
            cv2.line(frame, (x1,y1),(x2,y2),(0,0,255),2)
        # 球體位置
        if pos:
            cv2.circle(roi, pos, 5, (0,255,255), -1)
            if cnt is not None:
                cv2.drawContours(roi, [cnt], 0, (255,0,0), 2)
        # 速度與 FPS
        cv2.putText(frame, f"Speed: {self.ball_speed:.1f} km/h", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, f"FPS:   {self.fps:.1f}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        if self.use_video_file:
            cv2.putText(frame, f"Frame: {self.frame_count}", (10,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    def check_timeout(self):
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.trajectory.clear()
            self.ball_speed = 0.0

    def run(self, frame_limit=None):
        """
        執行程式主迴圈，擷取和處理影像
        
        Args:
            frame_limit: 處理的最大幀數，如果不指定則無限循環
        """
        # 用於計算處理效能的變數
        processing_times = []
        frame_times = []
        last_frame_time = time.time()
        processed_frames = 0
        
        # 讀取緩衝區數量和設定
        buffer_size = 10  # 預設值
        
        if self.use_vimba:
            # ---- Vimba 同步擷取迴圈 ----
            target_format = PixelFormat.Bgr8
            from vmbpy import COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
            
            while frame_limit is None or processed_frames < frame_limit:
                try:
                    # 記錄開始處理的時間
                    start_time = time.time()
                    
                    frame = self.cam.get_frame()
                    
                    # 嘗試轉換為 OpenCV 兼容格式
                    try:
                        # 如果相機使用的是彩色格式
                        if frame.get_pixel_format() in COLOR_PIXEL_FORMATS:
                            # 嘗試轉換為 BGR8
                            if frame.get_pixel_format() == target_format:
                                img = frame.as_opencv_image()
                            else:
                                # 轉換為目標格式
                                converted_frame = frame.convert_pixel_format(target_format)
                                img = converted_frame.as_opencv_image()
                        # 如果相機使用的是單色格式
                        elif frame.get_pixel_format() in MONO_PIXEL_FORMATS:
                            if frame.get_pixel_format() == PixelFormat.Mono8:
                                # 單色圖像轉為三通道顯示
                                mono_img = frame.as_opencv_image()
                                img = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
                            else:
                                # 先轉換為 Mono8，再轉為 BGR
                                mono_frame = frame.convert_pixel_format(PixelFormat.Mono8)
                                mono_img = mono_frame.as_opencv_image()
                                img = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
                        else:
                            # 嘗試直接轉換
                            try:
                                converted_frame = frame.convert_pixel_format(target_format)
                                img = converted_frame.as_opencv_image()
                            except Exception as e:
                                print(f"無法轉換為 BGR8: {e}")
                                raw_img = frame.as_numpy_array()
                                if len(raw_img.shape) == 2:
                                    img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                                else:
                                    img = raw_img
                    except Exception as e:
                        print(f"圖像轉換錯誤: {e}")
                        # 如果轉換失敗，嘗試作為灰階圖像處理
                        try:
                            raw_img = frame.as_numpy_array()
                            if len(raw_img.shape) == 2:
                                img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                            else:
                                img = raw_img
                        except:
                            # 建立空白圖像
                            img = np.zeros((frame.get_height(), frame.get_width(), 3), dtype=np.uint8)
                            
                    self.frame_count += 1
                    processed_frames += 1

                    # 處理影像並計算速度
                    roi, _ = self.preprocess_frame(img)
                    mask = self.detect_fmo()
                    if mask is not None:
                        pos, cnt = self.detect_ball(roi, mask)
                        self.calculate_speed()
                        self.draw_visualizations(img, roi, pos, cnt)
                    else:
                        self.draw_visualizations(img, roi)
                    self.check_timeout()
                    
                    # 計算和顯示性能指標
                    now = time.time()
                    process_time = now - start_time
                    frame_time = now - last_frame_time
                    last_frame_time = now
                    
                    processing_times.append(process_time)
                    frame_times.append(frame_time)
                    
                    # 每 30 幀計算一次平均值
                    if len(processing_times) > 30:
                        processing_times.pop(0)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    
                    avg_process_time = sum(processing_times) / len(processing_times)
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    
                    # 在影像上顯示更多性能信息
                    cv2.putText(img, f"Process: {avg_process_time*1000:.1f} ms", (10,110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    cv2.putText(img, f"Actual FPS: {actual_fps:.1f}", (10,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    cv2.putText(img, f"Frame: {self.frame_count}", (10,190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                    # 顯示影像
                    cv2.imshow("PingPong (Vimba)", img)
                    
                    # 如果處理時間小於影格時間，可以等待以減少 CPU 使用率
                    if process_time < 1.0/self.fps:
                        wait_time = max(1, int((1.0/self.fps - process_time)*1000))
                    else:
                        wait_time = 1
                        
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                        
                except VmbFrameError as e:
                    print(f"擷取幀錯誤: {e}")
                    continue

            # 清理 Vimba
            self.cam.__exit__(None, None, None)
            self.vmb.__exit__(None, None, None)

        else:
            # ---- OpenCV VideoCapture 迴圈 ----
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("影片結束或相機未連線")
                    break
                self.frame_count += 1

                if not self.use_video_file and self.manual_fps_calc:
                    self.update_fps()

                roi, _ = self.preprocess_frame(frame)
                mask = self.detect_fmo()
                if mask is not None:
                    pos, cnt = self.detect_ball(roi, mask)
                    self.calculate_speed()
                    self.draw_visualizations(frame, roi, pos, cnt)
                else:
                    self.draw_visualizations(frame, roi)
                self.check_timeout()

                cv2.imshow("PingPong Speed Tracker", frame)
                cv2.imshow("ROI", roi)
                if mask is not None:
                    cv2.imshow("FMO Mask", mask)

                key = cv2.waitKey(1 if not self.use_video_file else 30)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord(' '):
                    cv2.waitKey(0)

            if self.cap is not None:
                self.cap.release()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker')
    parser.add_argument('--video', action='store_true',
                        help='Use video file instead of camera')
    parser.add_argument('--use-vimba', action='store_true',
                        help='Use Allied Vision Vimba API instead of OpenCV VideoCapture')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS for webcam / Vimba stream')
    parser.add_argument('--buffer-count', type=int, default=10,
                        help='Number of frame buffers to use (higher values can improve performance)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Target camera width resolution')
    parser.add_argument('--height', type=int, default=1080,
                        help='Target camera height resolution')
    parser.add_argument('--table-length', type=int, default=274,
                        help='Length of ping pong table in cm')
    parser.add_argument('path', nargs='?', default='',
                        help='Path to video file (if --video)')
    args = parser.parse_args()

    if args.video:
        tracker = PingPongSpeedTracker(
            video_source=args.path,
            use_video_file=True,
            target_fps=args.fps,
            use_vimba=False,
            table_length_cm=args.table_length
        )
    else:
        tracker = PingPongSpeedTracker(
            video_source=0,
            use_video_file=False,
            target_fps=args.fps,
            use_vimba=args.use_vimba,
            table_length_cm=args.table_length
        )
    tracker.run()

if __name__ == "__main__":
    main()