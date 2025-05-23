#!/usr/bin/env python3
# test_fps_windows.py
#
# Windows 專用：以 DirectShow 後端量測攝影機真實 FPS
# ------------------------------------------------------

import cv2
import time
import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="測量 Windows 攝影機真實 FPS")
    ap.add_argument("-d", "--device", type=int, default=0,
                    help="DirectShow 裝置索引 (預設 0)")
    ap.add_argument("-t", "--seconds", type=int, default=5,
                    help="測量時間（秒）")
    return ap.parse_args()

def main():
    args = parse_args()

    # ➤ 1. 用 DirectShow 後端開啟裝置
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)

    # ➤ 2. 鎖定 MJPG（高幀率常見格式）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # ➤ 3. 設解析度與目標 FPS（可自行調整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,          120)

    print(f"[Driver] {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}×"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
          f"{cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    # ➤ 4. 連續抓取並計算平均 FPS
    start = time.time()
    frames = 0
    while time.time() - start < args.seconds:
        ret, _ = cap.read()
        if not ret:
            print("⚠️  讀流失敗，檢查裝置占用或訊號")
            break
        frames += 1

    elapsed = time.time() - start
    if frames:
        print(f"[Real]  {frames} frames / {elapsed:.2f} s → "
              f"{frames/elapsed:.2f} fps")

    cap.release()

if __name__ == "__main__":
    main()