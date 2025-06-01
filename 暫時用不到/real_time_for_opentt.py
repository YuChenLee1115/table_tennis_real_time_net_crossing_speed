import cv2
import numpy as np
import time
from collections import deque
import math
import argparse

class PingPongSpeedTracker:
    def __init__(self, video_source=0, table_length_cm=274, detection_timeout=0.3, use_video_file=False, target_fps=120):
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        self.use_video_file = use_video_file
        
        if not self.use_video_file:
            # Try to set webcam properties for better tracking
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Try to set FPS (may not work on all webcams)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # Verify what FPS was actually set
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Webcam FPS set to: {self.fps}")
            
            # If webcam doesn't report FPS or reports unreasonable value, use manual FPS calculation
            if self.fps <= 0 or self.fps > 1000:
                self.fps = 60  # Default assumption
                self.manual_fps_calc = True
                self.frame_times = deque(maxlen=20)  # Store recent frame timestamps
                print(f"Using manual FPS calculation (starting with estimate of {self.fps})")
            else:
                self.manual_fps_calc = False
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {self.fps}")
            self.manual_fps_calc = False
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Table dimensions
        self.table_length_cm = table_length_cm
        
        # Calculate the pixel to cm ratio based on frame width
        self.pixels_per_cm = self.frame_width / table_length_cm
        
        # Set the region of interest (middle 20% of the frame)
        self.roi_start_x = int(self.frame_width * 0.4)
        self.roi_end_x = int(self.frame_width * 0.6)
        
        # Tracking parameters
        self.detection_timeout = detection_timeout
        self.last_detection_time = time.time()
        self.trajectory = deque(maxlen=30)
        self.ball_speed = 0
        
        # For video files, we need to track frame numbers and times
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Store previous frames for FMO detection
        self.prev_frames = deque(maxlen=3)
        
        # Kernels for morphological operations
        self.opening_kernel = np.ones((6, 6), np.uint8)
        self.closing_kernel = np.ones((9, 9), np.uint8)
        
        print(f"Video resolution: {self.frame_width}x{self.frame_height}")
        print(f"ROI X range: {self.roi_start_x} to {self.roi_end_x}")
    
    def update_fps(self):
        """Update FPS calculation based on actual frame times"""
        if self.manual_fps_calc:
            current_time = time.time()
            self.frame_times.append(current_time)
            
            if len(self.frame_times) >= 10:
                # Calculate FPS based on recent frames
                time_diff = self.frame_times[-1] - self.frame_times[0]
                if time_diff > 0:
                    measured_fps = (len(self.frame_times) - 1) / time_diff
                    # Smooth FPS updates
                    self.fps = 0.7 * self.fps + 0.3 * measured_fps
            
            self.last_frame_time = current_time
    
    def preprocess_frame(self, frame):
        # Extract the ROI (middle 20% of the frame width)
        roi = frame[:, self.roi_start_x:self.roi_end_x].copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Store for FMO detection
        self.prev_frames.append(gray)
        
        return roi, gray
    
    def detect_fmo(self):
        if len(self.prev_frames) < 3:
            return None
        
        # Take three consecutive frames
        frame1 = self.prev_frames[-3]
        frame2 = self.prev_frames[-2]
        frame3 = self.prev_frames[-1]
        
        # Compute absolute differences between frames
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        
        # Bitwise AND to get only moving parts that are present in both differences
        fmo_mask = cv2.bitwise_and(diff1, diff2)
        
        # Threshold to get binary mask
        _, thresh = cv2.threshold(fmo_mask, 5, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise and fill gaps
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.opening_kernel)
        mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)
        
        return mask
    
    def detect_ball(self, roi, mask):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Filter contours by size and shape
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter out very small or very large contours
                if 20 < area < 2000:
                    # Find the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Convert coordinates to the original frame
                        cx_original = cx + self.roi_start_x
                        
                        # Update last detection time
                        self.last_detection_time = time.time()
                        
                        # For video files, use frame number for timing
                        if self.use_video_file:
                            timestamp = self.frame_count / self.fps
                        else:
                            timestamp = time.time()
                        
                        # Add to trajectory
                        self.trajectory.append((cx_original, cy, timestamp))
                        
                        return (cx, cy), contour
        
        return None, None
    
    def calculate_speed(self):
        if len(self.trajectory) >= 2:
            # Get the two most recent tracking points
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            
            # Calculate distance in pixels
            distance_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Convert distance to cm
            distance_cm = distance_pixels / self.pixels_per_cm
            
            # Calculate time difference
            time_diff = p2[2] - p1[2]
            
            if time_diff > 0:
                # Calculate speed in cm/s
                speed_cms = distance_cm / time_diff
                
                # Convert to km/h
                speed_kmh = speed_cms * 0.036
                
                # Update ball speed with smoothing
                self.ball_speed = 0.7 * self.ball_speed + 0.3 * speed_kmh if self.ball_speed > 0 else speed_kmh
    
    def draw_visualizations(self, frame, roi, ball_position=None, ball_contour=None):
        # Draw ROI boundaries
        cv2.line(frame, (self.roi_start_x, 0), (self.roi_start_x, self.frame_height), (0, 255, 0), 2)
        cv2.line(frame, (self.roi_end_x, 0), (self.roi_end_x, self.frame_height), (0, 255, 0), 2)
        
        # Draw trajectory
        for i in range(1, len(self.trajectory)):
            # Draw a line between consecutive points
            cv2.line(frame, 
                     (self.trajectory[i-1][0], self.trajectory[i-1][1]), 
                     (self.trajectory[i][0], self.trajectory[i][1]), 
                     (0, 0, 255), 2)
        
        # Draw current ball position
        if ball_position:
            cv2.circle(roi, ball_position, 5, (0, 255, 255), -1)
            
            # Draw the ball contour
            if ball_contour is not None:
                cv2.drawContours(roi, [ball_contour], 0, (255, 0, 0), 2)
        
        # Display ball speed
        cv2.putText(frame, f"Ball Speed: {self.ball_speed:.1f} km/h", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # If using video file, display frame number
        if self.use_video_file:
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def check_timeout(self):
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_timeout:
            # Clear trajectory if no detection for more than timeout
            self.trajectory.clear()
            self.ball_speed = 0
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file reached or camera disconnected.")
                break
            
            self.frame_count += 1
            
            # Update FPS calculation for webcam mode
            if not self.use_video_file and self.manual_fps_calc:
                self.update_fps()
            
            # Preprocess the frame
            roi, gray = self.preprocess_frame(frame)
            
            # Detect FMO
            fmo_mask = self.detect_fmo()
            
            if fmo_mask is not None:
                # Detect the ball
                ball_position, ball_contour = self.detect_ball(roi, fmo_mask)
                
                # Calculate speed
                self.calculate_speed()
                
                # Draw visualizations
                self.draw_visualizations(frame, roi, ball_position, ball_contour)
            else:
                # Still draw visualizations without ball position
                self.draw_visualizations(frame, roi)
            
            # Check for detection timeout (both in webcam and video mode)
            self.check_timeout()
            
            # Display the frame
            cv2.imshow("Ping Pong Speed Tracker", frame)
            
            # Display the ROI and FMO mask if available
            if roi is not None:
                cv2.imshow("ROI", roi)
            
            if fmo_mask is not None:
                cv2.imshow("FMO Mask", fmo_mask)
            
            # Control playback speed for video files (adjust the wait key value to speed up/slow down)
            if self.use_video_file:
                key = cv2.waitKey(10)  # Lower value = faster playback
            else:
                key = cv2.waitKey(1)
            
            # Exit on 'q' key press
            if key & 0xFF == ord('q'):
                break
            
            # Pause/resume on 'space' key press
            if key & 0xFF == ord(' '):
                cv2.waitKey(0)
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Ping Pong Speed Tracker')
    parser.add_argument('--video', type=str, default='', help='Path to video file (leave empty for webcam)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (0=built-in, 1=external/iPhone)')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS for webcam (default: 60)')
    args = parser.parse_args()
    
    if args.video:
        # 使用影片文件
        tracker = PingPongSpeedTracker(args.video, use_video_file=True)
    else:
        # 使用網絡攝像頭，允許指定相機編號
        tracker = PingPongSpeedTracker(args.camera, use_video_file=False, target_fps=args.fps)
    
    tracker.run()

if __name__ == "__main__":
    main()