import cv2
import numpy as np
import time
from collections import deque
import math

class PingPongSpeedTracker:
    def __init__(self, video_source=0, table_length_cm=274, detection_timeout=1):
        # Initialize webcam
        self.cap = cv2.VideoCapture(video_source)
        
        # Try to set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get actual webcam resolution
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
        
        # Store previous frames for FMO detection
        self.prev_frames = deque(maxlen=3)
        
        # Kernels for morphological operations
        self.opening_kernel = np.ones((2, 2), np.uint8)
        self.closing_kernel = np.ones((5, 5), np.uint8)
        
        print(f"Webcam resolution: {self.frame_width}x{self.frame_height}")
        print(f"ROI X range: {self.roi_start_x} to {self.roi_end_x}")
    
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
        _, thresh = cv2.threshold(fmo_mask, 10, 255, cv2.THRESH_BINARY)
        
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
                if 10 < area < 500:
                    # Find the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Convert coordinates to the original frame
                        cx_original = cx + self.roi_start_x
                        
                        # Update last detection time
                        self.last_detection_time = time.time()
                        
                        # Add to trajectory
                        self.trajectory.append((cx_original, cy, time.time()))
                        
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
                break
                
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
            
            # Check for detection timeout
            self.check_timeout()
            
            # Display the frame
            cv2.imshow("Ping Pong Speed Tracker", frame)
            
            # Display the ROI and FMO mask if available
            if roi is not None:
                cv2.imshow("ROI", roi)
            
            if fmo_mask is not None:
                cv2.imshow("FMO Mask", fmo_mask)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    tracker = PingPongSpeedTracker()
    tracker.run()

if __name__ == "__main__":
    main()