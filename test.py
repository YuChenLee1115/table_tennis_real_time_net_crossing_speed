import cv2

gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1920, height=1080, framerate=120/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 在此處處理每一幀影像
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()