import cv2
import mediapipe as mp
import os

# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "video_crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================

VIDEO_PATH = os.path.join(VIDEO_DIR, "input_video.mp4")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

# ===============================
# MEDIAPIPE FACE DETECTOR

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)

frame_count = 0
saved_count = 0

# ===============================
# PROCESS VIDEO FRAME BY FRAME

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Keep box inside image
            x, y = max(0, x), max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face_crop = frame[y:y + bh, x:x + bw]

            if face_crop.size == 0:
                continue

            save_name = f"frame_{frame_count}_face_{i}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, face_crop)
            saved_count += 1


            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Detection from Video File", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"[DONE] Total frames processed: {frame_count}")
print(f"[DONE] Total faces saved: {saved_count}")
print(f"[DONE] Cropped faces saved in: {OUTPUT_DIR}")
