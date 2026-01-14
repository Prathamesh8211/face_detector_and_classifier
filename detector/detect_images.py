import cv2
import mediapipe as mp
import os

# Initialize MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.2
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CROP_DIR = os.path.join(BASE_DIR, "data", "cropped")


# Create cropped directory if not exists
os.makedirs(CROP_DIR, exist_ok=True)

for category in os.listdir(RAW_DIR):
    raw_category_path = os.path.join(RAW_DIR, category)
    crop_category_path = os.path.join(CROP_DIR, category)

    os.makedirs(crop_category_path, exist_ok=True)

    for img_name in os.listdir(raw_category_path):
        img_path = os.path.join(raw_category_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_detector.process(rgb_img)

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Ensure bounding box is within image
                x, y = max(0, x), max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                face_crop = img[y:y+bh, x:x+bw]

                if face_crop.size == 0:
                    continue

                save_name = f"{os.path.splitext(img_name)[0]}_face{i}.jpg"
                save_path = os.path.join(crop_category_path, save_name)

                cv2.imwrite(save_path, face_crop)
