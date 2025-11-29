import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model

# Load YOLO detector
yolo_model = YOLO("src/yolov8n.pt")


# Load DenseNet classifier
clf_model = load_model(r"D:\AI-MachineLearning\Ml-project\models\DenseNet169.keras")

# Class mapping
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}

IMAGE_SIZE = (64, 64)

# Input image
img_path = r"C:\Users\Abdo\Downloads\meanings-of-shapes-and-colours-in-traffic-signs-in-india.png"
orig_img = cv2.imread(img_path)

# Run YOLO detection
results = yolo_model.predict(img_path)

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # Crop detection
        crop = orig_img[y1:y2, x1:x2]

        # Preprocess for DenseNet
        crop_resized = cv2.resize(crop, IMAGE_SIZE)
        crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_resized = crop_resized.astype("float32") / 255.0
        crop_resized = np.expand_dims(crop_resized, axis=0)

        # Classification
        preds = clf_model.predict(crop_resized)
        pred_idx = np.argmax(preds)
        pred_class = classes[pred_idx]
        confidence = preds[0][pred_idx]

        # Draw on original image
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            orig_img,
            f"{pred_class} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

# Show final image
cv2.imshow("YOLO + DenseNet", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("final_result.jpg", orig_img)
