from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import traceback
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROCESSED_FOLDER"] = "processed"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["ALLOWED_IMAGE_EXTENSIONS"] = {"png", "jpg", "jpeg", "bmp"}
app.config["ALLOWED_VIDEO_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

TRAFFIC_SIGN_CLASSES = {
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
    10: "No passing veh > 3.5 tons",
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


def allowed_file(filename, file_type):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    if file_type == "image":
        return ext in app.config["ALLOWED_IMAGE_EXTENSIONS"]
    elif file_type == "video":
        return ext in app.config["ALLOWED_VIDEO_EXTENSIONS"]
    elif file_type == "detection":  # New type for image detection
        return ext in app.config["ALLOWED_IMAGE_EXTENSIONS"]
    return False


# -------- MODEL LOADING --------
def load_image_classification_model():
    model_path = "models/DenseNet169.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Image classification model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading image classification model:", str(e))
        return None


def load_object_detection_model():
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print("Detection model not found, skipping...")
        return None
    try:

        model = YOLO(model_path)
        print("Object detection model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading detection model:", str(e))
        return None


classification_model = None
detection_model = None


# -------- IMAGE PREPROCESSING --------
def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocess image for model input:
    - Converts BGR to RGB
    - Resizes to target_size
    - Normalizes by 255.0
    - Expands batch dimension
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Cannot read image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


# -------- CLASSIFICATION --------
def classify_traffic_sign(image_path):
    if classification_model is None:
        return {"success": False, "error": "Classification model not loaded"}

    try:
        img_array = preprocess_image(image_path, target_size=(64, 64))
        predictions = classification_model.predict(img_array)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])

        top5_indices = np.argsort(predictions[0])[-3:][::-1]
        all_predictions = {
            TRAFFIC_SIGN_CLASSES[i]: float(predictions[0][i]) for i in top5_indices
        }

        return {
            "success": True,
            "class": TRAFFIC_SIGN_CLASSES[class_idx],
            "confidence": confidence,
            "all_predictions": all_predictions,
        }

    except Exception:
        return {"success": False, "error": traceback.format_exc()}


# -------- DETECTION (NEW: Image Detection) --------
def detect_signs_in_image(image_path):
    """Detect multiple traffic signs in a single image using YOLO"""
    if detection_model is None:
        return {"success": False, "error": "Detection model not loaded"}

    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Cannot read image"}

        # Run YOLO detection
        results = detection_model(image)

        detections = []
        annotated_image = image.copy()

        # Process each detection
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = float(confs[i])
                class_name = detection_model.names[int(clss[i])]

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        # Save annotated image
        original_filename = os.path.basename(image_path)
        annotated_filename = f"detected_{original_filename}"
        annotated_path = os.path.join(
            app.config["PROCESSED_FOLDER"], annotated_filename
        )
        cv2.imwrite(annotated_path, annotated_image)

        # Get unique classes
        unique_classes = list(set([d["class"] for d in detections]))

        return {
            "success": True,
            "detections": detections,
            "annotated_image": annotated_filename,
            "total_detections": len(detections),
            "unique_classes": unique_classes,
        }

    except Exception:
        return {"success": False, "error": traceback.format_exc()}


# -------- VIDEO DETECTION --------
def detect_objects_in_video(video_path):
    if detection_model is None:
        return {"success": False, "error": "Detection model not loaded"}

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_skip = max(1, fps // 2)
        detections = []
        processed_frames = 0

        while cap.isOpened() and processed_frames < 100:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_num % frame_skip == 0:
                results = detection_model(frame)
                frame_detections = []
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy()
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        frame_detections.append(
                            {
                                "class": detection_model.names[int(clss[i])],
                                "confidence": float(confs[i]),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            }
                        )

                detections.append(
                    {
                        "frame": frame_num,
                        "timestamp": frame_num / fps,
                        "objects": frame_detections,
                    }
                )
                processed_frames += 1

        cap.release()

        summary = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "fps": fps,
            "duration": total_frames / fps if fps > 0 else 0,
            "unique_objects": list(
                set([obj["class"] for det in detections for obj in det["objects"]])
            ),
        }

        return {"success": True, "detections": detections[:50], "summary": summary}

    except Exception:
        return {"success": False, "error": traceback.format_exc()}


# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        analysis_type = request.form.get("type", "image")

        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename, analysis_type):
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        timestamp = str(int(tf.timestamp() * 1000))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Route to appropriate function based on type
        if analysis_type == "image":
            result = classify_traffic_sign(filepath)
        elif analysis_type == "detection":
            result = detect_signs_in_image(filepath)
        else:  # video
            result = detect_objects_in_video(filepath)

        result["filename"] = filename
        return jsonify(result)

    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "classification_model_loaded": classification_model is not None,
            "detection_model_loaded": detection_model is not None,
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Traffic Analysis System Starting...")
    print("=" * 50)

    print("Loading models...")
    classification_model = load_image_classification_model()
    detection_model = load_object_detection_model()
    print("Models ready.")
    print("\nServer running on http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5000)


# from flask import Flask, render_template, request, jsonify, send_from_directory
# import os
# import tensorflow as tf
# from werkzeug.utils import secure_filename
# import cv2
# import numpy as np
# import traceback

# app = Flask(__name__)

# # Configuration
# app.config["UPLOAD_FOLDER"] = "uploads"
# app.config["PROCESSED_FOLDER"] = "processed"
# app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = {"png", "jpg", "jpeg", "bmp"}
# app.config["ALLOWED_VIDEO_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

# TRAFFIC_SIGN_CLASSES = {
#     0: "Speed limit (20km/h)",
#     1: "Speed limit (30km/h)",
#     2: "Speed limit (50km/h)",
#     3: "Speed limit (60km/h)",
#     4: "Speed limit (70km/h)",
#     5: "Speed limit (80km/h)",
#     6: "End of speed limit (80km/h)",
#     7: "Speed limit (100km/h)",
#     8: "Speed limit (120km/h)",
#     9: "No passing",
#     10: "No passing veh > 3.5 tons",
#     11: "Right-of-way at intersection",
#     12: "Priority road",
#     13: "Yield",
#     14: "Stop",
#     15: "No vehicles",
#     16: "Veh > 3.5 tons prohibited",
#     17: "No entry",
#     18: "General caution",
#     19: "Dangerous curve left",
#     20: "Dangerous curve right",
#     21: "Double curve",
#     22: "Bumpy road",
#     23: "Slippery road",
#     24: "Road narrows on the right",
#     25: "Road work",
#     26: "Traffic signals",
#     27: "Pedestrians",
#     28: "Children crossing",
#     29: "Bicycles crossing",
#     30: "Beware of ice/snow",
#     31: "Wild animals crossing",
#     32: "End speed + passing limits",
#     33: "Turn right ahead",
#     34: "Turn left ahead",
#     35: "Ahead only",
#     36: "Go straight or right",
#     37: "Go straight or left",
#     38: "Keep right",
#     39: "Keep left",
#     40: "Roundabout mandatory",
#     41: "End of no passing",
#     42: "End no passing veh > 3.5 tons",
# }


# def allowed_file(filename, file_type):
#     if "." not in filename:
#         return False
#     ext = filename.rsplit(".", 1)[1].lower()
#     if file_type == "image":
#         return ext in app.config["ALLOWED_IMAGE_EXTENSIONS"]
#     elif file_type == "video":
#         return ext in app.config["ALLOWED_VIDEO_EXTENSIONS"]
#     return False


# # -------- MODEL LOADING --------
# def load_image_classification_model():
#     model_path = "models/DenseNet169.keras"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print("Image classification model loaded successfully.")
#         return model
#     except Exception as e:
#         print("Error loading image classification model:", str(e))
#         return None


# def load_object_detection_model():
#     model_path = "models/traffic_detection.pt"
#     if not os.path.exists(model_path):
#         print("Detection model not found, skipping...")
#         return None
#     try:
#         from ultralytics import YOLO

#         model = YOLO(model_path)
#         print("Object detection model loaded successfully.")
#         return model
#     except Exception as e:
#         print("Error loading detection model:", str(e))
#         return None


# classification_model = None
# detection_model = None


# # -------- IMAGE PREPROCESSING --------
# def preprocess_image(image_path, target_size=(64, 64)):
#     """
#     Preprocess image for model input:
#     - Converts BGR to RGB
#     - Resizes to target_size
#     - Normalizes by 255.0
#     - Expands batch dimension
#     """
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError("Cannot read image")

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, target_size)
#         image = image.astype("float32") / 255.0
#         image = np.expand_dims(image, axis=0)
#         return image
#     except Exception as e:
#         raise ValueError(f"Image preprocessing failed: {e}")


# # -------- CLASSIFICATION --------
# def classify_traffic_sign(image_path):
#     if classification_model is None:
#         return {"success": False, "error": "Classification model not loaded"}

#     try:
#         img_array = preprocess_image(image_path, target_size=(64, 64))
#         predictions = classification_model.predict(img_array)
#         class_idx = int(np.argmax(predictions[0]))
#         confidence = float(predictions[0][class_idx])

#         top5_indices = np.argsort(predictions[0])[-3:][::-1]
#         all_predictions = {
#             TRAFFIC_SIGN_CLASSES[i]: float(predictions[0][i]) for i in top5_indices
#         }

#         return {
#             "success": True,
#             "class": TRAFFIC_SIGN_CLASSES[class_idx],
#             "confidence": confidence,
#             "all_predictions": all_predictions,
#         }

#     except Exception:
#         return {"success": False, "error": traceback.format_exc()}


# # -------- DETECTION --------
# def detect_objects_in_video(video_path):
#     if detection_model is None:
#         return {"success": False, "error": "Detection model not loaded"}

#     try:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))

#         frame_skip = max(1, fps // 2)
#         detections = []
#         processed_frames = 0

#         while cap.isOpened() and processed_frames < 100:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#             if frame_num % frame_skip == 0:
#                 results = detection_model(frame)
#                 frame_detections = []
#                 for r in results:
#                     boxes = r.boxes.xyxy.cpu().numpy()
#                     confs = r.boxes.conf.cpu().numpy()
#                     clss = r.boxes.cls.cpu().numpy()
#                     for i in range(len(boxes)):
#                         x1, y1, x2, y2 = boxes[i]
#                         frame_detections.append(
#                             {
#                                 "class": detection_model.names[int(clss[i])],
#                                 "confidence": float(confs[i]),
#                                 "bbox": [int(x1), int(y1), int(x2), int(y2)],
#                             }
#                         )

#                 detections.append(
#                     {
#                         "frame": frame_num,
#                         "timestamp": frame_num / fps,
#                         "objects": frame_detections,
#                     }
#                 )
#                 processed_frames += 1

#         cap.release()

#         summary = {
#             "total_frames": total_frames,
#             "processed_frames": processed_frames,
#             "fps": fps,
#             "duration": total_frames / fps if fps > 0 else 0,
#             "unique_objects": list(
#                 set([obj["class"] for det in detections for obj in det["objects"]])
#             ),
#         }

#         return {"success": True, "detections": detections[:50], "summary": summary}

#     except Exception:
#         return {"success": False, "error": traceback.format_exc()}


# # -------- ROUTES --------
# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/upload", methods=["POST"])
# def upload_file():
#     try:
#         if "file" not in request.files:
#             return jsonify({"success": False, "error": "No file provided"}), 400

#         file = request.files["file"]
#         analysis_type = request.form.get("type", "image")

#         if file.filename == "":
#             return jsonify({"success": False, "error": "No file selected"}), 400

#         if not allowed_file(file.filename, analysis_type):
#             return jsonify({"success": False, "error": "Invalid file type"}), 400

#         filename = secure_filename(file.filename)
#         timestamp = str(int(tf.timestamp() * 1000))
#         filename = f"{timestamp}_{filename}"
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)

#         if analysis_type == "image":
#             result = classify_traffic_sign(filepath)
#         else:
#             result = detect_objects_in_video(filepath)

#         result["filename"] = filename
#         return jsonify(result)

#     except Exception:
#         return jsonify({"success": False, "error": traceback.format_exc()}), 500


# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# @app.route("/health")
# def health():
#     return jsonify(
#         {
#             "status": "healthy",
#             "classification_model_loaded": classification_model is not None,
#             # "detection_model_loaded": detection_model is not None,
#         }
#     )


# if __name__ == "__main__":
#     print("=" * 50)
#     print("Traffic Analysis System Starting...")
#     print("=" * 50)

#     print("Loading models...")
#     classification_model = load_image_classification_model()
#     # detection_model = load_object_detection_model()
#     print("Models ready.")
#     print("\nServer running on http://localhost:5000")
#     print("=" * 50)

#     app.run(debug=True, host="0.0.0.0", port=5000)


# # from flask import Flask, render_template, request, jsonify, send_from_directory
# # import os
# # import tensorflow as tf
# # from werkzeug.utils import secure_filename
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import traceback

# # app = Flask(__name__)

# # # Configuration
# # app.config["UPLOAD_FOLDER"] = "uploads"
# # app.config["PROCESSED_FOLDER"] = "processed"
# # app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# # app.config["ALLOWED_IMAGE_EXTENSIONS"] = {"png", "jpg", "jpeg", "bmp"}
# # app.config["ALLOWED_VIDEO_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

# # os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
# # os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

# # TRAFFIC_SIGN_CLASSES = {
# #     0: "Speed limit (20km/h)",
# #     1: "Speed limit (30km/h)",
# #     2: "Speed limit (50km/h)",
# #     3: "Speed limit (60km/h)",
# #     4: "Speed limit (70km/h)",
# #     5: "Speed limit (80km/h)",
# #     6: "End of speed limit (80km/h)",
# #     7: "Speed limit (100km/h)",
# #     8: "Speed limit (120km/h)",
# #     9: "No passing",
# #     10: "No passing veh > 3.5 tons",
# #     11: "Right-of-way at intersection",
# #     12: "Priority road",
# #     13: "Yield",
# #     14: "Stop",
# #     15: "No vehicles",
# #     16: "Veh > 3.5 tons prohibited",
# #     17: "No entry",
# #     18: "General caution",
# #     19: "Dangerous curve left",
# #     20: "Dangerous curve right",
# #     21: "Double curve",
# #     22: "Bumpy road",
# #     23: "Slippery road",
# #     24: "Road narrows on the right",
# #     25: "Road work",
# #     26: "Traffic signals",
# #     27: "Pedestrians",
# #     28: "Children crossing",
# #     29: "Bicycles crossing",
# #     30: "Beware of ice/snow",
# #     31: "Wild animals crossing",
# #     32: "End speed + passing limits",
# #     33: "Turn right ahead",
# #     34: "Turn left ahead",
# #     35: "Ahead only",
# #     36: "Go straight or right",
# #     37: "Go straight or left",
# #     38: "Keep right",
# #     39: "Keep left",
# #     40: "Roundabout mandatory",
# #     41: "End of no passing",
# #     42: "End no passing veh > 3.5 tons",
# # }


# # def allowed_file(filename, file_type):
# #     if "." not in filename:
# #         return False
# #     ext = filename.rsplit(".", 1)[1].lower()
# #     if file_type == "image":
# #         return ext in app.config["ALLOWED_IMAGE_EXTENSIONS"]
# #     elif file_type == "video":
# #         return ext in app.config["ALLOWED_VIDEO_EXTENSIONS"]
# #     return False


# # # -------- MODEL LOADING --------
# # def load_image_classification_model():
# #     model_path = "models/DenseNet169.keras"
# #     if not os.path.exists(model_path):
# #         raise FileNotFoundError(f"Model not found: {model_path}")
# #     try:
# #         model = tf.keras.models.load_model(model_path)
# #         print("Image classification model loaded successfully.")
# #         return model
# #     except Exception as e:
# #         print("Error loading image classification model:", str(e))
# #         return None


# # def load_object_detection_model():
# #     model_path = "models/traffic_detection.pt"
# #     if not os.path.exists(model_path):
# #         print("Detection model not found, skipping...")
# #         return None
# #     try:
# #         from ultralytics import YOLO

# #         model = YOLO(model_path)
# #         print("Object detection model loaded successfully.")
# #         return model
# #     except Exception as e:
# #         print("Error loading detection model:", str(e))
# #         return None


# # classification_model = None
# # detection_model = None


# # # -------- CLASSIFICATION --------
# # def classify_traffic_sign(image_path):
# #     if classification_model is None:
# #         return {"success": False, "error": "Classification model not loaded"}

# #     try:
# #         img = Image.open(image_path).convert("RGB")
# #         img = img.resize((80, 80))
# #         img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# #         predictions = classification_model.predict(img_array)
# #         class_idx = int(np.argmax(predictions[0]))
# #         confidence = float(predictions[0][class_idx])

# #         top5_indices = np.argsort(predictions[0])[-5:][::-1]
# #         all_predictions = {
# #             TRAFFIC_SIGN_CLASSES[i]: float(predictions[0][i]) for i in top5_indices
# #         }

# #         return {
# #             "success": True,
# #             "class": TRAFFIC_SIGN_CLASSES[class_idx],
# #             "confidence": confidence,
# #             "all_predictions": all_predictions,
# #         }

# #     except Exception:
# #         return {"success": False, "error": traceback.format_exc()}


# # # -------- DETECTION --------
# # def detect_objects_in_video(video_path):
# #     if detection_model is None:
# #         return {"success": False, "error": "Detection model not loaded"}

# #     try:
# #         cap = cv2.VideoCapture(video_path)
# #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #         fps = int(cap.get(cv2.CAP_PROP_FPS))

# #         frame_skip = max(1, fps // 2)
# #         detections = []
# #         processed_frames = 0

# #         while cap.isOpened() and processed_frames < 100:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
# #             if frame_num % frame_skip == 0:
# #                 results = detection_model(frame)
# #                 frame_detections = []
# #                 for r in results:
# #                     boxes = r.boxes.xyxy.cpu().numpy()
# #                     confs = r.boxes.conf.cpu().numpy()
# #                     clss = r.boxes.cls.cpu().numpy()
# #                     for i in range(len(boxes)):
# #                         x1, y1, x2, y2 = boxes[i]
# #                         frame_detections.append(
# #                             {
# #                                 "class": detection_model.names[int(clss[i])],
# #                                 "confidence": float(confs[i]),
# #                                 "bbox": [int(x1), int(y1), int(x2), int(y2)],
# #                             }
# #                         )

# #                 detections.append(
# #                     {
# #                         "frame": frame_num,
# #                         "timestamp": frame_num / fps,
# #                         "objects": frame_detections,
# #                     }
# #                 )
# #                 processed_frames += 1

# #         cap.release()

# #         summary = {
# #             "total_frames": total_frames,
# #             "processed_frames": processed_frames,
# #             "fps": fps,
# #             "duration": total_frames / fps if fps > 0 else 0,
# #             "unique_objects": list(
# #                 set([obj["class"] for det in detections for obj in det["objects"]])
# #             ),
# #         }

# #         return {"success": True, "detections": detections[:50], "summary": summary}

# #     except Exception:
# #         return {"success": False, "error": traceback.format_exc()}


# # # -------- ROUTES --------
# # @app.route("/")
# # def index():
# #     return render_template("index.html")


# # @app.route("/upload", methods=["POST"])
# # def upload_file():
# #     try:
# #         if "file" not in request.files:
# #             return jsonify({"success": False, "error": "No file provided"}), 400

# #         file = request.files["file"]
# #         analysis_type = request.form.get("type", "image")

# #         if file.filename == "":
# #             return jsonify({"success": False, "error": "No file selected"}), 400

# #         if not allowed_file(file.filename, analysis_type):
# #             return jsonify({"success": False, "error": "Invalid file type"}), 400

# #         filename = secure_filename(file.filename)
# #         timestamp = str(int(tf.timestamp() * 1000))
# #         filename = f"{timestamp}_{filename}"
# #         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
# #         file.save(filepath)

# #         if analysis_type == "image":
# #             result = classify_traffic_sign(filepath)
# #         else:
# #             result = detect_objects_in_video(filepath)

# #         result["filename"] = filename
# #         return jsonify(result)

# #     except Exception:
# #         return jsonify({"success": False, "error": traceback.format_exc()}), 500


# # @app.route("/uploads/<filename>")
# # def uploaded_file(filename):
# #     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# # @app.route("/health")
# # def health():
# #     return jsonify(
# #         {
# #             "status": "healthy",
# #             "classification_model_loaded": classification_model is not None,
# #             "detection_model_loaded": detection_model is not None,
# #         }
# #     )


# # if __name__ == "__main__":
# #     print("=" * 50)
# #     print("Traffic Analysis System Starting...")
# #     print("=" * 50)

# #     print("Loading models...")
# #     classification_model = load_image_classification_model()
# #     detection_model = load_object_detection_model()
# #     print("Models ready.")
# #     print("\nServer running on http://localhost:5000")
# #     print("=" * 50)

# #     app.run(debug=True, host="0.0.0.0", port=5000)
