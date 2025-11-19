import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np

# === FIX ĐƯỜNG DẪN MODEL TUYỆT ĐỐI ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DEF = os.path.join(BASE_DIR, "nsfw_model", "deploy.prototxt")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "nsfw_model", "resnet_50_1by2_nsfw.caffemodel")

print(f"Đang tìm model:")
print(MODEL_DEF)
print(MODEL_WEIGHTS)

if not os.path.exists(MODEL_DEF):
    raise FileNotFoundError(f"Không tìm thấy deploy.prototxt tại: {MODEL_DEF}")
if not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"Không tìm thấy caffemodel tại: {MODEL_WEIGHTS}")

print("Đang load model Caffe...")
net = cv2.dnn.readNetFromCaffe(MODEL_DEF, MODEL_WEIGHTS)
print("Model loaded thành công!")

app = FastAPI(title="NSFW Detector - OpenCV Caffe")

def classify_image(image_bytes: bytes) -> float:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể decode ảnh")
    
    blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=False, crop=False)
    net.setInput(blob)
    pred = net.forward()
    return float(pred[0][1])

@app.post("/classify_nsfw")
async def classify_nsfw(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Chỉ nhận file ảnh")
    
    contents = await file.read()
    score = classify_image(contents)
    
    return {
        "filename": file.filename,
        "nsfw_score": round(score, 6),
        "is_nsfw": score > 0.58
    }

@app.get("/")
def root():
    return {"status": "NSFW API đang chạy ngon lành!"}