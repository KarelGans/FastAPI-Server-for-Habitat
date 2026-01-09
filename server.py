from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io
import base64
from object_detection import ObjectDetector
from PIL import Image
import os
import re
import shutil  # for moving files
import threading

app = FastAPI(title="YOLOe Object Detection API")

ANNOTATED_LOG_DIR = "annotated_img_logs"
os.makedirs(ANNOTATED_LOG_DIR, exist_ok=True)  # ensure folder exists



@app.post("/detect/")
async def detect_object(
    file: UploadFile = File(...),
    object_name: str = Form(...)
):
    # Load image into PIL
    try:
        image = Image.open(file.file).convert("RGB")
        image.save("original_img.jpg")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run detection and save temp.jpg
    temp_path = "temp.jpg"
    detector = ObjectDetector(checkpoint="yoloe-v8l-seg.pt", device="cuda:0")
    result = detector.ObjectDetection(img_source="original_img.jpg", names=[object_name], output=temp_path)

    # Convert annotated PIL image to JPEG bytes for API response
    buf = io.BytesIO()
    result["image"].save(buf, format="JPEG")
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    # ---------------- Save annotated image to logs ----------------
    has_obj_str = "true" if result["hasObject"] else "false"

    # List existing files for this object+hasObject combination
    pattern = re.compile(rf"^{re.escape(object_name)}_{has_obj_str}_(\d+)\.jpg$")
    existing_files = [
        f for f in os.listdir(ANNOTATED_LOG_DIR) if pattern.match(f)
    ]

    if existing_files:
        # Get the highest existing number
        numbers = [int(pattern.match(f).group(1)) for f in existing_files]
        next_number = max(numbers) + 1
    else:
        next_number = 1

    # Final save path in logs
    save_path = os.path.join(
        ANNOTATED_LOG_DIR, f"{object_name}_{has_obj_str}_{next_number}.jpg"
    )

    # Move the temp file to logs
    shutil.move(temp_path, save_path)

    payload = {
        "hasObject": result["hasObject"],
        "objectCoordinates": result["objectCoordinates"],
        "annotatedImageBase64": img_b64,
        "savedPath": save_path
    }

    return JSONResponse(content=payload)
