from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import logging

app = FastAPI()

# Setup logging/directories
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Config templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load face detection model
try:
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )
    logger.info("Successfully loaded face detection model")
except Exception as e:
    logger.error(f"Failed to load face detection model: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    filter_type: str = Form("low_pass")
):
    try:
        logger.info(f"Starting processing for file: {file.filename}")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        logger.info(f"Image loaded successfully. Dimensions: {image.shape}")

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        logger.info(f"Found {detections.shape[2]} potential faces")

        faces_processed = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                face_region = image[y:y2, x:x2]
                if filter_type == "low_pass":
                    image[y:y2, x:x2] = cv2.GaussianBlur(face_region, (99, 99), 30)
                else:  
                    blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
                    image[y:y2, x:x2] = cv2.addWeighted(face_region, 1.5, blurred, -0.5, 0)
                
                faces_processed += 1

        logger.info(f"Processed {faces_processed} faces with {filter_type} filter")
        
        processed_path = f"uploads/processed_{file.filename}"
        cv2.imwrite(processed_path, image)
        logger.info(f"Saved processed image to {processed_path}")
        
        return FileResponse(processed_path)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")