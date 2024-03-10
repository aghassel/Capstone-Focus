from fastapi import FastAPI, UploadFile, File
from car_detector import CarDetector
import cv2
import numpy as np

app = FastAPI()

# Create a CarDetector object
detector = CarDetector()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Convert the file to an opencv image
    image_stream = np.fromstring(await file.read(), np.uint8)
    frame = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

    # Get the bounding boxes
    bbox = detector.predict_single_frame(frame)

    # Convert the bounding boxes to a list of lists for JSON serialization
    bbox = bbox.tolist()

    return {"prediction": bbox}