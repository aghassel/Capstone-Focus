import argparse
from fastapi.testclient import TestClient
from api import app
import cv2
import numpy as np

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_predict(image_path):
    # Load an image file
    img = cv2.imread(image_path)
    _, img_encoded = cv2.imencode('.jpg', img)
    data = img_encoded.tostring()

    # Create a file-like object from the image
    file = (image_path, data)

    response = client.post("/predict", files={'file': file})
    assert response.status_code == 200

    # Check if the response is a list of lists (bounding boxes)
    prediction = response.json()['prediction']
    assert isinstance(prediction, list)
    for bbox in prediction:
        assert isinstance(bbox, list)
        assert len(bbox) == 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test API with image files.')
    parser.add_argument('image_paths', type=str, nargs='+', help='Paths to image files')
    args = parser.parse_args()

    test_read_root()
    for image_path in args.image_paths:
        test_predict(image_path)


# Usage: 
# python api_test.py image1.jpg image2.jpg image3.jpg