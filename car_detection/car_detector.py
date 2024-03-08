import torch
import torchvision
import cv2
import numpy as np
from tracker import EuclideanDistTracker
import time

class CarDetector:
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s', device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = EuclideanDistTracker()
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
        self.model = self.model.to(self.device)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a
        dummy_tensor = torch.zeros((1, 3, 640, 640)).to(self.device)
        dummy_out = self.model(dummy_tensor)
        print(f'total memory: {t/1024**3} GB, reserved memory: {r/1024**3} GB, allocated memory: {a/1024**3} GB, free memory: {f/1024**3} GB')
        dummy_out = dummy_out.to('cpu')


    def predict_single_frame(self, frame):
        # Resize the frame before making predictions
        resized_frame = cv2.resize(frame, (1280, 720))
        results = self.model(resized_frame)

        # Filter results for car class (class index 2)
        results = [x for x in results.xyxy[0] if x[5] == 2]

        # Scale the bounding box coordinates back to the original frame size
        bbox = [(x[0]*frame.shape[1]/1280, x[1]*frame.shape[0]/720, x[2]*frame.shape[1]/1280, x[3]*frame.shape[0]/720) for x in results]

        bbox = self.tracker.update(bbox)

        return bbox

    def demo(self, cap, fps=30):
        delay = 1 / fps
        frame_count = 0
        time_total = 0
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            
            tic = time.time()
            bbox = self.predict_single_frame(frame)
            toc = time.time()
            time_total += toc - tic
            frame_count += 1

            for i in bbox:
                cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
                cv2.putText(frame, 'Car', (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('Frame', frame)

            # Calculate the time taken to process the frame
            time_taken = time.time() - start_time

            # If the frame was processed quicker than the delay, wait for the remaining time
            if time_taken < delay:
                time.sleep(delay - time_taken)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # convert total time to milliseconds
        time_total *= 1000
        print(f'Average time taken to process a frame: {round(time_total/frame_count),2}ms')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('test_data/cars.mp4')
    fps = 30
    detector = CarDetector()
    detector.demo(cap=cap, fps=fps)
    cap.release()