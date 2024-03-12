import torch
import cv2
import numpy as np
from tracker import EuclideanDistTracker
import time
from PIL import Image



class CarDetector:
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s', device=None):
        print ("CarDetector __init__")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = EuclideanDistTracker()
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
        self.model = self.model.to(self.device)
        if self.device == 'cuda':
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a
            dummy_tensor = torch.zeros((1, 3, 640, 640)).to(self.device)
            dummy_out = self.model(dummy_tensor)
            print(f'total memory: {t/1024**3} GB, reserved memory: {r/1024**3} GB, allocated memory: {a/1024**3} GB, free memory: {f/1024**3} GB')
            dummy_out = dummy_out.to('cpu')


    def pred_bbox(self, frame, threshold=None):
        assert frame is not None, "Frame is None"
        assert threshold >= 0 and threshold <= 1, "Treshold must be between 0 and 1"
        
   
        resized_frame = cv2.resize(frame, (640, 480))
        
        results = self.model(resized_frame)

    
        results = [x for x in results.xyxy[0] if x[5] == 2]

        bbox = [(x[0]*frame.shape[1]/640, x[1]*frame.shape[0]/480, x[2]*frame.shape[1]/640, x[3]*frame.shape[0]/480) for x in results]

        bbox = self.tracker.update(bbox)

        if threshold is not None:
            threshold_position = int(frame.shape[1] * (1 - threshold))
            bbox = [i for i in bbox if i[0] < threshold_position]
            bbox = [(i[0], i[1], threshold_position, i[3]) if i[2] > threshold_position else i for i in bbox]

        return bbox
    
    def predict_single_frame(self, frame, threshold=None):
        bbox = self.pred_bbox(frame, threshold)
        # create a empty black image in cv2
        black_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        for i in bbox:
            cv2.rectangle(black_image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 255, 255), 2)
            
        
        # convert to pillow image
        black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        black_image = Image.fromarray(black_image)

        return black_image, bbox   

    def return_warning(self, frame):  
        bbox = self.pred_bbox(frame)
        if bbox:
            max_bbox = max(bbox, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            bbox_width = max_bbox[2] - max_bbox[0]
            if bbox_width < 100:
                return "Car is far away"
            elif bbox_width < 200:
                return "Car is at a moderate distance"
            else:
                return "Car is close"
        else:
            return "No car detected"

        

    def demo(self, cap, fps=30, threshold=None):
        delay = 1 / fps
        frame_count = 0
        time_total = 0
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # make frame a black image
            black_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            frame = black_image
            
            tic = time.time()
            black_image, bbox = self.predict_single_frame(frame, threshold)
            toc = time.time()
            time_total += toc - tic
            frame_count += 1
            black_image = np.array(black_image)


            for i in bbox:
                cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
                cv2.putText(frame, 'Car', (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            if threshold is not None:
                threshold_position = int(frame.shape[1] * (1 - threshold))
                cv2.line(frame, (threshold_position, 0), (threshold_position, frame.shape[0]), (0, 0, 255), 2)
                cv2.line(black_image, (threshold_position, 0), (threshold_position, frame.shape[0]), (0, 0, 255), 2)


            cv2.imshow('Frame', frame)
            cv2.imshow('Black Image', black_image)

            # Calculate the time taken to process the frame
            time_taken = time.time() - start_time

            # If the frame was processed quicker than the delay, wait for the remaining time
            if time_taken < delay:
                time.sleep(delay - time_taken)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # convert total time to milliseconds
        time_total *= 1000
        print(f'Average time taken to process a frame: {(time_total/frame_count):.2f}ms')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('test_data/cars.mp4')
    fps = 30
    detector = CarDetector()
    detector.demo(cap=cap, fps=fps, threshold=0.4)
    cap.release()