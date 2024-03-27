import torch
import cv2
import numpy as np
import time
from PIL import Image
try:
    from car_detection.tracker import EuclideanDistTracker
except:
    from tracker import EuclideanDistTracker


class ObjectDetector:
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s', device=None):
        print ("Object Detector Initialization...")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.tracker = EuclideanDistTracker(threshold=50)
        self.class_labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        for i in range(len(self.class_labels)):
            print(f'{i}: {self.class_labels[i]}')

        self.hist_left = None
        self.hist_center = None
        self.hist_right = None

        self.prev_detections = []

        
        if self.device == 'cuda':
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a
            dummy_tensor = torch.zeros((1, 3, 640, 640)).to(self.device)
            dummy_out = self.model(dummy_tensor)
            print(f'total memory: {t/1024**3} GB, reserved memory: {r/1024**3} GB, allocated memory: {a/1024**3} GB, free memory: {f/1024**3} GB')
            dummy_out = dummy_out.to('cpu')

    def pred_bbox(self, frame, conf_thres=0.50):

        assert frame is not None, "Frame is None"

        resized_frame = cv2.resize(frame, (640, 480))

        results = self.model(resized_frame)

        confidences = results.xyxy[0][:, 4]

        results.xyxy[0] = results.xyxy[0][confidences > conf_thres]

        #for i in range(len(results.xyxy[0])):
        #    print(f'{self.class_labels[int(results.xyxy[0][i][5])]}: {results.xyxy[0][i][4]}')

        bbox = [(x[0]*frame.shape[1]/640, x[1]*frame.shape[0]/480, x[2]*frame.shape[1]/640, x[3]*frame.shape[0]/480, x[5]) for x in results.xyxy[0]]

        return bbox
    

    
    def warning(self, frame):

        bbox = self.pred_bbox(frame)
        bbox = self.tracker.update(bbox)
     
        if len(bbox) == 0:
            return "Unknown", None, None, None

        # get a list of the center locations of each object
        centers = [(int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2)) for x in bbox]

        # splits bbox into three lists for center left and right
        left, center, right = [], [], []

        for i in range(len(centers)):
            if centers[i][0] < frame.shape[1] / 3:
                left.append(bbox[i])
            elif centers[i][0] > 2 * frame.shape[1] / 3:
                right.append(bbox[i])
            else:
                center.append(bbox[i])

        # get the largest bounding boxes from each list
        left = max(left, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])) if len(left) > 0 else None

        center = max(center, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])) if len(center) > 0 else None

        right = max(right, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])) if len(right) > 0 else None

        warning = f""

        if left is not None and left != self.hist_left and left[5] not in self.prev_detections:
            warning += f"{self.class_labels[int(left[4])]}: left\n"
            self.hist_left = left
            self.prev_detections.append(left[5])  

        if center is not None and center != self.hist_center and center[5] not in self.prev_detections:
            warning += f"{self.class_labels[int(center[4])]}: center\n"
            self.hist_center = center
            self.prev_detections.append(center[5])

        if right is not None and right != self.hist_right and right[5] not in self.prev_detections:
            warning += f"{self.class_labels[int(right[4])]}: right\n"
            self.hist_right = right
            self.prev_detections.append(right[5])

        return warning, left, center, right


    def demo(self, cap, fps=30):
        assert cap is not None, "Cap is None"

        delay = 1 / fps
        frame_count = 0
        time_total = 0
        while True:

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count < 200:
                frame_count += 1
                continue

            
            tic = time.time()
            bbox = self.pred_bbox(frame)
            toc = time.time()
            time_total += toc - tic
            frame_count += 1

            warning_string, left, center, right = self.warning(frame)

            if left is not None:
                cv2.rectangle(frame, (int(left[0]), int(left[1])), (int(left[2]), int(left[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{left[5]}", (int(left[0]), int(left[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if center is not None:
                cv2.rectangle(frame, (int(center[0]), int(center[1])), (int(center[2]), int(center[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{center[5]}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if right is not None:
                cv2.rectangle(frame, (int(right[0]), int(right[1])), (int(right[2]), int(right[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{right[5]}", (int(right[0]), int(right[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, warning_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
            cv2.imshow('Frame', frame)

            print (warning_string) 
           
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
    cap = cv2.VideoCapture('test_data/fullstreet.mp4')
    fps = 30
    detector = ObjectDetector()
    detector.demo(cap=cap, fps=fps)
    cap.release()