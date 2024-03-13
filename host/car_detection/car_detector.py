import torch
import cv2
import numpy as np
import time
from PIL import Image


class ObjectDetector:
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s', device=None):
        print ("Object Detector Initialization...")
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tracker = EuclideanDistTracker()
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.class_labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        for i in range(len(self.class_labels)):
            print(f'{i}: {self.class_labels[i]}')

        
        if self.device == 'cuda':
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a
            dummy_tensor = torch.zeros((1, 3, 640, 640)).to(self.device)
            dummy_out = self.model(dummy_tensor)
            print(f'total memory: {t/1024**3} GB, reserved memory: {r/1024**3} GB, allocated memory: {a/1024**3} GB, free memory: {f/1024**3} GB')
            dummy_out = dummy_out.to('cpu')

    def pred_bbox(self, frame):

        assert frame is not None, "Frame is None"

        resized_frame = cv2.resize(frame, (640, 480))

        results = self.model(resized_frame)

        bbox = [(x[0]*frame.shape[1]/640, x[1]*frame.shape[0]/480, x[2]*frame.shape[1]/640, x[3]*frame.shape[0]/480, x[5]) for x in results.xyxy[0]]

        return bbox
    

    def warning(self, frame):
        bbox = self.pred_bbox(frame)
        bbox.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        if len(bbox) <= 0:
            return "No oncoming object detected"

        top3 = bbox[:3]

        warning_string = f""
        for box in top3:
            warning_string += f"{self.class_labels[int(box[4])]}:"

            image_center_x = frame.shape[1] / 2

            box_center_x = (box[0] + box[2]) / 2

            position_x = box_center_x - image_center_x

            if position_x < -0.3 * image_center_x or position_x > 0.3 * image_center_x:
                if position_x < -0.3 * image_center_x:
                    warning_string += " left\n"
                else:
                    warning_string += " right\n"
            else:
                warning_string += " center\n"

        return warning_string
        

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

            tic = time.time()
            bbox = self.pred_bbox(frame)
            toc = time.time()
            time_total += toc - tic
            frame_count += 1

            # Sort the bounding boxes by their area in descending order
            bbox.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

            # Get the largest bounding boxe
            if len(bbox) <= 0:
                print ("No oncoming object detected")
                continue
            box = bbox[0]

            # Calculate the center of the image
            image_center_x = frame.shape[1] / 2
            image_center_y = frame.shape[0] / 2

            # Calculate the position of each bounding box in relation to the center of the image
            warning_string = f"{self.class_labels[int(box[4])]} "
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            position_x = box_center_x - image_center_x
            position_x = box_center_x - image_center_x
            if position_x < -0.3 * image_center_x or position_x > 0.3 * image_center_x:
                if position_x < -0.3 * image_center_x:
                    warning_string += "to your left"        
                else:
                    warning_string += "to your right"
            else:
                warning_string += "in front of you"
           
            
            print (warning_string)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
            cv2.putText(frame, warning_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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
        print(f'Average time taken to process a frame: {(time_total/frame_count):.2f}ms')
        cap.release()
        cv2.destroyAllWindows()
    
        
if __name__ == "__main__":
    cap = cv2.VideoCapture('test_data/fullstreet.mp4')
    fps = 30
    detector = ObjectDetector()
    detector.demo(cap=cap, fps=fps)
    cap.release()