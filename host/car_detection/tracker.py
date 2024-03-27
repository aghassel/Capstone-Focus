import math

class EuclideanDistTracker:
    def __init__(self, threshold=30):
        # Id and cente of each object
        self.center_points = {}
        self.id_count = 0
        self.threshold = threshold

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h, lbl = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.threshold:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, lbl, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, lbl, self.id_count])
                self.id_count += 1

        #clean the dictionary 
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        #update the dictionary
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
