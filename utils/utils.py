import cv2
import os

#### VIDEO UTILS #####
def read_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: # no more frames to read
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_name):
    
    folder = 'output'
    if not os.path.exists(folder):
        os.makedirs(folder)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(folder, output_video_name), fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


#### BOX UTILS #####

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_indix in keypoint_indices:
       keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_indix
    
   return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))


#### OTHER ####

def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, refrence_height_in_pixels):
    return (meters * refrence_height_in_pixels) / refrence_height_in_meters

