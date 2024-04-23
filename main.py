from utils import (read_video, 
                   save_video,
                   measure_distance,
                   convert_pixel_distance_to_meters
                   )
import globals
from trackers import PlayerTracker, BallTracker
from court_detector import CourtDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main(first_time=True):
    # Read Video
    input_video_path = "input/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='trackers/yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, first_time=False)

    ball_tracker = BallTracker(model_path='models/last.pt')    
    ball_detections = ball_tracker.detect_frames(video_frames, first_time=False)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    
    # Court Line Detector model
    court_model_path = "models/keypoints_court.pth"
    court_line_detector = CourtDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.player_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.ball_boxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    save_video(output_video_frames, "output_video.avi")

if __name__ == "__main__":
    main()