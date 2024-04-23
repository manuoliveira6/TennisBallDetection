from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
import os

class BallTracker:
    def __init__(self, model_path, conf=0.1):
        self.model = YOLO(model_path)
        self.conf = conf

    def interpolate_ball_positions(self, ball_detections):

        ball_detections = [x.get(1,[]) for x in ball_detections]
        df = pd.DataFrame(ball_detections,columns=['x1','y1','x2','y2'])

        df = df.interpolate()
        df = df.bfill()

        ball_detections = [{1:x} for x in df.to_numpy().tolist()]

        return ball_detections

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits
    

    def detect_frames(self, frames, first_time=True):
        
        ball_detections = []
        folder = 'ball_detections'
        save_path = f'../{folder}/ball_detections.pkl'

        if not os.path.exists(folder):
            os.makedirs(folder)

        if not first_time:
            with open(save_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        else:
            for frame in frames:
                ball_detections.append(self.detect_frame(frame))

                with open(save_path, 'wb') as f:
                    pickle.dump(ball_detections, f)

        return ball_detections
    

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=self.conf)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    
    def ball_boxes(self, video_frames, ball_detections):

        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, box in ball_dict.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Ball ID: {track_id}", (int(box[0]), int(box[1] -10 )), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    