import cv2.cv2
import fastapi 
import uvicorn
import numpy as np
import logging
import tempfile
import os
import cv2
import mediapipe as mp


# 1- Extract frames 
def extract_frames(video_path):
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret , frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def calculate_motion(frames):
    prev_gray = cv2.cvtColor(frames[0] , cv2.COLOR_BAYER_BG2BGRA)
    motion_scores = []
    
    for i in range(1 , len(frames)):
        curr_gray = cv2.cvtColor(frames[i] , cv2.COLOR_BAYER_BG2BGRA)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None ,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
            )
        
        motion_magnitude = cv2.magnitude(flow[..., 0 ] , flow[... , 1])
        motion_scores.append(motion_magnitude.mean())
        prev_gray = curr_gray
        
    return motion_scores



