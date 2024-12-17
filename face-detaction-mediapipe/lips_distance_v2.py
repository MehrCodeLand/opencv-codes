from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import random

app = FastAPI()
video_path = './api/IMG_6251.mp4'

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

@app.post("/detect-speaking")
async def detect_speaking(speaking_threshold: float = 5.0):
    try:
        lip_distance = 0
        random_images = select_random_frames()

        for img in random_images:
            # Convert the image to RGB (MediaPipe requires RGB format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform face mesh detection
            result = face_mesh.process(img_rgb)

            if not result.multi_face_landmarks:
                raise HTTPException(status_code=400, detail="No face detected in the image")

            # Get face landmarks
            landmarks = result.multi_face_landmarks[0].landmark

            # Extract key landmarks for upper and lower lips
            upper_lip = landmarks[13]  # Index 13 corresponds to upper lip center
            lower_lip = landmarks[14]  # Index 14 corresponds to lower lip center

            # Calculate the distance between upper and lower lips
            height, width, _ = img.shape
            upper_lip_coords = (int(upper_lip.x * width), int(upper_lip.y * height))
            lower_lip_coords = (int(lower_lip.x * width), int(lower_lip.y * height))
            lip_distance += np.linalg.norm(np.array(upper_lip_coords) - np.array(lower_lip_coords))
        
        lip_average = lip_distance / 10

        return JSONResponse(content={
            "lip_distance": lip_average
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def select_random_frames(num_frames=10):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select random frame numbers
    random_frame_numbers = random.sample(range(total_frames), num_frames)

    # Read the selected frames
    frames = []
    x = 0 
    for frame_number in random_frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        x = x + 1
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            cv2.imwrite(f'm{x}.jpeg' , frame)    
    # Release the video capture object
    video.release()

    return frames

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
