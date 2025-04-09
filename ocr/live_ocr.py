import cv2 as cv
import easyocr  

reader = easyocr.Reader(['en'])
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0
ocr_frequency = 10  # Perform OCR every 10 frames

while True:
    ret, frame = cap.read()
    
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Apply OCR every nth frame
    if frame_count % ocr_frequency == 0:
        small_frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        result = reader.readtext(small_frame)
    
    # Draw rectangles on detected text
    for res in result:
        (top_left, top_right, bottom_right, bottom_left) = res[0]
        top_left = tuple([int(coord) for coord in top_left])
        bottom_right = tuple([int(coord) for coord in bottom_right])
        print(res)
        
        cv.rectangle(small_frame, top_left, bottom_right, (0, 0, 255), 2)
        text = res[1]
    # Display the frame
    cv.imshow('OCR Video', small_frame)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv.destroyAllWindows()
https://linkssh.xyz/link/2Jb1xgKiXdFo0WHELJSn
