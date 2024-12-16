


# distance between upper and lower lips 
def is_mouth_moving(mouth_landmarks, previous_mouth_distance, threshold=2.0):
    # Check if there are enough landmarks
    if len(mouth_landmarks) < 4:
        return False, previous_mouth_distance

    # Calculate upper and lower lip distances
    upper_lip = np.mean([mouth_landmarks[i] for i in range(2)], axis=0)
    lower_lip = np.mean([mouth_landmarks[i] for i in range(2, 4)], axis=0)

    # Compute mouth distance and detect movement
    current_mouth_distance = dist.euclidean(upper_lip, lower_lip)
    if abs(current_mouth_distance - previous_mouth_distance) > threshold:
        return True, current_mouth_distance
    return False, current_mouth_distance

