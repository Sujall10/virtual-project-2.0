import cv2
import mediapipe as mp
import math
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle(point1, point2):
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

def overlay_image(background, overlay, top_left_x, top_left_y, overlay_size=None):
    bg_h, bg_w, bg_channels = background.shape
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    rows, cols = h, w

    if top_left_x < 0 or top_left_y < 0 or top_left_x + w > bg_w or top_left_y + h > bg_h:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:]

    background_part = background[top_left_y:top_left_y+rows, top_left_x:top_left_x+cols]

    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(background_part, background_part, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    dst = cv2.add(img_bg, img_fg)
    background[top_left_y:top_left_y+rows, top_left_x:top_left_x+cols] = dst

    return background

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 360-angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


img_file = {
    1: 'glasses/glass3.png',
    2: 'glasses/glass4.png',
    3: 'glasses/glass5.png',
    4: 'glasses/glass6.png',
    5: 'glasses/glass7.png',
    6: 'glasses/glass8.png',
    7: 'glasses/glass9.png',
    8: 'glasses/glass10.png',
    9: 'glasses/glass11.png',
    10: 'glasses/glass14.png',
    11: 'glasses/glass16.png',
    12: 'glasses/glass17.png',
    13: 'glasses/glass22.png',
    14: 'glasses/glass24.png',
    15: 'glasses/glass28.png'
}

glasses_dict = {key: cv2.imread(file, cv2.IMREAD_UNCHANGED) for key, file in img_file.items()}
current_glasses_key = 1

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []

            for landmark_index in [33, 263]:  # Left eye corner and right eye corner
                landmark = face_landmarks.landmark[landmark_index]
                h, w, c = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
                # cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                
            if len(landmark_points) == 2:
                distance = calculate_distance(landmark_points[0], landmark_points[1])
                angle = calculate_angle(landmark_points[0], landmark_points[1])

                center_x = int((landmark_points[0][0] + landmark_points[1][0]) / 2)
                center_y = int((landmark_points[0][1] + landmark_points[1][1]) / 2)
                
                glasses = glasses_dict.get(current_glasses_key)
                if glasses is not None:
                    glasses_width = int(distance * 1.5)  
                    glasses_height = int(glasses.shape[0] * (glasses_width / glasses.shape[1]))
                    resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))

                    rotated_glasses = rotate_image(resized_glasses, angle)

                    top_left_x = center_x - rotated_glasses.shape[1] // 2
                    top_left_y = center_y - rotated_glasses.shape[0] // 2

                    image = overlay_image(image, rotated_glasses, top_left_x, top_left_y)

    cv2.imshow('MediaPipe Face Mesh', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        current_glasses_key = current_glasses_key +1
    elif key == ord('s'):
        current_glasses_key = current_glasses_key -1
    
    

cap.release()
cv2.destroyAllWindows()
