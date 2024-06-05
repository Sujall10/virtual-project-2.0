import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Capture video from the webcam
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
            for landmark_index in [68, 298]:
              
                landmark = face_landmarks.landmark[landmark_index]
                h, w, c = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
                
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            if len(landmark_points) == 2:
                distance = calculate_distance(landmark_points[0], landmark_points[1])
                
                cv2.putText(image, f"Distance: {distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''                if landmark_index == 68:
                    x1 = {x}
                    y1 = {y}
                    landmark_points[0] = {x1,y1}
                if landmark_index == 298:
                    x2 = {x}
                    y2 = {y}
                    landmark_points[1]= {x2,y2}
                    # print(f"Landmark {landmark_index} coordinate y: (y={y})")
            if len(landmark_points) == 2:
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)'''