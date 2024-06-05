import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def overlay_image(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w, bg_channels = background.shape
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    rows, cols = h, w

    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:]

    background_part = background[y:y+rows, x:x+cols]

    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(background_part, background_part, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    dst = cv2.add(img_bg, img_fg)
    background[y:y+rows, x:x+cols] = dst

    return background

glasses = cv2.imread('glasses/glass5.png') 

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image color to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to get the face landmarks
    results = face_mesh.process(image)

    # Convert the image color back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            for landmark_index in [68, 298]:
                # Get the landmark coordinates
                landmark = face_landmarks.landmark[landmark_index]
                h, w, c = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
                # Draw a circle at the landmark
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                # Print the coordinates of the landmark
                print(f"Landmark {landmark_index} coordinates: (x={x}, y={y})")

            # Calculate and display the distance between landmarks 68 and 298
            if len(landmark_points) == 2:
                distance = calculate_distance(landmark_points[0], landmark_points[1])
                # Display the distance on the image
                cv2.putText(image, f"Distance: {distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # Print the distance
                print(f"Distance between landmarks 68 and 298: {distance:.2f}")
                center_x = int((landmark_points[0][0] + landmark_points[1][0]) / 2)
                center_y = int((landmark_points[0][1] + landmark_points[1][1]) / 2)

                # Resize the glasses image based on the distance
                glasses_width = int(distance * 1.5)  # Adjust the scaling factor as needed
                glasses_height = int(glasses.shape[0] * (glasses_width / glasses.shape[1]))
                resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))

                # Calculate the top-left corner of the overlay
                top_left_x = center_x - glasses_width // 2
                top_left_y = center_y - glasses_height // 2

                # Overlay the glasses on the image
                image = overlay_image(image, resized_glasses, top_left_x, top_left_y)


    # Display the image
    cv2.imshow('MediaPipe Face Mesh', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

