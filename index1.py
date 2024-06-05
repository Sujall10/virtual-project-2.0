import cv2
import cvzone
import mediapipe as mp
from cvzone.FaceDetectionModule import FaceDetector

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    

    while True:
        success, img = cap.read()

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
            img, bboxs = detector.findFaces(img, draw=False)        
            if bboxs:

                glasses = cv2.imread('glasses/glass5.png', cv2.IMREAD_UNCHANGED) 
                glasses = cv2.resize(glasses,(0,0),None,0.5,0.5)
                img = cvzone.overlayPNG(img,glasses,(311,282))

        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()