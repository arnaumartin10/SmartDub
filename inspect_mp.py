import cv2
import mediapipe as mp
import numpy as np

img = cv2.imread("data/inputs/generic_face.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
    res = fm.process(rgb)
    print("Has face landmarks:", bool(res.multi_face_landmarks))
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0]
        lm0 = lms.landmark[0]
        print("LM0 fields:", dir(lm0))
        print("x:", lm0.x, "y:", lm0.y, "z:", lm0.z)
        print("visibility:", getattr(lm0, 'visibility', 'N/A'))
        print("presence:", getattr(lm0, 'presence', 'N/A'))
    
    # Are there other fields?
    print("Result fields:", dir(res))
    
    # Let's check face detection module specifically to see if it gives confidence
    print("\n--- Face Detection Module ---")
    with mp.solutions.face_detection.FaceDetection() as fd:
        res2 = fd.process(rgb)
        print("Has face detections:", bool(res2.detections))
        if res2.detections:
            d = res2.detections[0]
            print("Detection fields:", dir(d))
            print("Score:", d.score)
