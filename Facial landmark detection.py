import cv2
import numpy as np
import mediapipe as mp

def draw_landmarks(frame, landmarks, connections, color):
    for connection in connections:
        point1 = landmarks[connection[0]]
        point2 = landmarks[connection[1]]
        cv2.line(frame, point1, point2, color, 1)
    for idx, point in enumerate(landmarks):
        cv2.circle(frame, point, 1, color, -1)
        # Adăugat numere pentru landmark-uri (opțional)
        #cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

def detect_expressions(landmarks):
    # Calcul metrici pentru expresii
    mouth_width = landmarks[291][0] - landmarks[61][0]
    mouth_height = landmarks[0][1] - landmarks[17][1]
    smile_ratio = mouth_width / max(mouth_height, 1)  # Evitare împărțire la zero
    
    left_eye_height = abs(landmarks[159][1] - landmarks[145][1])
    right_eye_height = abs(landmarks[386][1] - landmarks[374][1])
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    
    eyebrow_left = landmarks[105][1] - landmarks[66][1]
    eyebrow_right = landmarks[334][1] - landmarks[296][1]
    
    return {
        'smile': smile_ratio > 80,
        'eyes_closed': avg_eye_height < 5,
        'surprise': (eyebrow_left > 2.1) or (eyebrow_right > 2.1)
    }

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5)
    
    # Definire conexiuni complete pentru trăsături faciale
    eye_connections = [
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), 
        (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
        (362, 382), (382, 381), (381, 380), (380, 374), (374, 373),
        (373, 390), (390, 249), (249, 263), (362, 466), (466, 388)
    ]
    
    mouth_connections = [
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), 
        (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375), (375, 291)
    ]
    
    nose_connections = [
        (1, 2), (2, 98), (98, 97), (97, 3), (3, 326), 
        (326, 327), (327, 420), (420, 4), (4, 275), (275, 294)
    ]
    
    face_oval_connections = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
        (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
        (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
        (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
        (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
        (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
        (109, 10)
    ]
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            height, width, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                # Conversie landmarks pixeli
                landmarks = [(int(pt.x * width), int(pt.y * height)) for pt in face_landmarks.landmark]
                
                # Desenare toate trăsăturile
                draw_landmarks(frame, landmarks, eye_connections, (255, 0, 0))  # Ochi - albastru
                draw_landmarks(frame, landmarks, mouth_connections, (0, 255, 0))  # Gură - verde
                draw_landmarks(frame, landmarks, nose_connections, (0, 0, 255))  # Nas - roșu
                draw_landmarks(frame, landmarks, face_oval_connections, (255, 255, 0))  # Contur față - galben
                
                # Detectare expresii
                expressions = detect_expressions(landmarks)
                
                # Afișare rezultate expresii
                y_offset = 30
                for expr, state in expressions.items():
                    color = (0, 255, 0) if state else (0, 0, 255)
                    cv2.putText(frame, f"{expr}: {'DA' if state else 'NU'}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, color, 2)
                    y_offset += 30
        
        cv2.imshow('Facial Feature Tracking with Expression Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()