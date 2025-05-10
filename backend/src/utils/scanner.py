import cv2
import mediapipe as mp
import numpy as np

# Inicjalizacja MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Inicjalizacja kamery (możesz też wczytać plik wideo)
cap = cv2.VideoCapture("./nba_videos/cut_shots/Mój Film-4-podkreślenie.mp4")  # 0 dla kamery domyślnej, lub ścieżka do pliku wideo

# Lista do przechowywania historii pozycji punktów
position_history = []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Konwersja BGR na RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Przetwarzanie obrazu i detekcja pozy
    results = pose.process(image)
    
    # Konwersja z powrotem na BGR do wyświetlania
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Rysowanie szkieletu na obrazie
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        # Zbieranie aktualnych pozycji punktów
        current_positions = []
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            current_positions.append((cx, cy))
            
            # Oznakowanie punktów numerami
            cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        position_history.append(current_positions)
        
        # Ograniczenie historii do ostatnich N klatek (dla płynności)
        if len(position_history) > 30:
            position_history.pop(0)
        
        # Rysowanie ścieżki ruchu dla wybranych punktów
        for point_idx in [0, 15, 16, 23, 24]:  # Nos, nadgarstki, biodra
            for i in range(1, len(position_history)):
                if len(position_history[i]) > point_idx and len(position_history[i-1]) > point_idx:
                    cv2.line(image, 
                            position_history[i-1][point_idx], 
                            position_history[i][point_idx], 
                            (255, 255, 0), 2)

    # Wyświetlanie wyników
    cv2.imshow('Śledzenie szkieletu ciała', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC aby wyjść
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()