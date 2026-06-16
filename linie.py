import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO
from collections import defaultdict
import time

# ================= CONFIGURARE =================

# 1. Datele Camerei
IP = "86.124.84.104"
PORT = "896"
USER = "admin"
PASS = "Ifis_2022"
CHANNEL = "2"
HTTP_URL = f"http://{IP}:{PORT}/cgi-bin/snapshot.cgi?channel={CHANNEL}"

# 2. Modelul AI
MODEL_PATH = 'yolov8n.pt' 

# 3. CONFIGURAREA LINIEI (Punct Start -> Punct Final)
# Pentru o linie la 45 grade, trebuie sa cresti X si Y proportional.
# Format: (Coordonata X, Coordonata Y)
# Ex: (100, 100) -> (600, 600) este o diagonala perfecta

LINE_START = (135, 559)
LINE_END   = (348, 391)
# ===============================================

def get_image_from_http(url, username, password):
    try:
        response = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=3)
        if response.status_code == 200:
            image_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, -1)
            return frame
    except:
        return None
    return None

# Functie matematica: Calculeaza pe ce parte a liniei este un punct
# Rezultat > 0 : O parte
# Rezultat < 0 : Cealalta parte
def ccw(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def is_crossing(line_start, line_end, point_prev, point_curr):
    # Verificam pozitia relativa fata de linie pentru punctul vechi si cel nou
    pos_prev = ccw(line_start, line_end, point_prev)
    pos_curr = ccw(line_start, line_end, point_curr)

    # Daca semnele sunt diferite (unul + si unul -), inseamna ca a trecut linia
    if pos_prev * pos_curr < 0:
        if pos_prev < 0:
            return 1  # Intrare (Directia A)
        else:
            return -1 # Iesire (Directia B)
    return 0 # Nu a trecut

def main():
    print("Incarc modelul...")
    model = YOLO(MODEL_PATH)
    track_history = defaultdict(lambda: [])
    
    count_A = 0 # Intrari (sau directia 1)
    count_B = 0 # Iesiri (sau directia 2)
    counted_ids = set()

    print(f"Pornire pe: {HTTP_URL}")

    while True:
        frame = get_image_from_http(HTTP_URL, USER, PASS)
        if frame is None:
            time.sleep(1)
            continue

        # Detectie si Tracking
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        # Desenam Linia Diagonala
        cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3) # Galben

        # Marcam punctele de start si final ca sa stim directia
        cv2.circle(frame, LINE_START, 5, (0, 255, 0), -1) # Start (Verde)
        cv2.circle(frame, LINE_END, 5, (0, 0, 255), -1)   # Final (Rosu)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (float(x), float(y)) # Centrul persoanei (picioare aprox)
                
                track = track_history[track_id]
                track.append(center)
                if len(track) > 30: track.pop(0)

                if track_id not in counted_ids and len(track) > 2:
                    prev_pt = track[-2]
                    curr_pt = track[-1]

                    # Verificam trecerea
                    crossing = is_crossing(LINE_START, LINE_END, prev_pt, curr_pt)

                    if crossing == 1:
                        count_A += 1
                        counted_ids.add(track_id)
                        cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 5) # Flash Verde
                        print(f"ID {track_id} a trecut in sensul A ->")
                    
                    elif crossing == -1:
                        count_B += 1
                        counted_ids.add(track_id)
                        cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 5) # Flash Rosu
                        print(f"ID {track_id} a trecut in sensul B <-")

                # Desenam
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Afisare text
        cv2.putText(frame, f"Directia A: {count_A}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Directia B: {count_B}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Numarare Diagonala", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()