import cv2
from ultralytics import YOLO
import numpy as np
import os
import math
import json
import time

# Setări stabilitate
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Fișierul unde ținem minte poziția
CONFIG_FILE = "pozitie_linie_retinuta.json"

# 1. Încărcăm modelul
model = YOLO('yolo11n.pt') 

# 2. URL RTSP (Port 10554 pentru exterior)
RTSP_URL = "rtsp://admin:Asfa_2024@82.76.164.107:10554/Streaming/Channels/1201"

# --- FUNCȚII PENTRU MEMORIE (SAVE/LOAD) ---
def load_config():
    # Dacă există fișierul, luăm coordonatele din el
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                print(f"[MEMORIE] Am încărcat linia de la: {data}")
                return data
        except:
            print("[MEMORIE] Fișier corupt, folosim default.")
    # Dacă nu există, folosim poziția standard
    return {"pt1": [200, 450], "pt2": [1080, 450]}

def save_config(p1, p2):
    # Scriem coordonatele pe disc
    data = {"pt1": p1, "pt2": p2}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f)
    print("[MEMORIE] Poziția liniei a fost salvată!")

# Inițializare variabile
line_pts = load_config()
locked = True # Pornim cu linia blocată (gata de treabă)
drag_point = -1
count_in = 0
count_out = 0
track_history = {} 
counted_ids = []

def mouse_callback(event, x, y, flags, param):
    global drag_point, line_pts
    if locked: return # Nu lăsăm să se miște dacă e blocată
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificăm dacă click-ul e aproape de capete
        if math.dist([x, y], line_pts["pt1"]) < 40: drag_point = 0
        elif math.dist([x, y], line_pts["pt2"]) < 40: drag_point = 1
    
    elif event == cv2.EVENT_MOUSEMOVE and drag_point != -1:
        # Mutăm capătul selectat
        if drag_point == 0: line_pts["pt1"] = [x, y]
        else: line_pts["pt2"] = [x, y]
        
    elif event == cv2.EVENT_LBUTTONUP:
        drag_point = -1

# Inițializare fereastră
cv2.namedWindow("Sistem Numarare Scara")
cv2.setMouseCallback("Sistem Numarare Scara", mouse_callback)

print(f"[CONECTARE] Încerc conectarea la: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

while True:
    ret, frame = cap.read()
    
    # Reconectare automată dacă pică netul
    if not ret:
        print("[!] Semnal pierdut. Reîncerc în 3 secunde...")
        cap.release()
        time.sleep(3)
        cap = cv2.VideoCapture(RTSP_URL)
        continue

    # Resize HD
    frame = cv2.resize(frame, (1280, 720))

    # --- DETECȚIE ---
    results = model.track(frame, persist=True, verbose=False, imgsz=640, conf=0.35, classes=[0], tracker="bytetrack.yaml")
    
    # Coordonate linie curente
    p1 = (int(line_pts["pt1"][0]), int(line_pts["pt1"][1]))
    p2 = (int(line_pts["pt2"][0]), int(line_pts["pt2"][1]))
    line_y = (p1[1] + p2[1]) // 2
    offset = 15 # Zonă tampon
    
    # Desenăm linia
    color_line = (0, 0, 255) if locked else (0, 255, 255) # Roșu=Blocat, Galben=Editare
    cv2.line(frame, p1, p2, color_line, 3)
    
    # Dacă suntem în mod editare, arătăm zona de siguranță cu gri
    if not locked:
        cv2.line(frame, (p1[0], p1[1]-offset), (p2[0], p2[1]-offset), (100,100,100), 1)
        cv2.line(frame, (p1[0], p1[1]+offset), (p2[0], p2[1]+offset), (100,100,100), 1)
        # Cercuri la capete ca să știi de unde tragi
        cv2.circle(frame, p1, 10, (0, 255, 255), -1)
        cv2.circle(frame, p2, 10, (0, 255, 255), -1)

    # Procesare Oameni
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, y2

                # Desenare
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                # Numărare (Doar dacă e între capetele liniei pe orizontală)
                if min(p1[0], p2[0]) < cx < max(p1[0], p2[0]):
                    if track_id not in track_history: track_history[track_id] = []
                    track_history[track_id].append(cy)
                    if len(track_history[track_id]) > 30: track_history[track_id].pop(0)

                    if track_id not in counted_ids:
                        history = track_history[track_id]
                        if len(history) >= 2:
                            prev_y, curr_y = history[0], history[-1]
                            
                            # IN (Sus -> Jos)
                            if prev_y < (line_y - offset) and curr_y > (line_y + offset):
                                count_in += 1
                                counted_ids.append(track_id)
                                cv2.line(frame, p1, p2, (0, 255, 0), 4) # Flash Verde

                            # OUT (Jos -> Sus)
                            elif prev_y > (line_y + offset) and curr_y < (line_y - offset):
                                count_out += 1
                                counted_ids.append(track_id)
                                cv2.line(frame, p1, p2, (0, 0, 255), 4) # Flash Roșu

    # Afișare Scor și Info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, f"IN:  {count_in}", (20, 45), 0, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 85), 0, 0.9, (0, 0, 255), 2)
    
    stare = "BLOCAT (Se salveaza)" if locked else "EDITARE (Muta punctele)"
    cv2.putText(frame, stare, (20, 115), 0, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sistem Numarare Scara", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Ieșire și Salvare Automată
    if key == ord('q'): 
        save_config(line_pts["pt1"], line_pts["pt2"]) # SALVARE LA IEȘIRE
        break
        
    # Blocare/Deblocare și Salvare Manuală
    if key == ord('l'):
        locked = not locked
        if locked:
            save_config(line_pts["pt1"], line_pts["pt2"]) # SALVARE LA BLOCARE
            
    # Resetare Scor
    if key == ord('r'):
        count_in = 0
        count_out = 0
        counted_ids = []
        print("[INFO] Scor resetat.")

cap.release()
cv2.destroyAllWindows()