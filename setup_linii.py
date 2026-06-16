import cv2
import json
import os
import math

# --- CONFIGURAȚIE ---
RTSP_BASE = "rtsp://admin:Asfa_2024@82.76.164.107:10554/Streaming/Channels/"
CONFIG_FILE = "config_16_camere.json"

# ȘTERGEM VECHEA CONFIGURAȚIE PENTRU RESETARE
if os.path.exists(CONFIG_FILE):
    os.remove(CONFIG_FILE)
    print("[RESET] Configurația veche a fost ștearsă. Începem de la zero.")

CHANNELS = list(range(1, 17))
camera_configs = {}

# Variabile editare
temp_pt1 = [200, 360] # Default pe mijloc (bazat pe 1280x720)
temp_pt2 = [1080, 360]
drag_point = -1

def mouse_callback(event, x, y, flags, param):
    global drag_point, temp_pt1, temp_pt2
    if event == cv2.EVENT_LBUTTONDOWN:
        if math.dist([x, y], temp_pt1) < 40: drag_point = 0
        elif math.dist([x, y], temp_pt2) < 40: drag_point = 1
    elif event == cv2.EVENT_MOUSEMOVE and drag_point != -1:
        if drag_point == 0: temp_pt1 = [x, y]
        else: temp_pt2 = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        drag_point = -1

cv2.namedWindow("CONFIGURARE RESET")
cv2.setMouseCallback("CONFIGURARE RESET", mouse_callback)

print("=== CONFIGURARE 16 CANALE (RESETAT) ===")
print("S = Salvează | N = Sari camera | Q = Ieșire")

for chan in CHANNELS:
    url = f"{RTSP_BASE}{chan}01"
    temp_pt1, temp_pt2 = [200, 360], [1080, 360] # Reset la fiecare cameră

    print(f"\n[CONNECT] Conectare Camera {chan}...")
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"[!] Camera {chan} indisponibilă. Apasă 'N'.")
    
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            # Placeholder negru
            import numpy as np
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"CAM {chan}: NO SIGNAL", (300, 360), 0, 1.5, (0,0,255), 2)
        else:
            frame = cv2.resize(frame, (1280, 720))

        # Desenare
        p1 = tuple(map(int, temp_pt1))
        p2 = tuple(map(int, temp_pt2))
        
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        cv2.circle(frame, p1, 10, (0, 0, 255), -1)
        cv2.circle(frame, p2, 10, (0, 0, 255), -1)
        
        cv2.putText(frame, f"SETARI CAMERA {chan}/16", (30, 50), 0, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, "'S'=Save | 'N'=Skip", (30, 100), 0, 0.8, (255, 255, 255), 2)

        cv2.imshow("CONFIGURARE RESET", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            camera_configs[str(chan)] = {"pt1": temp_pt1, "pt2": temp_pt2}
            print(f"[OK] Camera {chan} salvată.")
            running = False
        elif key == ord('n'):
            print(f"[SKIP] Camera {chan} ignorată.")
            running = False
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

with open(CONFIG_FILE, 'w') as f:
    json.dump(camera_configs, f)
    
cv2.destroyAllWindows()
print("\n[GATA] Totul este resetat și salvat. Pornește serverul web.")