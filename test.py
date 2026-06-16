import cv2
import numpy as np
import time

# --- CONFIGURARE ---
WIDTH, HEIGHT = 640, 480
LINIE_X = 320         # Linia verticală la mijloc
VITEZA = 5            # Pixeli pe frame
RAZA_CERC = 30

def main():
    # Inițializare variabile
    cx = 0                # Poziția X a cercului (începe din stânga)
    cy = HEIGHT // 2      # Poziția Y (mijloc)
    
    counter_out = 0       # Numără trecerile spre dreapta
    last_cx = 0           # Pentru a detecta momentul trecerii

    print("Pornire simulare continuă... Apasă 'q' sau 'ESC' pentru a opri.")

    while True:
        # 1. Creăm un cadru negru (resetăm imaginea la fiecare frame)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # 2. Mișcăm cercul
        last_cx = cx      # Ținem minte unde era înainte de mișcare
        cx += VITEZA      # Mutăm cercul spre dreapta

        # --- RESETARE: Dacă cercul a ieșit din ecran, îl punem înapoi la start ---
        if cx - RAZA_CERC > WIDTH:
            cx = -RAZA_CERC  # Îl punem puțin în afara ecranului în stânga
            last_cx = cx     # Resetăm și istoricul ca să nu numere fals la resetare

        # 3. Logica de Numărare
        # Dacă înainte era în stânga liniei ȘI acum este în dreapta liniei
        if last_cx < LINIE_X and cx >= LINIE_X:
            counter_out += 1
            print(f"Detectat trecere! Total OUT: {counter_out}")
            # Efect vizual: Linia se face VERDE când trece cercul
            cv2.line(frame, (LINIE_X, 0), (LINIE_X, HEIGHT), (0, 255, 0), 4)
        else:
            # Altfel, linia este ROȘIE
            cv2.line(frame, (LINIE_X, 0), (LINIE_X, HEIGHT), (0, 0, 255), 2)

        # 4. Desenăm Cercul
        cv2.circle(frame, (int(cx), cy), RAZA_CERC, (255, 255, 255), -1)

        # 5. Afișăm Textul
        cv2.putText(frame, f"PERSOANE OUT: {counter_out}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "Simulare continua (L-R)", (20, HEIGHT - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 6. Afișăm fereastra
        cv2.imshow("Test Motion Detection", frame)

        # Control viteză simulare (30ms pauză = aprox 30 FPS)
        key = cv2.waitKey(30)
        if key == 27 or key == ord('q'): # ESC sau q pentru ieșire
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()