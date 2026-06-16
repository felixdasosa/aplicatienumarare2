import socket

IP = "82.76.164.107"
# Lista cu porturile suspecte. 896 e cel de web, restul sunt pentru video.
POSIBILE = [554, 55896, 10554, 896, 1024, 8000]

print(f"Scanez IP-ul {IP} pentru a gasi portul video...")

found = None
for port in POSIBILE:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)
    result = sock.connect_ex((IP, port))
    if result == 0:
        print(f"[DESCHIS] Portul {port} este activ.")
        # De obicei 896 e web, deci il ignoram pentru video daca gasim altele
        if port != 896: 
            found = port
    sock.close()

if found:
    print(f"\n>>> CONCLUZIE: Foloseste portul {found} in aplicatie! <<<")
else:
    print("\n[!] Niciun port video nu pare deschis. Verifica routerul.")