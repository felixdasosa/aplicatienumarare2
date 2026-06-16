
import subprocess
import sys

def install(package):
    print(f"Instalez {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    install("opencv-python")
    install("ultralytics")
    install("numpy")
    install("lapx")
    print("\n--- TOTUL A FOST INSTALAT CU SUCCES! ---")
except Exception as e:
    print(f"\nEroare la instalare: {e}")