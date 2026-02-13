import mss
import cv2
import time
try:
    with mss.mss() as sct:
        # Get physical monitor size from MSS
        monitor = sct.monitors[1]
        phys_w, phys_h = monitor["width"], monitor["height"]
        
        # Get logical size from PyAutoGUI
        log_w, log_h = pyautogui.size()
        
        if MANUAL_SCALE_OVERRIDE:
            SCALE_FACTOR = MANUAL_SCALE_OVERRIDE
            method = "MANUAL"
        else:
            SCALE_FACTOR = phys_w / log_w
            method = "AUTO"

        print("\n" + "="*40)
        print(f"🖥️  SCREEN CONFIGURATION ({method})")
        print(f"    Physical (MSS): {phys_w}x{phys_h} (Pixels)")
        print(f"    Logical (GUI):  {log_w}x{log_h} (Points)")
        print(f"    Scale Factor:   {SCALE_FACTOR:.2f}x")
        print("="*40 + "\n")

except Exception as e:
    print(f"⚠️ Scale Scaling Failed: {e}. Defaulting to 1.0.")
    SCALE_FACTOR = 1.0

import easyocr
import torch

print("\n" + "="*40)
print("🧠 LOADING BRAIN (OCR MODEL)...")

# GPU CHECK
use_gpu = False
if torch.cuda.is_available():
    print(f"✅ CUDA GPU DETECTED: {torch.cuda.get_device_name(0)}")
    use_gpu = True
else:
    print("⚠️  CUDA GPU NOT FOUND. Fallback to CPU (Slower).")

try:
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    print(f"✅ BRAIN READY. (GPU={use_gpu})")
except Exception as e:
    print(f"❌ OCR LOAD FAILED: {e}")
    # Fallback to CPU if GPU failed despite check
    if use_gpu:
        print("🔄 Retrying with CPU...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ BRAIN READY (CPU Mode).")
    else:
        raise e

print("="*40 + "\n")