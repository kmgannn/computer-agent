import platform
import time
import random
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pyautogui
import mss
import cv2
import numpy as np
import webbrowser

# --- CONFIGURATION ---
import warnings
# Filter specific pytorch/dataloader warning that spams console
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data.dataloader')

# SAFETY: Slam mouse to any corner to kill the bot instantly.
pyautogui.FAILSAFE = True 
# Speed: How long it takes to move the mouse (seconds)
MOUSE_DURATION = 0.5
# Set this to a float (e.g. 1.25, 1.5) to force a specific scale. Set to None to auto-detect.
MANUAL_SCALE_OVERRIDE = None 

# Detect Windows DPI Scaling (Robust Method)
import mss
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

app = FastAPI(title="Ghost Layer Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---

class SearchQuery(BaseModel):
    query: str # e.g. "Buy Logitech Mouse"
    mode: Optional[str] = "manual" # "manual" or "auto_pilot"
    phase: Optional[str] = "SEARCH"  # SEARCH, PRODUCT_PAGE, ADDED_TO_CART, IN_CART, CHECKOUT

class TargetCoordinates(BaseModel):
    found: bool
    x: int
    y: int
    width: int
    height: int
    confidence: float
    description: str
    action_type: Optional[str] = "NONE" # CRITICAL, NAVIGATE, ACTION
    next_phase: Optional[str] = "SEARCH"  # What phase to transition to after this action

class ActionRequest(BaseModel):
    action: str # "CLICK", "TYPE", "SCROLL"
    x: Optional[int] = 0
    y: Optional[int] = 0
    text: Optional[str] = ""

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "ghost_layer_active", "screen_size": pyautogui.size()}

# Init OCR (Load model once at startup)
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

@app.post("/scan_target", response_model=TargetCoordinates)
def scan_screen_for_target(search: SearchQuery):
    """
    THE EYES (REAL VISION + OCR):
    Modes:
    - "manual": Looks for keywords from the query.
    - "auto_pilot": Looks for specific commerce funnel buttons in priority order.
    """    
    print(f"📖 OCR SCANNING FOR: {search.query}...")
    
    try:
        # 1. Wait for UI to settle / Page to Load (Safety Buffer)
        time.sleep(1.0)

        # 2. Capture Screen
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            # Convert to BGR (OpenCV standard)
            img_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            debug_img = img_bgr.copy()
        
        # QUALITY: Reverting to explicit 1.0x (Original) because 0.5x killed accuracy for small text
        # "k380" became "kssd" -> We need full pixels for model numbers.
        scale_ratio = 1.0 
        
        # No resize needed, just use original
        img_small = img_bgr 

        print(f"🔎 Running Full Screen OCR (Size: {img_small.shape[1]}x{img_small.shape[0]})...")
        
        # RUN 1: Standard OCR on original image
        results = reader.readtext(img_small) 
        print(f"   [Pass 1] Original image: {len(results)} text blocks")
        
        # OPTIMIZATION: Check if we found our targets in Pass 1. If so, skip Pass 2 (Speed Boost).
        # We look for ANY critical or action keywords.
        found_action_button_pass1 = False
        quick_check_keywords = ["add to", "buy", "checkout", "place order", "cart", "basket"]
        
        for (_, text, _) in results:
            text_lower = text.lower()
            if any(k in text_lower for k in quick_check_keywords):
                found_action_button_pass1 = True
                break
        
        if found_action_button_pass1:
            print("⚡ Speed Opt: Found action buttons in Pass 1. Skipping Enhanced Pass.")
        else:
            # RUN 2: Enhanced OCR for buttons (only if needed)
            # Convert to grayscale and invert to make white text black (more readable)
            gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Run OCR on enhanced image
            results_enhanced = reader.readtext(enhanced)
            print(f"   [Pass 2] Enhanced image: {len(results_enhanced)} text blocks")
            
            # Merge results (dedupe by similar position)
            for (bbox, text, prob) in results_enhanced:
                # Check if this text already exists (avoid duplicates)
                text_lower = text.lower().strip()
                is_duplicate = False
                for (existing_bbox, existing_text, _) in results:
                    if existing_text.lower().strip() == text_lower:
                        is_duplicate = True
                        break
                if not is_duplicate and len(text_lower) > 2:
                    results.append((bbox, text, prob))
        
        print(f"✅ OCR Finished. Found {len(results)} text blocks.")
        
        # --- SHARED HELPERS ---
        ocr_substitutions = {
            '0': 'o', 'o': '0',
            '1': 'l', 'l': '1', 'i': '1',
            '5': 's', 's': '5',
            '8': 'b', 'b': '8',
            '6': 'g', 'g': '6',
            '2': 'z', 'z': '2'
        }
        
        def fuzzy_match(query_token, text):
            """Check if token matches text, accounting for OCR errors."""
            if query_token in text:
                return True
            for orig, sub in ocr_substitutions.items():
                fuzzy_token = query_token.replace(orig, sub)
                if fuzzy_token in text:
                    return True
            return False
            
        candidates = []
        
        # 3. Filter Results
        for (bbox, text, prob) in results:
            text_clean = text.lower().strip()

            if prob > 0.4 and len(text_clean) > 2:
                # bbox is on 1.0x image.
                (tl, tr, br, bl) = bbox
                x = int(tl[0])
                y = int(tl[1])
                w = int(br[0] - tl[0])
                h = int(br[1] - tl[1])
                
                # DEBUG: Show all detected text (look for button text!)
                if "cart" in text_clean or "buy" in text_clean or "add" in text_clean or "checkout" in text_clean:
                    print(f"   🎯 BUTTON CANDIDATE: '{text_clean}' (conf: {prob:.2f})")
                else:
                    print(f"   PLEASE SEE: '{text_clean}'")
                     
                candidates.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "text": text_clean,
                    "area": w * h,
                    "conf": prob
                })

        print(f"🔎 Total Candidates: {len(candidates)}")
        
        # --- PHASE 1: CONTEXT AWARENESS ---
        # Determine if we are on a "Checkout" or "High Risk" page.
        # This protects against "Buy Now" buttons on product pages (Safe) vs "Place Order" (Unsafe).
        
        # STRICTER INDICATORS: Simple "checkout" is too common (e.g. "Tax at checkout").
        risk_indicators = [
            "payment method", 
            "order summary", 
            "review your order", 
            "shipping address",
            "place your order",
            "gp/buy",      # Amazon Checkout URL path
            "spc",         # Single Page Checkout
            "secure checkout" # Usually a header
        ]
        
        is_critical_page = False
        context_triggers = []

        for cand in candidates:
            text = cand["text"]
            for indicator in risk_indicators:
                if indicator in text:
                    is_critical_page = True
                    context_triggers.append(text)
        
        # Special check: If we see "checkout" ALONE, it might be a header or URL, but be careful.
        # Let's rely on the URL path "gp/buy" which OCR catches well.
        
        if is_critical_page:
            print(f"⚠️  CRITICAL PAGE CONTEXT DETECTED: {list(set(context_triggers))[:3]}")
        else:
            print("✅ SAFE PAGE CONTEXT (Browsing/Product Mode)")

        # --- DECISION LOGIC ---
        best_match = None
        action_type = "NONE"

        # "Place Order" is ALWAYS critical.
        
        always_critical_btns = ["place order", "place your order"]
        # REMOVED solo "buy" - it matches "Buy Again" menu item!
        conditional_critical_btns = ["buy now", "pay now", "complete purchase"]

        # Navigation buttons - must be more specific to avoid menu matches
        navigate_btns = ["proceed to checkout", "go to cart", "view cart"]
        # If not critical, "Buy Now" is a navigation action (e.g. Product Page -> Checkout)
        
        # ACTION BUTTONS: Use keyword pairs for robust matching (handles OCR splitting text)
        # We'll check if BOTH keywords appear anywhere in the text
        action_keywords = [
            ("add", "cart"),      # "Add to Cart"
            ("add", "basket"),    # "Add to Basket" 
        ]

        def matches_action_button(text):
            """Check if text contains action button keywords (more robust than exact phrase)."""
            for kw1, kw2 in action_keywords:
                if kw1 in text and kw2 in text:
                    return True
            return False

        # Track the next phase for state machine
        next_phase = search.phase if search.phase else "SEARCH"

        if search.mode == "auto_pilot":
            # ==========================================
            # DETERMINISTIC PHASE-BASED STATE MACHINE
            # ==========================================
            current_phase = search.phase or "SEARCH"
            query_tokens = search.query.lower().split() if search.query else []
            
            print(f"🎮 AUTO-PILOT Phase: {current_phase}")
            print(f"🔎 Query: '{search.query}'")
            
            # 1. CRITICAL CHECK - Always first regardless of phase
            # Safety: Use fuzzy matching for critical keywords to avoid OCR misses
            for cand in candidates:
                text = cand["text"]
                # Check for critical keywords with fuzzy matching
                is_critical = False
                for crit in always_critical_btns:
                    if fuzzy_match(crit, text) or crit in text:
                        is_critical = True
                        break
                
                # In CHECKOUT phase, "Buy Now" is also Critical
                if not is_critical and current_phase == "CHECKOUT":
                    for crit in conditional_critical_btns:
                        if fuzzy_match(crit, text) or crit in text:
                            is_critical = True
                            print(f"⚠️ High Risk Context (CHECKOUT): Treating '{text}' as CRITICAL.")
                            break

                if is_critical:
                    best_match = cand
                    action_type = "CRITICAL"
                    next_phase = "DONE"
                    print(f"🚨 CRITICAL MATCH: {text}")
                    break
            
            # 2. PHASE-BASED ACTIONS
            if not best_match:
                if current_phase == "INIT":
                    # Check if we are on Amazon. Look for "Amazon" or "Search"
                    is_amazon = any("amazon" in c["text"].lower() for c in candidates)
                    if not is_amazon:
                        print("🌐 Not on Amazon. Triggering Navigation...")
                        return {
                            "found": True, "x": 0, "y": 0, "width": 0, "height": 0, "confidence": 1.0,
                            "description": "Navigation to Amazon",
                            "action_type": "NAVIGATE_URL",
                            "next_phase": "INIT"
                        }
                    else:
                        print("📍 Phase INIT: Looking for search bar...")
                        search_keywords = ["search amazon", "search", "what are you looking for"]
                        for cand in candidates:
                            text = cand["text"].lower()
                            if any(k in text for k in search_keywords):
                                best_match = cand
                                action_type = "TYPE_SEARCH"
                                next_phase = "SEARCH"
                                print(f"🔍 Found Search Bar: {cand['text']}")
                                break

                elif current_phase == "SEARCH":
                    # SEARCH PHASE: Look for product link matching query
                    print("📍 Phase SEARCH: Looking for product link...")
                    best_score = 0
                    for cand in candidates:
                        cand_text = cand["text"]
                        score = 0
                        for token in query_tokens:
                            if len(token) > 2 and fuzzy_match(token, cand_text):
                                score += 10
                        # Skip buttons - we want product titles
                        if matches_action_button(cand_text) or "more buying" in cand_text:
                            continue
                        if score > best_score and len(cand_text) > 20:
                            best_score = score
                            best_match = cand
                            action_type = "NAVIGATE"
                            next_phase = "PRODUCT_PAGE"
                    
                    if best_match:
                        print(f"🎯 Click Product: '{best_match['text'][:40]}...'")
                
                elif current_phase == "PRODUCT_PAGE":
                    # PRODUCT PAGE: Click "Add to Cart"
                    print("📍 Phase PRODUCT_PAGE: Looking for Add to Cart...")
                    for cand in candidates:
                        if matches_action_button(cand["text"]):
                            best_match = cand
                            action_type = "ACTION"
                            next_phase = "ADDED_TO_CART"
                            print(f"🛒 Click Add to Cart: {cand['text']}")
                            break
                
                elif current_phase == "ADDED_TO_CART":
                    # ADDED TO CART: Click "Proceed to Checkout" or "Go to Cart"
                    print("📍 Phase ADDED_TO_CART: Looking for Checkout/Cart...")
                    # Removing "cart" and "see cart" to avoid clicking header icons
                    cart_btns = ["proceed to checkout", "go to cart", "view cart", "checkout"]
                    for cand in candidates:
                        cand_text = cand["text"]
                        if len(cand_text) < 30 and any(k in cand_text for k in cart_btns):
                            best_match = cand
                            action_type = "NAVIGATE"
                            # If we found checkout directly, skip IN_CART
                            if "checkout" in cand_text:
                                next_phase = "CHECKOUT"
                            else:
                                next_phase = "IN_CART"
                            print(f"🛒 Click Navigation: {cand['text']}")
                            break
                
                elif current_phase == "IN_CART":
                    # IN CART: Click "Proceed to Checkout"
                    print("📍 Phase IN_CART: Looking for Proceed to Checkout...")
                    checkout_btns = ["proceed to checkout", "checkout", "proceed"]
                    for cand in candidates:
                        cand_text = cand["text"]
                        if len(cand_text) < 30 and any(k in cand_text for k in checkout_btns):
                            best_match = cand
                            action_type = "NAVIGATE"
                            next_phase = "CHECKOUT"
                            print(f"✅ Click Checkout: {cand['text']}")
                            break
                
                elif current_phase == "CHECKOUT":
                    # CHECKOUT: This should trigger CRITICAL for Place Order
                    print("📍 Phase CHECKOUT: Looking for Place Order (CRITICAL)...")
                    for cand in candidates:
                        text = cand["text"]
                        if any(k in text for k in always_critical_btns + conditional_critical_btns):
                            best_match = cand
                            action_type = "CRITICAL"
                            next_phase = "DONE"
                            print(f"🚨 CRITICAL: {text}")
                            break
        
        else:
            # MANUAL MODE 1. product + action > 2. query = name/button, 3. best name match
            
            query_tokens = search.query.lower().split()
            
            # Action keywords suggest user wants to DO something
            action_intent_keywords = ["buy", "cart", "add", "shop", "checkout", "purchase"]
            has_action_intent = any(kw in search.query.lower() for kw in action_intent_keywords)
            
            # STEP 1: Check if action buttons exist on this page
            # Helper function for clarity, using the existing matches_action_button logic
            def is_add_to_cart_button(text):
                return matches_action_button(text)

            action_button_candidates = [c for c in candidates if is_add_to_cart_button(c["text"])]
            action_button_found = action_button_candidates[0] if action_button_candidates else None
            cart_button_count = len(action_button_candidates)
            
            # STEP 2: Check if product name is visible
            product_tokens_found = 0
            for token in query_tokens:
                if len(token) > 2:
                    for cand in candidates:
                        if token in cand["text"]:
                            product_tokens_found += 1
                            break
            
            is_on_product_page = (product_tokens_found >= 2) and (cart_button_count <= 1) # Amazon
            #is_on_product_page = (product_tokens_found >= 2) and (cart_button_count == 1) Shopee/Lazada
            
            # REMOVED: Do not check search.phase == "SEARCH" here because Manual Mode 
            # typically sends "manual" mode with default "SEARCH" phase, even if user is on product page.
            # We trust the heuristic above (button count).
            
            print(f"📍 Context Detection: {product_tokens_found} tokens, {cart_button_count} cart buttons => is_product_page={is_on_product_page}")
            
            # DECISION LOGIC:
            # FIX v2: Prioritize SPATIAL SCORING (Case B) above "Product Page" assumption (Case A).
            # This ensures that on a search page, we pick the RIGHT item. 
            # On a product page, Spatial might fail (no title-button pair), so we fallback to Case A.
            
            best_match = None
            
            # --- ATTEMPT 1: SPATIAL ASSOCIATION (The "Smart" Way) ---
            # Uses shared fuzzy_match helper defined above


            if has_action_intent:
                # print(f"🕵️ Intent detected. Running Spatial Logic...")
                
                # A. Score potential product titles
                product_candidates = []
                for cand in candidates:
                    cand_text = cand["text"]
                    score = 0
                    if len(cand_text) < 5 or matches_action_button(cand_text): continue
                    for token in query_tokens:
                        if len(token) > 2 and fuzzy_match(token, cand_text): score += 10
                    model_tokens = [t for t in query_tokens if any(c.isdigit() for c in t)]
                    for model in model_tokens:
                        if fuzzy_match(model, cand_text): score += 20
                    if score > 0: product_candidates.append((score, cand))
                
                product_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # B. Find associated button
                target_btn = None
                for score, prod_cand in product_candidates[:5]:
                    p_cx = prod_cand["center_x"]
                    p_y_bottom = prod_cand["y"] + prod_cand["h"]
                    best_btn_dist = 9999
                    best_local_btn = None
                    
                    for btn_cand in candidates:
                        if matches_action_button(btn_cand["text"]):
                            btn_cx = btn_cand["center_x"]
                            btn_y = btn_cand["y"]
                            dy = btn_y - p_y_bottom
                            dx = abs(btn_cx - p_cx)
                            # Constraints: Below (0-600px), Aligned (0-500px)
                            if 0 < dy < 600 and dx < 500:
                                if dy < best_btn_dist:
                                    best_btn_dist = dy
                                    best_local_btn = btn_cand
                    
                    if best_local_btn:
                        target_btn = best_local_btn
                        print(f"   🎯 SPATIAL MATCH: Found button for '{prod_cand['text']}' (Score: {score})")
                        break
                
                if target_btn:
                    best_match = target_btn
                    action_type = "ACTION"
            
            # --- ATTEMPT 2: PRODUCT PAGE FALLBACK ---
            if not best_match and action_button_found and is_on_product_page:
                # If Spatial failed (maybe title is far from button, or layout is unique), 
                # but we are confident this is a product page, just click the action button.
                best_match = action_button_found
                action_type = "ACTION"
                print(f"🎯 PRODUCT PAGE FALLBACK: Clicking main Action Button.")

            # --- ATTEMPT 3: TEXT NAVIGATION ---
            # GUARD: If we are on a Product Page, DO NOT click text.
            # We should only click Action Buttons. If button is missing (OCR failed?), 
            # clicking the Title is worse than doing nothing (it refreshes page).
            if is_on_product_page and not action_button_found:
                 print("⚠️ On Product Page but NO Action Button found. Preventing Text Click fallback.")
            elif not best_match:
                # Fallback: Score by query token matching (Navigation / Reading)
                best_score = 0
                print(f"🔎 Scoring Candidates for query: {query_tokens}")
                
                # Fallback: Score by query token matching (Navigation / Reading)
                best_score = 0
                print(f"🔎 Scoring Candidates for query: {query_tokens}")
                
                # Uses shared fuzzy_match helper defined above

                for cand in candidates:
                    cand_text = cand["text"]
                    score = 0
                    matched_tokens = 0
                    
                    # 1. Product Name Match Score (with fuzzy matching)
                    for token in query_tokens:
                        if len(token) > 2:  # Skip short words
                            if fuzzy_match(token, cand_text):
                                score += 10
                                matched_tokens += 1
                    
                    # BONUS: Extra points if this looks like the specific product model
                    # (has model number like m510, g309, k380)
                    model_tokens = [t for t in query_tokens if any(c.isdigit() for c in t)]
                    for model in model_tokens:
                        if fuzzy_match(model, cand_text):
                            score += 15  # Strong bonus for model number match
                            print(f"   ⭐ Model number '{model}' matched in '{cand_text[:20]}...'")
                    
                    # 2. Functional Match Score (lower weight)
                    for kw in action_intent_keywords:
                        if kw in cand_text:
                            score += 5
                    
                    if score > 0:
                         print(f"   Candidate: '{cand_text[:30]}...' | Score: {score}")

                    if score > best_score:
                        best_score = score
                        best_match = cand
                        
                        # Check safety using SAME context logic
                        is_btn_critical = any(k in cand_text for k in always_critical_btns)
                        if is_critical_page and any(k in cand_text for k in conditional_critical_btns):
                            is_btn_critical = True
                            
                        if is_btn_critical:
                            action_type = "CRITICAL"
                        else:
                            # Heuristic: Is this text an Action Button or Content?
                            if matches_action_button(cand_text):
                                action_type = "ACTION"
                            else:
                                action_type = "NAVIGATE"

                if best_match:
                     print(f"🎯 BEST MATCH: '{best_match['text']}' (Score: {best_score}, Type: {action_type})")

        if best_match:
            x, y, w, h = best_match["x"], best_match["y"], best_match["w"], best_match["h"]
            
            # Draw Logic (Red)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.putText(debug_img, f"TARGET: {best_match['text']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Apply DPI Scaling
            x_scaled = int(x / SCALE_FACTOR)
            y_scaled = int(y / SCALE_FACTOR)
            w_scaled = int(w / SCALE_FACTOR)
            h_scaled = int(h / SCALE_FACTOR)
            
            # CRITICAL FIX: Update next_phase for Manual Mode handoff
            if search.mode == "manual":
                if action_type == "ACTION":
                    next_phase = "ADDED_TO_CART"
                elif action_type == "NAVIGATE":
                    # If we clicked a product, we go to product page.
                    # If we clicked "Go to Cart" (technically NAVIGATE), we go to IN_CART?
                    # For search results, NAVIGATE = Click Product.
                    if "cart" in best_match["text"] or "basket" in best_match["text"]:
                        next_phase = "IN_CART"
                    else:
                        next_phase = "PRODUCT_PAGE"
            
            print(f"✅ RETURN TARGET: {x_scaled},{y_scaled} ({w_scaled}x{h_scaled}) [Raw: {x},{y}] Type: {action_type} NextPhase: {next_phase}")
            
            # Save with timestamp to keep history
            timestamp = int(time.time() * 1000)
            filename = f"debug_vision_{timestamp}.png"
            cv2.imwrite(filename, debug_img)
            print(f"📸 Saved debug image: {filename}")
            
            return {
                "found": True,
                "x": x_scaled,
                "y": y_scaled,
                "width": w_scaled,
                "height": h_scaled,
                "confidence": 0.95,
                "description": f"Button: {best_match['text']}",
                "action_type": action_type,
                "next_phase": next_phase
            }
        else:
            print(f"❌ NO MATCHING TARGET FOUND (Phase: {next_phase})")
            timestamp = int(time.time() * 1000)
            filename = f"debug_vision_{timestamp}_failed.png"
            cv2.imwrite(filename, debug_img)
            return {
                "found": False,
                "x": 0, "y": 0, "width": 0, "height": 0,
                "confidence": 0.0,
                "description": "Target not found",
                "action_type": "NONE",
                "next_phase": next_phase
            }

    except Exception as e:
        print(f"Error scanning screen: {e}")
        return {
             "found": False,
             "x": 0, "y": 0, "width": 0, "height": 0,
             "confidence": 0.0,
             "description": f"Error: {str(e)}",
             "action_type": "ERROR"
        }

@app.post("/execute_action")
def execute_physical_action(req: ActionRequest):
    """
    THE HANDS:
    Actually moves the mouse or clicks.
    """
    print(f"Executing: {req.action} at ({req.x}, {req.y})")
    
    try:
        if req.action == "MOVE_TO_TARGET":
            # Smoothly move mouse to the target
            pyautogui.moveTo(req.x, req.y, duration=MOUSE_DURATION, tween=pyautogui.easeInOutQuad)
            return {"status": "moved"}
            
        elif req.action == "NAVIGATE":
            print(f"🌐 NAVIGATING TO: {req.text}")
            webbrowser.open(req.text)
            return {"status": "navigated"}

        elif req.action == "CLICK":
            # Click the mouse
            print(f"🖱️  CLICKING NOW at {req.x}, {req.y}")
            pyautogui.click(req.x, req.y)
            return {"status": "clicked"}
            
        elif req.action == "TYPE":
            pyautogui.write(req.text, interval=0.05)
            # If the user wants to submit, they can include \n or we can add a flag.
            # For simplicity, if it's a search, we just press enter.
            if "\n" in req.text or "ENTER" in req.action:
                pyautogui.press('enter')
            return {"status": "typed"}
            
    except pyautogui.FailSafeException:
        raise HTTPException(status_code=400, detail="FAILSAFE TRIGGERED: Mouse moved to corner.")
    
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    print("\n\n" + "="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
