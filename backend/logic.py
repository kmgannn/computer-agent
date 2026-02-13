from typing import Optional, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Placeholder for fuzzy_match and matches_action_button if they are not defined elsewhere
# You would typically define these helper functions above process_results or import them.
# For the purpose of this edit, we assume they are available in the scope.
def fuzzy_match(token: str, text: str) -> bool:
    """Simple fuzzy match for demonstration."""
    return token in text

def matches_action_button(text: str) -> bool:
    """Determines if text looks like an action button."""
    action_keywords = ["add to cart", "buy now", "checkout", "proceed to checkout", "pay now", "complete purchase"]
    return any(keyword in text.lower() for keyword in action_keywords)


def process_results(results, search: SearchQuery, scale_factor: float) -> Dict:
    """The 'Brain' - Decides what to do based on OCR results."""
    candidates = []
    
    # 1. Filter and Prepare Candidates
    for (bbox, text, prob) in results:
        text_clean = text.lower().strip()
        if prob > 0.4 and len(text_clean) > 2:
            (tl, tr, br, bl) = bbox
            x, y = int(tl[0]), int(tl[1])
            w, h = int(br[0] - tl[0]), int(br[1] - tl[1])
            candidates.append({
                "x": x, "y": y, "w": w, "h": h,
                "text": text_clean,
                "center_x": x + (w / 2),
                "center_y": y + (h / 2),
                "conf": prob
            })

    always_critical_btns = ["place order", "place your order"]
    conditional_critical_btns = ["buy now", "pay now", "complete purchase"]
    
    best_match = None
    action_type = "NONE"
    next_phase = search.phase or "SEARCH"
    
    query_tokens = search.query.lower().split()

    # --- HELPER: SPATIAL MATCHING ---
    def get_spatial_match(candidates, query_tokens):
        """Finds a button that is physically CLOSE to a high-scoring text match."""
        # A. Find Product Title Candidates
        product_candidates = []
        for cand in candidates:
            score = 0
            if matches_action_button(cand["text"]): continue
            
            # Base text match
            for token in query_tokens:
                if len(token) > 2 and fuzzy_match(token, cand["text"]): 
                    score += 10
            
            # MODEL NUMBER BONUS (e.g. "3060", "12gb")
            # This fixes the "MSI 8gb vs ASUS 12gb" issue
            model_tokens = [t for t in query_tokens if any(c.isdigit() for c in t)]
            for model in model_tokens:
                if fuzzy_match(model, cand["text"]): 
                    score += 20 # Huge bonus for specs
            
            if score > 0: 
                product_candidates.append((score, cand))
        
        product_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # B. Find Associated Button
        for score, prod_cand in product_candidates[:5]: # Check top 5
            p_cx = prod_cand["center_x"]
            p_y_bottom = prod_cand["y"] + prod_cand["h"]
            
            best_btn = None
            min_dist = 9999
            
            for btn in candidates:
                if matches_action_button(btn["text"]):
                    # Look for buttons BELOW the text, and roughly aligned
                    dy = btn["y"] - p_y_bottom
                    dx = abs(btn["center_x"] - p_cx)
                    
                    if 0 < dy < 600 and dx < 500: # Tune these pixels if needed
                         if dy < min_dist:
                             min_dist = dy
                             best_btn = btn
            
            if best_btn:
                print(f"   🎯 SPATIAL MATCH: Found button for '{prod_cand['text'][:20]}...' (Score: {score})")
                return best_btn
        return None

    # --- AUTO PILOT LOGIC ---
    if search.mode == "auto_pilot":
        current_phase = search.phase or "SEARCH"
        print(f"🎮 AUTO-PILOT Phase: {current_phase}")

        # 1. GLOBAL CRITICAL CHECK (Safety First)
        for cand in candidates:
            if any(k in cand["text"] for k in always_critical_btns):
                best_match = cand
                action_type = "CRITICAL"
                next_phase = "DONE"
                break
        
        if not best_match:
            # 2. PHASE HANDLERS
            if current_phase == "INIT":
                # Detect search bar
                search_keywords = ["search amazon", "search", "what are you looking for"]
                for cand in candidates:
                    if any(k in cand["text"] for k in search_keywords):
                        best_match = cand
                        action_type = "TYPE_SEARCH" # Frontend will type query
                        next_phase = "SEARCH"
                        break
                        
            elif current_phase == "SEARCH":
                # Use SPATIAL logic to find the right product button
                best_match = get_spatial_match(candidates, query_tokens)
                if best_match:
                    action_type = "NAVIGATE" # Clicking a product on search page is nav
                    next_phase = "PRODUCT_PAGE"
                else:
                    # Fallback: Just click the text if no button found (risky but needed sometimes)
                    pass

            elif current_phase == "PRODUCT_PAGE":
                # Look for "Add to Cart"
                for cand in candidates:
                    if matches_action_button(cand["text"]):
                        best_match = cand
                        action_type = "ACTION"
                        next_phase = "ADDED_TO_CART"
                        break
            
            elif current_phase == "ADDED_TO_CART" or current_phase == "IN_CART":
                 # Look for Checkout
                 checkout_btns = ["proceed to checkout", "go to cart", "checkout"]
                 for cand in candidates:
                     if any(k in cand["text"] for k in checkout_btns):
                         best_match = cand
                         action_type = "NAVIGATE"
                         next_phase = "CHECKOUT"
                         break

    # --- MANUAL MODE LOGIC ---
    else: 
        # 1. Try Spatial Match First (Smartest)
        best_match = get_spatial_match(candidates, query_tokens)
        if best_match:
            action_type = "ACTION"
        
        # 2. Fallback to direct text matching
        if not best_match:
            best_score = 0
            for cand in candidates:
                score = 0
                for token in query_tokens:
                    if len(token) > 2 and fuzzy_match(token, cand["text"]):
                        score += 10
                
                # Model Bonus here too
                model_tokens = [t for t in query_tokens if any(c.isdigit() for c in t)]
                for model in model_tokens:
                    if fuzzy_match(model, cand["text"]): score += 20

                if score > best_score:
                    best_score = score
                    best_match = cand
                    action_type = "ACTION" if matches_action_button(cand["text"]) else "NAVIGATE"

    # --- FINAL RETURN ---
    if best_match:
        # Check transition logic for Manual Mode
        if search.mode == "manual":
             if action_type == "ACTION": next_phase = "ADDED_TO_CART"
             elif "cart" in best_match["text"]: next_phase = "IN_CART"
             else: next_phase = "PRODUCT_PAGE"

        return {
            "found": True,
            "x": int(best_match["x"] / scale_factor),
            "y": int(best_match["y"] / scale_factor),
            "width": int(best_match["w"] / scale_factor),
            "height": int(best_match["h"] / scale_factor),
            "confidence": float(best_match["conf"]),
            "description": f"Target: {best_match['text']}",
            "action_type": action_type,
            "next_phase": next_phase
        }
    
    return {
        "found": False, 
        "x": 0, "y": 0, "width": 0, "height": 0, 
        "confidence": 0.0, 
        "description": "Not found", 
        "action_type": "NONE", 
        "next_phase": next_phase
    }
