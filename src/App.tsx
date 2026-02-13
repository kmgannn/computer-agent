import { useState, useEffect } from 'react';
import {
  CreditCard,
  CheckCircle2,
  Cpu,
  Lock
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';

type Step = 'IDLE' | 'SEARCHING' | 'MOVING' | 'TARGETING' | 'REVIEW' | 'EXECUTING' | 'DONE';

interface HighlightBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export default function App() {

  const [active] = useState(true); // Default to true so it works
  const [query, setQuery] = useState('');
  const [step, setStep] = useState<Step>('IDLE');

  // PIXEL BASED COORDINATES (Initial: Center of screen)
  const [cursor, setCursor] = useState({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
  const [highlightBox, setHighlightBox] = useState<HighlightBox | null>(null);
  const [pendingAction, setPendingAction] = useState<any>(null); // Added missing state

  // AUTO-PILOT STATE
  const [isAutoPilot, setIsAutoPilot] = useState(false);
  const [targetQuery, setTargetQuery] = useState(''); // Preserved from manual search
  const [currentPhase, setCurrentPhase] = useState('INIT'); // State machine phase
  const [isScanning, setIsScanning] = useState(false); // Hide UI during screenshot
  const [log, setLog] = useState<string[]>([]);
  const [recordingMode, setRecordingMode] = useState(false);

  const API_URL = 'http://localhost:8000';

  const toggleRecordingMode = async () => {
    const newMode = !recordingMode;
    setRecordingMode(newMode);
    await invoke('set_always_on_top', { alwaysOnTop: !newMode });
    addLog(newMode ? "🎥 Recording Mode: ON (Always On Top disabled)" : "🎥 Recording Mode: OFF");
  };

  const addLog = (msg: string) => setLog(prev => [msg, ...prev].slice(0, 5));

  useEffect(() => {
    if (active) {
      // Focus input when activated
      document.getElementById('agent-input')?.focus();
    }
  }, [active]);

  // Initial Setup: Set Window to Overlay Mode
  useEffect(() => {
    // Ensure window is transparent and click-through on start
    // invoke('set_click_through', { ignore: false }); // Start interactable?
    // Actually, we want it interactable only on the bar.
    // But for this prototype, we'll toggle click-through when executing.
  }, []);

  // --- AUTO-PILOT LOOP ---
  useEffect(() => {
    let isLooping = true;

    const runAutoPilot = async () => {
      if (!isAutoPilot || step !== 'IDLE') return;

      addLog("🤖 Scanning...");

      setIsScanning(true);
      await invoke('set_window_visible', { visible: false });
      await wait(200); // Allow OS compositor to update

      try {
        // 1. SCAN - Pass the target query and current phase
        const res = await fetch(`${API_URL}/scan_target`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: targetQuery,
            mode: "auto_pilot",
            phase: currentPhase
          })
        });

        await invoke('set_window_visible', { visible: true });
        setIsScanning(false);

        const data = await res.json();

        if (!isLooping || !isAutoPilot) return;

        if (data.found) {
          const targetX = data.x + (data.width / 2);
          const targetY = data.y + (data.height / 2);

          setHighlightBox({
            x: data.x, y: data.y, w: data.width, h: data.height
          });
          setCursor({ x: targetX, y: targetY });

          if (data.action_type === 'CRITICAL') {
            // STOP AND ASK
            addLog(`🚨 Critical Match: ${data.description}`);
            setPendingAction({
              action: 'CLICK',
              x: Math.round(targetX),
              y: Math.round(targetY),
              description: data.description,
              value: "Unknown Amount"
            });
            setStep('REVIEW'); // This pauses the loop because step is not IDLE
            setIsAutoPilot(false); // Disengage auto-pilot for safety
          } else if (data.action_type === 'NAVIGATE_URL') {
            addLog("🌐 Navigating to Amazon...");
            await executeAction({
              action: 'NAVIGATE',
              text: 'https://www.amazon.com'
            });
            await wait(5000); // Give it time to load
          } else if (data.action_type === 'TYPE_SEARCH') {
            addLog(`⌨️ Typing: ${targetQuery}`);
            await executeAction({
              action: 'CLICK',
              x: Math.round(targetX),
              y: Math.round(targetY),
            });
            await executeAction({
              action: 'TYPE',
              text: targetQuery + "\n"
            });
            setCurrentPhase('SEARCH');
            await wait(3000);
          } else if (data.action_type === 'NAVIGATE' || data.action_type === 'ACTION') {
            // AUTO EXECUTE
            addLog(`⚡ [${currentPhase}] ${data.description}`);

            await executeAction({
              action: 'CLICK',
              x: Math.round(targetX),
              y: Math.round(targetY),
              description: data.description
            });

            // UPDATE PHASE from backend response
            if (data.next_phase) {
              setCurrentPhase(data.next_phase);
              addLog(`📍 Phase: ${data.next_phase}`);
            }

            addLog("⏳ Waiting for page load...");
            await wait(4000); // Wait for page load

            // Loop continues automatically via useEffect dependency or re-run
          }
        } else {
          addLog("💤 No target found...");
          setHighlightBox(null);
        }

      } catch (e) {
        console.error(e);
        addLog("⚠️ Scan Error");
      }

      // Re-schedule loop if still active
      if (isLooping && isAutoPilot && step === 'IDLE') {
        setTimeout(runAutoPilot, 2000);
      }
    };

    if (isAutoPilot && step === 'IDLE') {
      runAutoPilot();
    }

    return () => { isLooping = false; };
  }, [isAutoPilot, step]);


  // Helper to execute action (used by both Auto-Pilot and Manual Auto-Confirm)
  const executeAction = async (action: any) => {
    setStep('EXECUTING');
    setHighlightBox(null);

    // Make window click-through so we can click the real button behind us
    await invoke('set_click_through', { ignore: true });

    try {
      await fetch(`${API_URL}/execute_action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action)
      });

      await wait(500);

      // Re-enable interaction
      await invoke('set_click_through', { ignore: false });
      setStep('DONE');

      setTimeout(() => {
        setStep('IDLE');
        setPendingAction(null);
        setCursor({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
      }, 5000);
    } catch (error) {
      console.error("Execution error:", error);
      await invoke('set_click_through', { ignore: false });
      setStep('IDLE');
    }
  };

  // --- COMMAND HANDLER (Unified Mode) ---
  const handleCommand = async (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      setStep('SEARCHING');
      addLog(`🔍 Searching: ${query}`);

      // Hide UI for screenshot logic
      setIsScanning(true);
      await invoke('set_window_visible', { visible: false });
      await wait(400); // Wait for fade out + paint

      try {
        const res = await fetch(`${API_URL}/scan_target`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, mode: "manual" })
        });

        await invoke('set_window_visible', { visible: true });
        setIsScanning(false); // Show UI again

        const data = await res.json();

        if (data.found) {
          const targetX = data.x + (data.width / 2);
          const targetY = data.y + (data.height / 2);

          const actionPayload = {
            action: 'CLICK',
            x: Math.round(targetX),
            y: Math.round(targetY),
            description: data.description
          };

          setHighlightBox({
            x: data.x, y: data.y, w: data.width, h: data.height
          });
          setCursor({ x: targetX, y: targetY });

          if (data.action_type === 'CRITICAL') {
            addLog(`🚨 Critical: ${data.description}`);
            setPendingAction(actionPayload);
            setStep('REVIEW');
          } else {
            // AUTO EXECUTE FOR MANUAL COMMANDS TOO
            addLog(`⚡ Auto-Clicking: ${data.description}`);

            await executeAction(actionPayload);

            // HANDOFF TO AUTO-PILOT with correct starting phase
            // If we clicked a product link -> next phase is PRODUCT_PAGE
            // If we clicked Add to Cart -> next phase is ADDED_TO_CART
            addLog(`🔄 Handing off to Auto-Pilot...`);
            setTargetQuery(query); // Preserve the search query for auto-pilot
            setCurrentPhase('INIT');
            setIsAutoPilot(true);
          }

        } else {
          setStep('IDLE');
          addLog("❌ Not found");
        }
      } catch (e) {
        console.error(e);
        setStep('IDLE');
      }
    }
  };

  const handleApprove = async () => {
    if (!pendingAction) return;
    await executeAction(pendingAction);
  };

  const wait = (ms: number) => new Promise(r => setTimeout(r, ms));

  return (
    <div className={`relative w-full h-screen overflow-hidden font-sans selection:bg-blue-500/30 pointer-events-none transition-colors duration-500 ${recordingMode ? 'bg-black/20' : 'bg-transparent'}`}>

      {/* DEBUG PANEL - Hide when scanning */}
      <div className={`absolute top-2 left-2 bg-black/80 text-green-400 font-mono text-[10px] p-2 rounded z-[100] pointer-events-none transition-opacity duration-200 ${isScanning ? 'opacity-0' : 'opacity-50 hover:opacity-100'}`}>
        <div className="flex justify-between items-center gap-4 mb-2">
          <span>DEBUG</span>
          <button
            onClick={toggleRecordingMode}
            className={`pointer-events-auto px-2 py-0.5 rounded text-[8px] font-bold transition-colors ${recordingMode ? 'bg-red-500 text-white' : 'bg-white/10 text-white/50 hover:bg-white/20'}`}
          >
            {recordingMode ? 'RECORDING ON' : 'RECORDING OFF'}
          </button>
        </div>
        <p>Step: {step}</p>
        <p>Active: {active ? 'YES' : 'NO'}</p>
        <p>Auto-Pilot: {isAutoPilot ? 'ON' : 'OFF'}</p>
        <hr className="border-white/20 my-1" />
        {log.map((l, i) => <div key={i}>{l}</div>)}
      </div>

      {/* CURSOR GHOST */}
      <div
        className="absolute w-6 h-6 border-2 border-red-500 rounded-full z-[9999] transition-all duration-500 ease-out pointer-events-none -translate-x-1/2 -translate-y-1/2 shadow-[0_0_15px_rgba(255,0,0,0.8)]"
        style={{ left: cursor.x, top: cursor.y }}
      />

      {/* TARGET HIGHLIGHT BOX */}
      {highlightBox && (
        <div
          className="absolute border-2 border-yellow-400 bg-yellow-400/10 z-[50] rounded pointer-events-none animate-pulse"
          style={{
            left: highlightBox.x,
            top: highlightBox.y,
            width: highlightBox.w,
            height: highlightBox.h
          }}
        />
      )}

      {/* 1. The Floating Input Bar (Spotlight Style) */}
      {/* Only visible when IDLE or user hits hotkey OR IS SCANNING */}
      <div className={`absolute top-[20%] left-1/2 -translate-x-1/2 w-[600px] z-50 transition-all duration-300 ${!isScanning && (active || step === 'IDLE') ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'}`}>
        <div className="bg-neutral-900/90 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl shadow-black/50 overflow-hidden text-white pointer-events-auto">
          <div className="flex items-center px-4 py-4 gap-3 border-b border-white/5">
            <div className={`p-1.5 rounded text-white ${isAutoPilot ? 'bg-purple-600 animate-pulse' : 'bg-blue-600'}`}>
              <Cpu size={16} />
            </div>
            <input
              id="agent-input"
              type="text"
              className="bg-transparent border-none outline-none flex-1 text-lg placeholder-white/30"
              placeholder={isAutoPilot ? "Auto-Pilot Active... (Press Esc to Stop)" : "Ask Ghost Agent..."}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleCommand}
              disabled={isAutoPilot}
            />
            <div className="text-xs text-white/30 font-mono bg-white/5 px-2 py-1 rounded">
              Cmd+K
            </div>
          </div>

          {/* AUTO PILOT TOGGLE */}
          {!active && step === 'IDLE' && (
            <div className="px-4 py-2 bg-white/5 flex justify-between items-center">
              <span className="text-xs text-gray-400">Experimental Mode</span>
              <button
                onClick={() => setIsAutoPilot(!isAutoPilot)}
                className={`text-xs px-3 py-1 rounded font-bold transition-colors ${isAutoPilot ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-purple-600 hover:bg-purple-500 text-white'}`}
              >
                {isAutoPilot ? 'STOP AUTO-PILOT' : 'START AUTO-PILOT'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 2. The "Ghost" Visualizer (Target Box) */}
      {/* This draws boxes around elements the agent is looking at */}
      {
        highlightBox && (
          <div
            className="absolute z-40 border-2 border-red-500 bg-red-500/10 rounded pointer-events-none transition-all duration-300 animate-pulse"
            style={{
              left: highlightBox.x,
              top: highlightBox.y,
              width: highlightBox.w,
              height: highlightBox.h
            }}
          >
            {/* Label Tag */}
            <div className="absolute -top-6 left-0 bg-red-600 text-white text-[10px] px-2 py-0.5 rounded-t font-mono">
              TARGET: {highlightBox ? "MATCH_FOUND" : "SCANNING"}
            </div>
          </div>
        )
      }

      {/* 3. The "Ghost" Cursor */}
      {/* Simulates the mouse moving autonomously */}
      {
        step !== 'DONE' && (
          <div
            className="absolute z-50 pointer-events-none transition-transform duration-75"
            style={{
              left: `${cursor.x}px`,
              top: `${cursor.y}px`,
              transform: `translate(-50%, -50%) ${step === 'EXECUTING' ? 'scale(0.9)' : 'scale(1)'}`,
              opacity: step === 'IDLE' ? 0.5 : 1 // Dim when idle
            }}
          >
            {/* The Cursor Graphic */}
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="drop-shadow-xl">
              <path d="M5.65376 12.3673H5.46026L5.31717 12.4976L0.500002 16.8829L0.500002 1.19841L11.7841 12.3673H5.65376Z" fill="#3b82f6" stroke="white" strokeWidth="1" />
            </svg>

            {/* Agent Tag */}
            <div className="absolute top-5 left-5 bg-blue-600 text-white text-[9px] px-1.5 py-0.5 rounded font-bold whitespace-nowrap opacity-80 shadow-sm">

              AGENT
            </div>
          </div>
        )
      }


      {/* 4. The Review/Security Modal (Overlay) */}
      {/* This dims the screen and forces interaction */}
      {
        step === 'REVIEW' && (
          <div className="absolute inset-0 z-[60] bg-black/40 backdrop-blur-sm flex items-center justify-center animate-in fade-in duration-200 pointer-events-auto">
            <div className="w-[400px] bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl overflow-hidden ring-1 ring-white/10">

              {/* Header */}
              <div className="h-14 bg-gradient-to-r from-blue-900/50 to-purple-900/50 border-b border-white/5 flex items-center px-6 justify-between">
                <div className="flex items-center gap-2">
                  <Lock className="text-blue-400" size={16} />
                  <span className="font-semibold text-white tracking-wide text-sm">Action Approval</span>
                </div>
                <span className="text-[10px] text-blue-200 bg-blue-500/20 border border-blue-500/30 px-2 py-0.5 rounded">High Risk</span>
              </div>

              {/* Content */}
              <div className="p-6 space-y-4">
                <p className="text-neutral-300 text-sm leading-relaxed">
                  The agent is preparing to execute a financial transaction.
                </p>

                <div className="bg-black/50 rounded-lg p-3 border border-white/5 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-500">Action</span>
                    <span className="text-white font-mono">CLICK_LEFT</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-500">Target</span>
                    <span className="text-white font-mono">{pendingAction?.description || "Unknown Target"}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-neutral-500">Value</span>
                    <span className="text-white font-mono">{pendingAction?.value || "N/A"}</span>
                  </div>
                </div>

                <div className="flex items-center gap-3 text-xs text-neutral-400 bg-white/5 p-2 rounded">
                  <CreditCard size={14} />
                  <span>Card ending in 4242 will be charged.</span>
                </div>
              </div>

              {/* Footer Actions */}
              <div className="p-4 bg-neutral-950 border-t border-white/5 flex gap-3">
                <button
                  onClick={() => setStep('IDLE')}
                  className="flex-1 py-2.5 rounded-lg border border-white/10 text-neutral-400 hover:bg-white/5 hover:text-white transition-colors text-sm font-medium"
                >
                  Abort
                </button>
                <button
                  onClick={handleApprove}
                  className="flex-1 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 transition-all text-sm font-semibold flex items-center justify-center gap-2"
                >
                  <CheckCircle2 size={16} />
                  Authorize
                </button>
              </div>
            </div>
          </div>
        )
      }

      {/* 5. Success Toast */}
      {
        step === 'DONE' && (
          <div className="absolute top-10 left-1/2 -translate-x-1/2 z-[70] bg-green-500 text-white px-6 py-3 rounded-full shadow-xl flex items-center gap-3 animate-in slide-in-from-top-4">
            <CheckCircle2 size={20} />
            <span className="font-medium">Task Completed Successfully</span>
          </div>
        )
      }

    </div >
  );
}
