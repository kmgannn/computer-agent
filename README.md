# Tauri + React + Typescript

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

Walkthrough - Commerce Agent (New Repo)
A fully scaffolded Tauri v2 application with a Python Computer Vision backend.

Project Structure
commerce_agent/ - Root Tauri Project
src-tauri/ - Rust Backend (Window Control)
src/ - React Frontend (Overlay UI)
backend/ - Python AI Backend (EasyOCR + Logic)
Setup
1. Prerequisites
Node.js & NPM
Python 3.10+
Rust (Cargo)
CUDA Toolkit (Optional, for GPU acceleration)

2. Frontend & Rust Setup
Navigate to the project root and install dependencies:
npm install

3. Python Backend Setup
Navigate to the backend folder and install Python dependencies:
cd backend
pip install -r requirements.txt
Note: If you have a GPU, ensure you install the CUDA-enabled version of PyTorch manually if needed.

Running the Agent
Step 1: Start the Python Brain
This must be running for the agent to "see".

# In terminal 1
Go to the backend directory \backend
python main.py
Wait for "✅ BRAIN READY" message.

Step 2: Start the Tauri Overlay
# In terminal 2
npm run tauri dev
Verification Scenarios

1. Overlay Check
Expected: A transparent window covers your screen. You can click through it.
Action: Press Cmd+K (or Ctrl+K) to focus the agent bar.
2. Auto-Pilot Test
Go to a commerce site (e.g., Amazon).
Type "Logitech Mouse" into the agent bar.
Click "START AUTO-PILOT".
Observe: The agent should scan, move the mouse, and navigate.
Review: When "Place Order" is found, the Action Approval modal should appear and block clicks until you approve.
