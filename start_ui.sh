#!/bin/bash

# Setup colors
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0;5m' # No Color
RESET='\033[0m'

echo -e "${PURPLE}====================================================${RESET}"
echo -e "${PURPLE}       LM Evaluation Harness Dashboard Launcher     ${RESET}"
echo -e "${PURPLE}====================================================${RESET}"

# Function to clean up background processes on exit
cleanup() {
    echo -e "\n${CYAN}[System] Shutting down UI servers...${RESET}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Catch Ctrl+C and exit signals
trap cleanup SIGINT SIGTERM EXIT

# Start FastAPI Backend
echo -e "${GREEN}[Backend] Starting FastAPI Server on http://127.0.0.1:8000...${RESET}"
.venv/bin/uvicorn ui.backend.app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait 2 seconds for backend to initialize
sleep 2

# Start React/Vite Frontend
echo -e "${GREEN}[Frontend] Starting Vite Development Server...${RESET}"
npm run dev --prefix ui/frontend &
FRONTEND_PID=$!

echo -e "${CYAN}[System] Both servers are running!${RESET}"
echo -e "${CYAN}[System] Open http://localhost:5173 in your browser.${RESET}"
echo -e "${CYAN}[System] Press Ctrl+C to terminate both servers.${RESET}"
echo -e "${PURPLE}====================================================${RESET}"

# Keep script running to capture logs/signals
wait
