# ♟️ AI Chess Arena

A full-stack chess application with real-time multiplayer support and AI-powered gameplay.

## 🚀 Live Demo
- Frontend: https://chessai-e0c8c.web.app/
- Backend: https://chess-backend-1072267285798.us-central1.run.app



## 🧠 Features

- ♟️ **Player vs Player (PVP)** mode
- 🤖 **Player vs AI (PVAI)** mode
- 🧠 AI move generation using a trained PyTorch model
- 🔁 Real-time game state updates
- 🎯 Legal move validation using `python-chess`
- 🎨 Interactive chessboard with custom UI
- ☁️ Fully deployed on Google Cloud (Cloud Run + Firebase)


## 🏗️ Tech Stack

### Frontend
- React (Vite)
- TypeScript
- CSS

### Backend
- FastAPI
- Python
- python-chess
- PyTorch (model inference)

### Cloud & Deployment
- Google Cloud Run (backend)
- Firebase Hosting (frontend)
- Google Cloud Datastore (persistent storage)

## ☁️ Deployment
### Backend (Cloud Run)
```bash
cd backend
gcloud run deploy chess-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```
### Frontend (Firebase)
```bash
cd frontend
npm run build
cd ..

firebase deploy --only hosting
```

## ⚙️ Setup Instructions to run locally

### 🔹 1. Clone the project locally

### 🔹 2. GCP Local Setup
```bash
gcloud config set project shatigoai
gcloud auth application-default login
gcloud auth application-default set-quota-project shatigoai
```

### 🔹 3. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Place your trained model at:
```bash
backend/latest.pt
```
Run backend:
```bash
python -m uvicorn main:app --port 8080
```
### 🔹 4. Frontend Setup
```bash
cd frontend
npm install
```
Create .env:
```bash
VITE_API_BASE=http://localhost:8080
```
Run frontend:
```bash
npm run dev
```
## ⚠️ Notes
- latest.pt is not included due to size constraints.
- Ensure correct CORS settings for deployed frontend.
- Assets must be placed in frontend/public/ for production builds.

## 🎯 Future Improvements
- Move history panel
- Difficulty levels for AI
- Game replay feature
- Authentication system
- WebSocket-based real-time updates

## 👤 Author
Harini Sai Padamata

## 📜 License

This project is for educational and demonstration purposes.
