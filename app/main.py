import os
from dotenv import load_dotenv

# Load environment variables from backend/.env if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import engine, Base, SessionLocal
from app.models import User
from app.auth import hash_password
from app.routes import auth_routes, dashboard_routes, detection_routes, admin_routes, model_routes

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Chemical Storage Cabinet Monitoring System API",
    description="Software-only AI monitoring platform for chemical storage cabinet gas exfiltration alerts.",
    version="1.0.0"
)

# Enable CORS for local Vite dev server and deployed frontend
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url.strip().rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Storage directory for serving original images and visualizations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

# Include Router Modules
app.include_router(auth_routes.router)
app.include_router(dashboard_routes.router)
app.include_router(detection_routes.router)
app.include_router(admin_routes.router)
app.include_router(model_routes.router)

@app.on_event("startup")
def startup_event():
    # Seed default Admin user if database is empty
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@lab.com").first()
        if not admin_user:
            default_admin = User(
                email="admin@lab.com",
                full_name="Lab Safety Administrator",
                hashed_password=hash_password("admin123"),
                role="admin"
            )
            db.add(default_admin)
            db.commit()
            print("Seeded default admin account: admin@lab.com / admin123")
    finally:
        db.close()

@app.get("/")
def root():
    return {
        "system": "Smart Chemical Storage Cabinet Monitoring System",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs"
    }
