from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyze as analyze_router
from routers import ws as ws_router


app = FastAPI(title="Tuner Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router.router, prefix="/analyze", tags=["analyze"])
app.include_router(ws_router.router, tags=["ws"])


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


