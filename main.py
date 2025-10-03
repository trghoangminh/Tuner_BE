# Import các thư viện cần thiết cho FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import các router từ thư mục routers
from routers import analyze as analyze_router  # Router xử lý phân tích file âm thanh
from routers import ws as ws_router  # Router xử lý WebSocket real-time

# Tạo ứng dụng FastAPI với tiêu đề "Tuner Backend"
app = FastAPI(title="Tuner Backend")

# Cấu hình CORS (Cross-Origin Resource Sharing) để cho phép frontend kết nối
# allow_origins=["*"]: cho phép tất cả domain kết nối
# allow_credentials=True: cho phép gửi cookies
# allow_methods=["*"]: cho phép tất cả HTTP methods (GET, POST, etc.)
# allow_headers=["*"]: cho phép tất cả headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router vào ứng dụng chính
# /analyze: endpoint để phân tích file âm thanh tĩnh
app.include_router(analyze_router.router, prefix="/analyze", tags=["analyze"])
# /ws: endpoint WebSocket để phân tích âm thanh real-time
app.include_router(ws_router.router, tags=["ws"])


@app.get("/")
def root():
    """Endpoint gốc để kiểm tra server có hoạt động không"""
    return {"status": "ok"}


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe của server"""
    return {"status": "healthy"}


