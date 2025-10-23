"""
FastAPI后端 - 说话人识别Web服务
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import shutil
from datetime import datetime
import json

app = FastAPI(
    title="Speaker Diarization Web Service",
    description="视频说话人识别与命名服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
UPLOAD_DIR = "./uploads"
RESULT_DIR = "./results"
FRONTEND_DIR = "../frontend"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")

# 任务状态存储（生产环境应该用数据库）
tasks = {}


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    result: Optional[dict] = None
    created_at: str


@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/upload")
async def upload_files(
    video: UploadFile = File(...),
    srt: UploadFile = File(...),
    api_key: str = Form(None),
    vlm_provider: str = Form("qwen"),
    background_tasks: BackgroundTasks = None
):
    """
    上传视频和SRT文件

    Args:
        video: 视频文件（mp4, avi等）
        srt: SRT字幕文件
        api_key: VLM API Key（可选，用于VLM命名）
        vlm_provider: VLM提供商（qwen或minimax，默认qwen）

    Returns:
        task_id: 任务ID，用于查询处理状态
    """
    # 验证文件格式
    if not video.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="视频格式不支持，请上传mp4/avi/mov/mkv文件")

    if not srt.filename.endswith('.srt'):
        raise HTTPException(status_code=400, detail="字幕格式不支持，请上传srt文件")

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 创建任务目录
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存上传的文件
    video_path = os.path.join(task_dir, "video" + os.path.splitext(video.filename)[1])
    srt_path = os.path.join(task_dir, "subtitle.srt")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    with open(srt_path, "wb") as f:
        shutil.copyfileobj(srt.file, f)

    # 初始化任务状态
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建，等待处理...",
        "video_path": video_path,
        "srt_path": srt_path,
        "api_key": api_key,  # 保存API Key
        "vlm_provider": vlm_provider,  # 保存VLM提供商
        "created_at": datetime.now().isoformat()
    }

    # 添加后台任务
    background_tasks.add_task(process_task, task_id, video_path, srt_path, api_key, vlm_provider)

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "message": "文件上传成功，开始处理..."
    })


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    查询任务状态

    Args:
        task_id: 任务ID

    Returns:
        任务状态信息
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    return JSONResponse(tasks[task_id])


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """
    获取处理结果

    Args:
        task_id: 任务ID

    Returns:
        处理结果（JSON格式）
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    # 读取结果文件
    result_dir = os.path.join(RESULT_DIR, task_id)
    result_json_path = os.path.join(result_dir, "result.json")

    if not os.path.exists(result_json_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")

    with open(result_json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    return JSONResponse(result)


def process_task(task_id: str, video_path: str, srt_path: str, api_key: str = None, vlm_provider: str = "qwen"):
    """
    后台任务：处理视频和SRT

    Args:
        task_id: 任务ID
        video_path: 视频文件路径
        srt_path: SRT文件路径
        api_key: VLM API Key（可选）
        vlm_provider: VLM提供商（qwen或minimax）
    """
    try:
        # 更新状态：开始处理
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["message"] = "解析SRT文件..."

        # 调用核心处理pipeline
        from pipeline import process_video_pipeline

        result_dir = os.path.join(RESULT_DIR, task_id)
        os.makedirs(result_dir, exist_ok=True)

        # 执行pipeline（带进度回调）
        def update_progress(progress: int, message: str):
            tasks[task_id]["progress"] = progress
            tasks[task_id]["message"] = message

        # 使用前端传来的API Key，如果没有则尝试从环境变量或配置获取
        final_api_key = api_key
        if not final_api_key:
            if vlm_provider == "qwen":
                final_api_key = os.getenv("QWEN_API_KEY")
            else:
                final_api_key = os.getenv("MINIMAX_API_KEY")

        if final_api_key:
            print(f"✅ 检测到API Key（提供商: {vlm_provider}，来源: {'前端输入' if api_key else '环境变量'}），将启用VLM命名")
        else:
            print(f"⚠️  未提供API Key，跳过VLM命名")

        result = process_video_pipeline(
            video_path=video_path,
            srt_path=srt_path,
            output_dir=result_dir,
            progress_callback=update_progress,
            api_key=final_api_key,
            vlm_provider=vlm_provider
        )

        # 保存结果
        result_json_path = os.path.join(result_dir, "result.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 更新状态：完成
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "处理完成！"
        tasks[task_id]["result"] = result

    except Exception as e:
        # 更新状态：失败
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"处理失败: {str(e)}"
        import traceback
        print(f"Task {task_id} failed:")
        print(traceback.format_exc())


if __name__ == "__main__":
    import uvicorn
    # 如果5444被占用，可以尝试其他端口
    import socket
    port = 5444
    # 测试端口是否可用
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        print(f"✅ 使用端口 {port}")
    except OSError:
        print(f"⚠️  端口 {port} 被占用，尝试端口 5445")
        port = 5445

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
