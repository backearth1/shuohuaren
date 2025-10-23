# 视频说话人识别与命名Web系统

基于人脸检测、DBSCAN聚类和VLM的视频说话人自动识别与命名系统。

## 功能特性

- ✅ **视频人脸检测**: 使用MTCNN进行高质量人脸检测
- ✅ **说话人聚类**: 基于DBSCAN算法的人脸聚类（优化参数：eps=0.28, min_samples=5）
- ✅ **智能命名**: 可选VLM（MiniMax）进行人物命名和角色分析
- ✅ **Web界面**: 简洁美观的上传和结果展示界面
- ✅ **进度跟踪**: 实时显示处理进度
- ✅ **详细分析**: 提供外观特征、性格分析、人物关系等信息

## 项目结构

```
speaker_diarization_web/
├── backend/
│   ├── app.py           # FastAPI主应用
│   └── pipeline.py      # 核心处理Pipeline
├── frontend/
│   └── index.html       # Web前端页面
├── uploads/             # 上传文件临时目录
├── results/             # 处理结果输出目录
├── test_api.py          # 完整API测试脚本
├── test_api_simple.py   # 简单API测试脚本
└── README.md
```

## 安装依赖

### 1. Python依赖

```bash
# 核心依赖
pip install fastapi uvicorn python-multipart
pip install torch torchvision
pip install facenet-pytorch
pip install opencv-python
pip install scikit-learn
pip install hdbscan

# VLM命名（可选）
pip install requests
```

### 2. 系统要求

- Python 3.8+
- CUDA（可选，用于GPU加速）
- 8GB+ RAM推荐

## 使用方法

### 1. 启动服务器

```bash
cd speaker_diarization_web/backend
python app.py
```

服务器将在 `http://localhost:8000` 启动。

### 2. 使用Web界面

1. 打开浏览器访问: `http://localhost:8000`
2. 上传视频文件（支持MP4, AVI, MOV, MKV）
3. 上传对应的SRT字幕文件
4. 点击"开始处理"
5. 等待处理完成（会显示实时进度）
6. 查看识别结果

### 3. 使用API

#### 上传文件

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "video=@video.mp4" \
  -F "srt=@subtitle.srt"
```

响应:
```json
{
  "success": true,
  "task_id": "uuid-task-id",
  "message": "文件上传成功，开始处理..."
}
```

#### 查询任务状态

```bash
curl http://localhost:8000/api/task/{task_id}
```

响应:
```json
{
  "task_id": "uuid-task-id",
  "status": "processing",
  "progress": 45,
  "message": "提取人脸特征向量...",
  "created_at": "2025-10-22T12:00:00"
}
```

#### 获取结果

```bash
curl http://localhost:8000/api/result/{task_id}
```

响应:
```json
{
  "num_speakers": 10,
  "speakers": [
    {
      "speaker_id": 1,
      "name": "阮娇",
      "role": "主角",
      "gender": "女",
      "face_count": 225,
      "segment_count": 62,
      "face_images": ["faces/speaker_1_frame_1.jpg", ...],
      "appearance": {
        "clothing": "白色上衣搭配黑色裙子",
        "facial_features": "长发，面容姣好",
        "age_estimate": "20-30岁"
      },
      "character_analysis": {
        "personality": "强势、自信",
        "importance": "非常高"
      }
    }
  ],
  "video_info": {
    "total_segments": 248,
    "total_faces_detected": 595,
    "total_faces_valid": 595
  }
}
```

## 测试

### 简单测试（仅测试API可用性）

```bash
cd speaker_diarization_web
python test_api_simple.py
```

### 完整测试（包含上传和处理）

```bash
cd speaker_diarization_web
python test_api.py
```

**注意**: 完整测试需要几分钟，会实际处理视频文件。

## 配置

### DBSCAN参数调整

在 `backend/pipeline.py` 中修改:

```python
result = process_video_pipeline(
    video_path=video_path,
    srt_path=srt_path,
    output_dir=result_dir,
    eps=0.28,           # 调整聚类距离阈值
    min_samples=5,      # 调整最小样本数
    api_key=None        # 设置MiniMax API Key启用VLM命名
)
```

### 启用VLM命名

1. 获取MiniMax API Key
2. 在环境变量中设置:
   ```bash
   export MINIMAX_API_KEY="your-api-key"
   ```
3. 修改 `backend/app.py` 的 `process_task` 函数:
   ```python
   result = process_video_pipeline(
       video_path=video_path,
       srt_path=srt_path,
       output_dir=result_dir,
       progress_callback=update_progress,
       api_key=os.getenv("MINIMAX_API_KEY")  # 启用VLM
   )
   ```

## 处理流程

1. **解析SRT** (10%): 解析字幕文件，提取时间片段
2. **抽取关键帧** (20%): 从每个字幕片段中抽取5帧
3. **人脸检测** (30%): 使用MTCNN检测人脸并进行质量过滤
4. **特征提取** (45%): 使用FaceNet提取512维特征向量
5. **DBSCAN聚类** (60%): 将人脸聚类为不同说话人
6. **生成代表图片** (70%): 为每个说话人选择2张最佳人脸图片
7. **统计分析** (75%): 统计每个说话人的人脸数、片段数
8. **VLM命名** (80%, 可选): 使用VLM分析人物特征并命名
9. **生成结果** (95%): 汇总所有信息生成最终结果

## 性能优化

- **GPU加速**: 自动检测CUDA，优先使用GPU
- **质量过滤**:
  - 置信度阈值: 0.95
  - 最小人脸面积: 1.5%画幅
  - 侧脸检测: 眼睛-鼻子距离比
- **单次VLM调用**: 所有人脸（20张）一次性发送，节省API费用

## 技术栈

- **后端**: FastAPI + Uvicorn
- **前端**: Bootstrap 5 + Font Awesome + Vanilla JS
- **人脸检测**: MTCNN (facenet-pytorch)
- **特征提取**: FaceNet (InceptionResnetV1)
- **聚类**: DBSCAN (scikit-learn)
- **VLM**: MiniMax API

## 测试结果

基于 `shengying_3.mp4` (8分钟视频) 的测试结果:

- ✅ 识别人数: 10个说话人（与实际相符）
- ✅ 检测人脸: 595张高质量人脸
- ✅ 主要角色命名准确度: 100%（阮娇、裴宴、小叔）
- ✅ 处理时间: ~5分钟（包括VLM命名）

## 常见问题

### Q: 服务器启动失败？
A: 检查8000端口是否被占用，可以修改 `app.py` 中的端口号。

### Q: 人脸检测失败？
A: 确保视频中有清晰的人脸镜头，尝试降低 `face_detection_poc.py` 中的质量阈值。

### Q: 聚类结果不准确？
A: 运行 `tune_clustering.py` 调整DBSCAN参数。

### Q: VLM命名失败？
A: 检查API Key是否正确，网络连接是否正常。

## License

MIT

## 作者

基于pyannote speaker diarization项目开发

## 致谢

- MTCNN人脸检测算法
- FaceNet人脸识别模型
- MiniMax VLM API
