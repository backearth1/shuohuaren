# VLM命名功能设置指南

## 功能说明

VLM（Vision Language Model）命名功能可以自动为识别出的说话人生成：
- 中文姓名
- 角色定位（主角/配角/次要角色）
- 性别判断
- 外观特征描述（服装、面部、年龄）
- 性格分析
- 人物关系推断

## 启用VLM命名

### 方法1: 设置环境变量（推荐）

在启动服务器前设置环境变量：

```bash
export MINIMAX_API_KEY="your-api-key-here"
cd /data1/devin/test/pyannote_speaker_diarization/speaker_diarization_web/backend
python app.py
```

### 方法2: 在启动脚本中设置

创建启动脚本 `start_server.sh`:

```bash
#!/bin/bash
export MINIMAX_API_KEY="your-api-key-here"
cd /data1/devin/test/pyannote_speaker_diarization/speaker_diarization_web/backend
python app.py
```

然后运行：
```bash
chmod +x start_server.sh
./start_server.sh
```

### 方法3: 使用一行命令

```bash
MINIMAX_API_KEY="your-api-key-here" python app.py
```

## 查看VLM调用日志

启用VLM后，在服务器日志中会看到以下信息：

### 1. 启动时确认
```
✅ 检测到MINIMAX_API_KEY，将启用VLM命名
```

或

```
⚠️  未检测到MINIMAX_API_KEY，跳过VLM命名
```

### 2. VLM调用过程
```
[80%] 调用VLM进行人物命名...

调用MiniMax VLM...
  API: https://api.minimaxi.com/v1/text/chatcompletion_v2
  模型: MiniMax-Text-01
  图片数量: 10
  Prompt长度: 12345 字符
  Trace-ID: 01234567-89ab-cdef-0123-456789abcdef
  HTTP状态码: 200
✓ VLM调用成功
  Trace-ID: 01234567-89ab-cdef-0123-456789abcdef
  返回内容长度: 5678 字符
```

### 3. 错误情况

如果VLM调用失败，会显示详细错误信息：

```
✗ VLM调用HTTP错误: 401 Client Error
  Trace-ID: 01234567-89ab-cdef-0123-456789abcdef
  HTTP状态码: 401
  错误详情: {"error": "Invalid API key"}
```

## Trace-ID 的作用

**Trace-ID** 是MiniMax API返回的请求追踪ID，用于：
- 追踪特定的API调用
- 排查问题时向技术支持提供
- 监控API使用情况

如果VLM命名出现问题，请：
1. 查看服务器日志中的 `Trace-ID`
2. 记录 `HTTP状态码` 和 `错误详情`
3. 联系MiniMax技术支持并提供Trace-ID

## 实时监控服务器日志

启动服务器后，可以用以下命令实时查看日志：

```bash
# 查看后台进程输出（如果使用nohup启动）
tail -f /tmp/fastapi_5445.log

# 或者直接在前台运行服务器
cd /data1/devin/test/pyannote_speaker_diarization/speaker_diarization_web/backend
python app.py
```

## 测试VLM功能

1. 设置API Key
2. 启动服务器
3. 访问 http://localhost:5445
4. 上传视频和SRT文件
5. 观察日志，确认看到：
   - `✅ 检测到MINIMAX_API_KEY，将启用VLM命名`
   - `[80%] 调用VLM进行人物命名...`
   - `Trace-ID: ...`

## 常见问题

### Q: 看不到Trace-ID？
A: 确保：
1. 已设置 `MINIMAX_API_KEY` 环境变量
2. 服务器已重启（使用最新代码）
3. 查看完整服务器日志

### Q: VLM调用失败？
A: 检查：
1. API Key是否正确
2. 网络连接是否正常
3. 查看错误详情中的具体原因
4. 使用Trace-ID联系技术支持

### Q: 如何禁用VLM命名？
A: 不设置 `MINIMAX_API_KEY` 环境变量即可，系统会自动跳过VLM命名步骤。

## 当前服务器状态

服务器地址: http://localhost:5445
状态: 运行中
VLM功能: 已集成（需要设置API Key）
Trace-ID日志: 已启用
