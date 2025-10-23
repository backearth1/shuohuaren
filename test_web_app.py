"""
测试Web应用
"""

import os
import sys
import requests
import time
from pathlib import Path

# 设置环境
backend_dir = Path(__file__).parent / "backend"
os.chdir(backend_dir)

print("=" * 60)
print("📋 Web应用测试脚本")
print("=" * 60)

# 检查是否有测试文件
project_root = Path(__file__).parent.parent
video_file = project_root / "shengying_3.mp4"
srt_file = project_root / "shengyin3.srt"

print(f"\n✅ 项目根目录: {project_root}")
print(f"✅ 测试视频: {video_file.name} ({'存在' if video_file.exists() else '❌ 不存在'})")
print(f"✅ 测试字幕: {srt_file.name} ({'存在' if srt_file.exists() else '❌ 不存在'})")

print("\n" + "=" * 60)
print("🚀 启动说明")
print("=" * 60)
print("\n要启动Web应用，请运行以下命令：")
print(f"\ncd {backend_dir}")
print("python app.py")
print("\n然后在浏览器中访问: http://localhost:8000")
print("\n或使用curl测试API:")
print("curl http://localhost:8000/")
print("\n" + "=" * 60)
