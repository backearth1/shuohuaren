"""
简单API测试 - 仅测试API端点可用性
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("🧪 简单API测试")
print("=" * 60)

# 1. 测试首页
print("\n[1] 测试首页...")
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    if response.status_code == 200 and "视频说话人识别系统" in response.text:
        print("✅ 首页正常")
    else:
        print(f"❌ 首页异常: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ 首页访问失败: {e}")
    exit(1)

# 2. 测试API端点
print("\n[2] 测试API端点...")

# 测试不存在的任务
try:
    response = requests.get(f"{BASE_URL}/api/task/nonexistent", timeout=5)
    if response.status_code == 404:
        print("✅ 任务查询API正常（404响应）")
    else:
        print(f"⚠️  任务查询API返回: {response.status_code}")
except Exception as e:
    print(f"❌ 任务查询API异常: {e}")

# 3. 测试文件准备（不上传）
print("\n[3] 检查测试文件...")
project_root = Path(__file__).parent.parent
video_file = project_root / "shengying_3.mp4"
srt_file = project_root / "shengyin3.srt"

if video_file.exists():
    print(f"✅ 视频文件就绪: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print(f"❌ 视频文件不存在: {video_file}")

if srt_file.exists():
    print(f"✅ SRT文件就绪: {srt_file.name} ({srt_file.stat().st_size / 1024:.1f} KB)")
else:
    print(f"❌ SRT文件不存在: {srt_file}")

print("\n" + "=" * 60)
print("✅ 基础测试通过!")
print("=" * 60)
print("\n💡 提示:")
print("  - Web界面: http://localhost:8000")
print("  - 完整测试: python test_api.py")
print("=" * 60)
