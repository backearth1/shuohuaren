"""
测试Web API - 完整流程测试
"""

import requests
import time
import json
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:8000"

print("=" * 60)
print("🧪 Web API测试")
print("=" * 60)

# 1. 测试首页
print("\n[1] 测试首页访问...")
try:
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("✅ 首页访问成功")
    else:
        print(f"❌ 首页访问失败: {response.status_code}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# 2. 准备测试文件
print("\n[2] 准备测试文件...")
project_root = Path(__file__).parent.parent
video_file = project_root / "shengying_3.mp4"
srt_file = project_root / "shengyin3.srt"

if not video_file.exists():
    print(f"❌ 视频文件不存在: {video_file}")
    exit(1)

if not srt_file.exists():
    print(f"❌ SRT文件不存在: {srt_file}")
    exit(1)

print(f"✅ 视频文件: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"✅ SRT文件: {srt_file.name} ({srt_file.stat().st_size / 1024:.1f} KB)")

# 3. 上传文件
print("\n[3] 上传文件...")
print("⚠️  注意: 这将触发完整的处理流程，可能需要几分钟")
print("⚠️  如果不想运行完整测试，请按 Ctrl+C 退出")

# 询问是否继续
try:
    response = input("\n是否继续完整测试? (y/N): ")
    if response.lower() != 'y':
        print("\n✋ 测试已取消")
        exit(0)
except KeyboardInterrupt:
    print("\n\n✋ 测试已取消")
    exit(0)

print("\n📤 开始上传...")
with open(video_file, 'rb') as vf, open(srt_file, 'rb') as sf:
    files = {
        'video': (video_file.name, vf, 'video/mp4'),
        'srt': (srt_file.name, sf, 'text/plain')
    }

    try:
        response = requests.post(f"{BASE_URL}/api/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                task_id = result['task_id']
                print(f"✅ 上传成功! Task ID: {task_id}")
            else:
                print(f"❌ 上传失败: {result.get('message')}")
                exit(1)
        else:
            print(f"❌ 上传失败: HTTP {response.status_code}")
            print(response.text)
            exit(1)
    except Exception as e:
        print(f"❌ 上传异常: {e}")
        exit(1)

# 4. 轮询任务状态
print("\n[4] 监控任务进度...")
print("=" * 60)

last_progress = -1
start_time = time.time()

while True:
    try:
        response = requests.get(f"{BASE_URL}/api/task/{task_id}")
        if response.status_code == 200:
            task = response.json()

            progress = task.get('progress', 0)
            status = task.get('status', 'unknown')
            message = task.get('message', '')

            # 只在进度变化时输出
            if progress != last_progress:
                elapsed = time.time() - start_time
                print(f"[{int(elapsed)}s] [{progress}%] {message}")
                last_progress = progress

            # 检查状态
            if status == 'completed':
                print("\n✅ 处理完成!")
                break
            elif status == 'failed':
                print(f"\n❌ 处理失败: {message}")
                exit(1)

        else:
            print(f"❌ 查询状态失败: {response.status_code}")
            exit(1)

    except Exception as e:
        print(f"❌ 查询异常: {e}")
        exit(1)

    time.sleep(2)  # 每2秒轮询一次

# 5. 获取结果
print("\n[5] 获取处理结果...")
print("=" * 60)

try:
    response = requests.get(f"{BASE_URL}/api/result/{task_id}")
    if response.status_code == 200:
        result = response.json()

        print(f"\n📊 统计信息:")
        print(f"  识别人数: {result.get('num_speakers')}")
        print(f"  对话片段: {result.get('video_info', {}).get('total_segments')}")
        print(f"  检测人脸: {result.get('video_info', {}).get('total_faces_detected')}")
        print(f"  有效人脸: {result.get('video_info', {}).get('total_faces_valid')}")
        print(f"  VLM命名: {'已启用' if result.get('vlm_enabled') else '未启用'}")

        print(f"\n👥 说话人列表:")
        for speaker in result.get('speakers', []):
            name = speaker.get('name', f"Speaker {speaker.get('speaker_id')}")
            role = speaker.get('role', '')
            gender = speaker.get('gender', '')
            face_count = speaker.get('face_count', 0)
            segment_count = speaker.get('segment_count', 0)

            role_str = f" [{role}]" if role else ""
            gender_str = f" ({gender})" if gender else ""

            print(f"  {name}{role_str}{gender_str}: {face_count}张人脸, {segment_count}个片段")

        # 保存结果到文件
        output_file = Path(__file__).parent / "web_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整结果已保存到: {output_file}")

        print(f"\n🌐 在浏览器中查看结果:")
        print(f"   {BASE_URL}/")
        print(f"   (Task ID: {task_id})")

    else:
        print(f"❌ 获取结果失败: {response.status_code}")
        print(response.text)
        exit(1)

except Exception as e:
    print(f"❌ 获取结果异常: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🎉 测试完成!")
print("=" * 60)
