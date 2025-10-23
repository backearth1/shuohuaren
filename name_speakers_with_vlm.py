"""
使用VLM给聚类后的说话人命名
方案：传完整画面（红框标注目标人脸） + SRT对话内容
"""

import json
import os
import cv2
import numpy as np
import base64
import requests
from io import BytesIO
from PIL import Image
from collections import defaultdict


def select_best_frames_per_speaker(clustering_result_json, valid_faces, num_frames=2):
    """
    为每个说话人选择最佳的代表帧

    选择策略：
    1. 人脸最大（距离镜头最近）
    2. 人脸置信度最高
    3. 均匀分布在时间轴上
    """
    with open(clustering_result_json, 'r', encoding='utf-8') as f:
        clustering_result = json.load(f)

    # 重建label到faces的映射
    label_to_faces = defaultdict(list)
    for face in valid_faces:
        # 需要知道每个face属于哪个speaker
        # 这里需要从聚类结果反推
        pass

    # 注：这部分需要传入labels数组
    # 暂时先设计接口

    return {}


def annotate_target_face(frame, face_box, color=(0, 0, 255), thickness=3):
    """
    在完整画面上用红框标注目标人脸

    Args:
        frame: 完整画面
        face_box: 人脸边界框 [x1, y1, x2, y2]
        color: BGR颜色 (默认红色)
        thickness: 线条粗细
    """
    annotated_frame = frame.copy()

    x1, y1, x2, y2 = [int(b) for b in face_box]

    # 画红框
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

    # 可选：在框上方添加箭头或文字
    arrow_start = (x1 + (x2 - x1) // 2, y1 - 20)
    arrow_end = (x1 + (x2 - x1) // 2, y1 - 5)
    cv2.arrowedLine(annotated_frame, arrow_start, arrow_end, color, thickness)

    return annotated_frame


def select_representative_frames(labels, valid_faces, num_frames_per_speaker=2):
    """
    为每个说话人选择代表性帧

    策略：
    1. 选择人脸面积最大的帧（人脸更清晰）
    2. 尽量分散在不同时间段（避免同一场景）
    """
    # 按label分组
    speaker_faces = defaultdict(list)
    for label, face in zip(labels, valid_faces):
        if label != -1:  # 排除噪声
            speaker_faces[label].append(face)

    selected_frames = {}

    for label, faces in sorted(speaker_faces.items()):
        # 按人脸面积排序
        faces_sorted = sorted(
            faces,
            key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]),
            reverse=True
        )

        # 选择前num_frames_per_speaker个
        # 但要保证时间分散
        selected = []

        if len(faces_sorted) <= num_frames_per_speaker:
            selected = faces_sorted
        else:
            # 第一张：最大人脸
            selected.append(faces_sorted[0])

            # 第二张：时间上距离第一张最远的
            first_seg = selected[0]['segment_id']

            # 找距离最远的
            max_distance = 0
            best_face = faces_sorted[1]

            for face in faces_sorted[1:]:
                distance = abs(face['segment_id'] - first_seg)
                if distance > max_distance:
                    max_distance = distance
                    best_face = face

            selected.append(best_face)

        selected_frames[label] = selected

    return selected_frames


def prepare_annotated_images(selected_frames, output_dir="./vlm_naming_images"):
    """
    准备标注后的图片

    Returns:
        image_info: {
            speaker_id: [
                {'path': 'xxx.jpg', 'segment': 10, 'time': 30.5},
                {'path': 'yyy.jpg', 'segment': 20, 'time': 60.2}
            ]
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    image_info = {}

    for label, faces in sorted(selected_frames.items()):
        speaker_id = label + 1  # label从0开始，speaker_id从1开始
        image_paths = []

        for idx, face in enumerate(faces, 1):
            # 获取完整画面
            frame = face['frame']
            box = face['box']

            # 标注人脸
            annotated = annotate_target_face(frame, box)

            # 保存
            filename = f"speaker_{speaker_id}_frame_{idx}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, annotated)

            image_paths.append({
                'path': filepath,
                'segment_id': face['segment_id'],
                'face_area_ratio': face.get('face_area_ratio', 0)
            })

        image_info[speaker_id] = image_paths

    return image_info


def build_vlm_prompt(image_info, srt_segments, clustering_result_path):
    """
    构建VLM prompt

    包含：
    1. SRT完整对话
    2. 每个speaker出现的片段信息
    3. 要求VLM完成的任务
    """
    # 加载聚类结果
    with open(clustering_result_path, 'r', encoding='utf-8') as f:
        clustering_result = json.load(f)

    # 构建SRT对话文本（全部片段）
    srt_text = ""
    for i, seg in enumerate(srt_segments, 1):
        srt_text += f"[片段{i}] {seg['text']}\n"

    # 构建speaker信息 - 包含每个speaker出现的对话内容
    speaker_info_text = ""
    for speaker_id in sorted(image_info.keys()):
        speaker_key = f"speaker_{speaker_id}"
        if speaker_key in clustering_result:
            info = clustering_result[speaker_key]
            segments = info['segments']
            face_count = info['face_count']

            speaker_info_text += f"\nSpeaker {speaker_id}（图{(speaker_id-1)*2+1}-{speaker_id*2}）:\n"
            speaker_info_text += f"  人脸检测: 在 {len(segments)} 个片段中检测到该人脸（共{face_count}张）\n"

            # 添加该Speaker出现的对话内容（取前15个片段，避免prompt过长）
            speaker_info_text += f"  该人脸出现时的对话内容（这些很可能是TA说的话）：\n"
            display_segments = segments[:15]  # 最多显示15个片段
            for seg_idx in display_segments:
                if 1 <= seg_idx <= len(srt_segments):
                    seg_text = srt_segments[seg_idx - 1]['text']
                    speaker_info_text += f"    [片段{seg_idx}] {seg_text}\n"

            if len(segments) > 15:
                speaker_info_text += f"    ... 还有 {len(segments) - 15} 个片段未显示\n"

    # 构建prompt
    prompt = f"""【视频人物命名任务】

你将看到 {len(image_info)} 个人物的照片（每人2张），请根据以下信息为**所有{len(image_info)}个人物**命名并分析角色：

**重要：必须输出所有{len(image_info)}个Speaker的信息，不要遗漏任何一个！**

## 1. 图片说明
共 {len(image_info) * 2} 张图片，按顺序：
- 图1-2: Speaker 1 的两张画面（红框标注的是目标人物）
- 图3-4: Speaker 2 的两张画面
- 图5-6: Speaker 3 的两张画面
... 以此类推

**注意**：每张图有红框标注，红框内的人物才是我们要分析的目标，其他人物请忽略。

## 2. 每个Speaker的相关对话（重要！）
说明：每个Speaker在视频中出现时，人脸检测系统记录了他们出现的时间片段。
下面列出了每个Speaker人脸出现时对应的字幕片段内容。

**关键理解（非常重要！必读！）**：
- **技术限制说明**：我们只有人脸检测（知道谁在画面中），但**没有语音识别**（不知道谁在说话）
- 当某个Speaker的人脸在某个时间段出现时，该时间段的字幕**不一定**是这个人说的！可能的情况：
  1. 画面显示Speaker A，但字幕是Speaker B在说话（镜头切换、反应镜头）
  2. 画面中有多个人，但只有一个人在说话
  3. 画面显示听众，但字幕是演讲者在说话

- **如何判断对话归属**：
  1. **分析对话内容**：看语气、称呼、角色特征是否匹配该Speaker的外貌（性别、年龄、着装）
  2. **查看完整对话**：如果某个Speaker的所有片段对话都很连贯，像是同一个人说的，那可能是TA
  3. **交叉验证**：如果Speaker A的对话提到"我是张三"，而Speaker B的对话是"张三你好"，那张三是A不是B
  4. **外貌对比**：如果对话是"你好，经理"，结合外貌看谁更像经理（年龄、着装）

- **重要**：不要被"出现在某片段"误导，要综合判断对话内容是否真的是TA说的

{speaker_info_text}

## 3. 完整对话内容参考（共{len(srt_segments)}个片段）
{srt_text}

## 4. 任务要求

请为每个Speaker输出以下信息（JSON格式）：

{{
  "speakers": [
    {{
      "speaker_id": 1,
      "name": "根据对话内容推测的名字或称呼（如：'张经理'、'小李'、'女主角'等）",
      "role": "角色定位（主角/重要配角/次要角色）",
      "gender": "性别（男/女/不确定）",
      "appearance": {{
        "clothing": "服装描述（综合2张图）",
        "facial_features": "面部特征（脸型、五官、发型等）",
        "age_estimate": "年龄估计（如：20-30岁）",
        "distinctive_features": "显著特征（如：戴眼镜、长发、胡须等）"
      }},
      "character_analysis": {{
        "personality": "从对话推测的性格特点",
        "importance": "重要程度（基于出现次数和对话内容）",
        "relationship": "与其他角色的关系",
        "dialogue_characteristics": "对话特点（说话风格、常用词汇等）"
      }},
      "key_dialogues": "该角色的代表性对话（从片段中选3-5句）"
    }}
  ]
}}

**命名分析策略（重要！）**：

1. **理解人脸与对话的关系**：
   - 每个Speaker下列出的对话，是TA的人脸出现时的字幕
   - 大部分情况下，画面中出现某人时，正在说话的就是TA
   - 因此，这些对话内容很可能就是该Speaker说的话

2. **【核心原则】说话人 vs 被称呼者（必须严格遵守！）**：

   **第一人称自称 = 说话人本人**：
   - ✅ "我叫张三" → 说话人是张三
   - ✅ "我是经理" → 说话人是经理
   - ✅ "咱们走吧" → 说话人是团队成员

   **第二人称称呼 = 对方，NOT说话人**：
   - ❌ "张三，你好" → 张三是对方，说话人不是张三
   - ❌ "李经理，文件准备好了" → 李经理是对方，说话人不是李经理
   - ❌ "王总，会议开始了" → 王总是对方，说话人不是王总

   **第三人称提及 = 第三方，NOT说话人**：
   - ❌ "张三说得对" → 张三是第三方，说话人不是张三
   - ❌ "经理让我们等着" → 经理是第三方，说话人不是经理

   **【关键警告】**：
   - 如果某个Speaker的对话中反复出现同一个称呼（如："X经理"、"Y总"），但都是"X，你..."或"哎 X，..."的格式
   - 那这个称呼是**对方**，不是该Speaker！
   - 绝对不能把对方的名字赋给说话人！

3. **对话内容优先命名**：
   - **第一优先**：第一人称自称（如："我叫张三"、"我是经理"）
   - **第二优先**：被别人一致称呼的名字（需要交叉验证：如果多个Speaker都称呼某人为"X"，那X可能是另一个Speaker）
   - **第三优先**：基于对话内容和语气推测的角色（如：命令式语气+发出指令 → "领导"）
   - **第四优先**：基于外貌和对话风格推测（如：制服+询问登记 → "工作人员"）
   - **禁止**：把第二人称称呼当作说话人名字
   - **禁止**：仅根据外貌编造名字

4. **交叉验证防止重复命名**：
   - **检查所有Speaker**：如果Speaker A的对话是"X经理，..."，Speaker B的对话是"好，就这么办"
   - 那么"X经理"很可能是Speaker B（被称呼的人），而不是Speaker A（称呼别人的人）
   - **防止重复**：确保每个名字/称呼只分配给一个Speaker
   - **一致性检查**：如果某个称呼在多个Speaker的对话中都作为"被称呼者"出现，找出谁最像这个角色

5. **如何判断名字属于谁（实例教学）**：

   **例1：称呼判断**
   - Speaker A: "李经理，文件准备好了" → Speaker A在称呼别人"李经理"，TA不是李经理
   - Speaker B: "好的，马上开会" → Speaker B发出指令，说话像领导，可能是李经理
   - 结论：Speaker B = 李经理，Speaker A ≠ 李经理

   **例2：自称判断**
   - Speaker C: "我叫王明" → Speaker C就是王明
   - Speaker D: "王明，你好" → Speaker D在称呼王明，TA不是王明
   - 结论：Speaker C = 王明，Speaker D ≠ 王明

   **例3：职业判断**
   - Speaker E: "我负责这个项目" → Speaker E自称负责，是项目负责人
   - Speaker F: "请问需要帮忙吗" → Speaker F提供服务，可能是助理/服务人员
   - 结论：Speaker E = 项目负责人，Speaker F = 助理

6. **综合分析技巧**：
   - 先标记所有第一人称自称（"我X"、"我是X"、"我叫X"）
   - 再标记所有第二人称称呼（"X，你..."、"哎 X，..."）
   - 交叉验证：A称呼的对象可能就是B
   - 结合外貌（性别、年龄、着装）验证名字是否合理
   - 主角通常出现片段多、对话量大、被多人称呼

7. **关系分析**：
   - 看哪些Speaker的片段编号有重叠（说明同时出现在画面中）
   - 分析对话中的称呼关系（如："您"表示尊敬，"小"表示亲昵）
   - 注意对话中明确提到的关系（如："你爸爸"、"我老板"）

只输出JSON，不要其他说明。
"""

    return prompt


def encode_image_to_base64(image_path):
    """将图片编码为base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_qwen_vlm(image_paths, prompt, api_key, model="qwen-vl-max"):
    """
    调用阿里通义千问VLM API

    Args:
        image_paths: 图片路径列表（按顺序）
        prompt: 文本prompt
        api_key: 阿里云API key
        model: 模型名称 (qwen-vl-max, qwen-vl-plus)
    """
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建content：图片和文本交替
    content_items = []

    # 添加所有图片
    for img_path in image_paths:
        img_base64 = encode_image_to_base64(img_path)
        content_items.append({
            "image": f"data:image/jpeg;base64,{img_base64}"
        })

    # 添加文本prompt
    content_items.append({
        "text": prompt
    })

    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"text": "你是一个专业的视频人物分析专家，擅长根据外观特征和对话内容识别和命名人物角色。"}
                    ]
                },
                {
                    "role": "user",
                    "content": content_items
                }
            ]
        },
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 4096
        }
    }

    print(f"\n调用阿里通义千问VLM...")
    print(f"  API: {url}")
    print(f"  模型: {model}")
    print(f"  图片数量: {len(image_paths)}")
    print(f"  Prompt长度: {len(prompt)} 字符")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)

        # 获取请求ID
        request_id = response.headers.get('X-DashScope-RequestId', 'N/A')
        print(f"  Request-ID: {request_id}")
        print(f"  HTTP状态码: {response.status_code}")

        response.raise_for_status()

        result = response.json()

        if 'output' in result and 'choices' in result['output'] and len(result['output']['choices']) > 0:
            content = result['output']['choices'][0]['message']['content']
            # 提取文本内容
            if isinstance(content, list):
                text_content = ' '.join([item.get('text', '') for item in content if 'text' in item])
            else:
                text_content = content

            print(f"✓ VLM调用成功")
            print(f"  Request-ID: {request_id}")
            print(f"  返回内容长度: {len(text_content)} 字符")
            print(f"  返回内容预览: {text_content[:500]}...")  # 添加调试输出
            return {'content': text_content, 'trace_id': request_id}
        else:
            print(f"✗ VLM返回格式异常")
            print(f"  Request-ID: {request_id}")
            print(f"  返回内容: {result}")
            return None

    except requests.exceptions.Timeout:
        print(f"✗ VLM调用超时（600秒）")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"✗ VLM调用HTTP错误: {e}")
        request_id = e.response.headers.get('X-DashScope-RequestId', 'N/A') if hasattr(e, 'response') else 'N/A'
        print(f"  Request-ID: {request_id}")
        print(f"  HTTP状态码: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"  错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"✗ VLM调用失败: {str(e)}")
        import traceback
        print(f"  详细错误:\n{traceback.format_exc()}")
        return None


def call_minimax_vlm(image_paths, prompt, api_key, model="MiniMax-Text-01"):
    """
    调用MiniMax VLM API

    Args:
        image_paths: 图片路径列表（按顺序）
        prompt: 文本prompt
        api_key: MiniMax API key
        model: 模型名称
    """
    url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建content：先文本，再图片
    content_items = [
        {"type": "text", "text": prompt}
    ]

    # 添加所有图片（base64格式）
    for img_path in image_paths:
        img_base64 = encode_image_to_base64(img_path)
        content_items.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })

    messages = [
        {
            "role": "system",
            "name": "MiniMax AI",
            "content": "你是一个专业的视频人物分析专家，擅长根据外观特征和对话内容识别和命名人物角色。"
        },
        {
            "role": "user",
            "name": "用户",
            "content": content_items
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,  # 低温度，更确定的输出
        "top_p": 0.95
    }

    print(f"\n调用MiniMax VLM...")
    print(f"  API: {url}")
    print(f"  模型: {model}")
    print(f"  图片数量: {len(image_paths)}")
    print(f"  Prompt长度: {len(prompt)} 字符")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)

        # 获取并打印Trace-ID
        trace_id = response.headers.get('Trace-ID', 'N/A')
        print(f"  Trace-ID: {trace_id}")
        print(f"  HTTP状态码: {response.status_code}")

        response.raise_for_status()

        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"✓ VLM调用成功")
            print(f"  Trace-ID: {trace_id}")
            print(f"  返回内容长度: {len(content)} 字符")
            return {'content': content, 'trace_id': trace_id}
        else:
            print(f"✗ VLM返回格式异常")
            print(f"  Trace-ID: {trace_id}")
            print(f"  返回内容: {result}")
            return None

    except requests.exceptions.Timeout:
        print(f"✗ VLM调用超时（600秒）")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"✗ VLM调用HTTP错误: {e}")
        trace_id = e.response.headers.get('Trace-ID', 'N/A') if hasattr(e, 'response') else 'N/A'
        print(f"  Trace-ID: {trace_id}")
        print(f"  HTTP状态码: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"  错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"✗ VLM调用失败: {str(e)}")
        import traceback
        print(f"  详细错误:\n{traceback.format_exc()}")
        return None


def call_vlm(image_paths, prompt, api_key, provider="qwen", model=None):
    """
    统一VLM调用接口

    Args:
        image_paths: 图片路径列表
        prompt: 文本prompt
        api_key: API key
        provider: VLM提供商 ("qwen" 或 "minimax")
        model: 模型名称（可选，使用默认值）

    Returns:
        VLM结果字典或None
    """
    if provider == "qwen":
        if model is None:
            model = "qwen-vl-max"
        return call_qwen_vlm(image_paths, prompt, api_key, model)
    elif provider == "minimax":
        if model is None:
            model = "MiniMax-Text-01"
        return call_minimax_vlm(image_paths, prompt, api_key, model)
    else:
        print(f"✗ 不支持的VLM提供商: {provider}")
        return None


def main():
    """主流程"""
    import sys

    # 配置
    clustering_result_json = "./face_detection_shengyin3_tuned/clustering_result.json"
    cache_file = "./shengyin3_tune_cache.npz"
    srt_path = "shengyin3.srt"

    # 检查文件
    if not os.path.exists(clustering_result_json):
        print(f"❌ 聚类结果文件不存在: {clustering_result_json}")
        sys.exit(1)

    if not os.path.exists(cache_file):
        print(f"❌ 缓存文件不存在: {cache_file}")
        sys.exit(1)

    # 加载数据
    print("="*80)
    print("VLM人物命名 - shengying_3.mp4")
    print("="*80)

    print("\n[1/5] 加载数据")
    print("-"*80)

    # 加载聚类结果和人脸数据
    data = np.load(cache_file, allow_pickle=True)
    embeddings = data['embeddings']
    valid_faces = data['valid_faces'].tolist()
    print(f"✓ 加载了 {len(valid_faces)} 个人脸")

    # 需要重新聚类获取labels
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.28, min_samples=5, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    num_speakers = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✓ 识别出 {num_speakers} 个说话人")

    # 加载SRT
    from srt_processor import SRTProcessor
    srt_processor = SRTProcessor()
    srt_segments = srt_processor.parse_srt(srt_path)
    print(f"✓ 加载了 {len(srt_segments)} 个对话片段")

    # [2/5] 选择代表帧
    print("\n[2/5] 为每个说话人选择代表帧")
    print("-"*80)
    selected_frames = select_representative_frames(labels, valid_faces, num_frames_per_speaker=2)

    for label in sorted(selected_frames.keys()):
        frames = selected_frames[label]
        print(f"  Speaker {label+1}: 选择了 {len(frames)} 帧")
        for i, f in enumerate(frames, 1):
            print(f"    帧{i}: segment {f['segment_id']}, 人脸面积比 {f.get('face_area_ratio', 0):.3f}")

    # [3/5] 准备标注图片
    print("\n[3/5] 生成标注图片（红框标注目标人脸）")
    print("-"*80)
    image_info = prepare_annotated_images(selected_frames)

    total_images = sum(len(imgs) for imgs in image_info.values())
    print(f"✓ 生成了 {total_images} 张标注图片")

    # [4/5] 构建prompt
    print("\n[4/5] 构建VLM prompt")
    print("-"*80)
    prompt = build_vlm_prompt(image_info, srt_segments, clustering_result_json)
    print(f"✓ Prompt长度: {len(prompt)} 字符")

    # 收集所有图片路径（按顺序）
    all_image_paths = []
    for speaker_id in sorted(image_info.keys()):
        for img_info in image_info[speaker_id]:
            all_image_paths.append(img_info['path'])

    print(f"✓ 图片顺序: {len(all_image_paths)} 张")

    # [5/5] 调用VLM
    print("\n[5/5] 调用VLM")
    print("-"*80)

    # 从配置文件获取默认VLM提供商
    from config import LLM_CONFIG
    vlm_provider = LLM_CONFIG.get('default_vlm_provider', 'qwen')

    # 获取对应的API key
    if vlm_provider == "qwen":
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            api_key = LLM_CONFIG.get('qwen_api_key')
        if not api_key:
            print("❌ 未设置阿里通义千问API Key")
            print("请设置环境变量: export QWEN_API_KEY='your_api_key'")
            print("或在config.py中配置 qwen_api_key")
            sys.exit(1)
    else:  # minimax
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            api_key = LLM_CONFIG.get('minimax_api_key')
        if not api_key:
            print("❌ 未设置 MINIMAX_API_KEY")
            print("请设置环境变量: export MINIMAX_API_KEY='your_api_key'")
            print("或在config.py中配置 minimax_api_key")
            sys.exit(1)

    print(f"使用VLM提供商: {vlm_provider}")
    result = call_vlm(all_image_paths, prompt, api_key, provider=vlm_provider)

    if result:
        # 保存结果
        output_file = "./speaker_names_vlm.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['content'])

        # 保存trace_id
        trace_id_file = "./vlm_trace_id.txt"
        with open(trace_id_file, 'w', encoding='utf-8') as f:
            f.write(f"Trace-ID: {result['trace_id']}\n")

        print(f"\n✓ VLM命名结果已保存: {output_file}")
        print(f"✓ Trace-ID已保存: {trace_id_file}")
        print("\n" + "="*80)
        print("完成！")
        print("="*80)
    else:
        print("\n❌ VLM调用失败")


if __name__ == "__main__":
    main()
