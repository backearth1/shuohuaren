"""
使用 LLM 自动给 SRT 字幕分配说话人

基于：
1. 人脸聚合结果（每个说话人出现在哪些片段）
2. VLM 给的人脸命名和角色信息
3. 字幕内容和对话逻辑

使用 MiniMax LLM API
"""

import json
import requests
from typing import List, Dict, Any, Optional


def build_speaker_assignment_prompt(
    srt_segments: List[Dict],
    speaker_profiles: Dict[int, Dict],
    clustering_info: Dict[str, Any]
) -> str:
    """
    构建说话人分配的 Prompt

    Args:
        srt_segments: SRT 字幕片段列表
        speaker_profiles: 说话人档案（包含VLM命名结果）
        clustering_info: 人脸聚合信息（每个说话人出现在哪些片段）
    """

    # 1. 构建说话人信息
    speaker_info_text = "## 识别到的说话人信息\n\n"

    for speaker_id in sorted(speaker_profiles.keys()):
        profile = speaker_profiles[speaker_id]

        speaker_info_text += f"### Speaker {speaker_id}\n"
        speaker_info_text += f"- **名字/称呼**: {profile.get('name', f'Speaker {speaker_id}')}\n"
        speaker_info_text += f"- **性别**: {profile.get('gender', '未知')}\n"
        speaker_info_text += f"- **角色**: {profile.get('role', '未知')}\n"

        if 'appearance' in profile:
            appearance = profile['appearance']
            if isinstance(appearance, dict):
                speaker_info_text += f"- **外观特征**:\n"
                if 'clothing' in appearance:
                    speaker_info_text += f"  - 服装: {appearance['clothing']}\n"
                if 'age_estimate' in appearance:
                    speaker_info_text += f"  - 年龄: {appearance['age_estimate']}\n"

        if 'character_analysis' in profile:
            char_analysis = profile['character_analysis']
            if isinstance(char_analysis, dict):
                if 'personality' in char_analysis:
                    speaker_info_text += f"- **性格特点**: {char_analysis['personality']}\n"

        # 添加人脸出现的片段信息
        segments_with_face = profile.get('segments', [])
        speaker_info_text += f"- **人脸出现的片段**: {len(segments_with_face)} 个片段\n"
        speaker_info_text += f"  片段编号: {', '.join(map(str, segments_with_face[:20]))}"
        if len(segments_with_face) > 20:
            speaker_info_text += f" ... (共{len(segments_with_face)}个)"
        speaker_info_text += "\n\n"

    # 2. 构建字幕内容
    srt_text = "## 字幕内容（需要分配说话人）\n\n"
    for i, seg in enumerate(srt_segments, 1):
        text = seg.get('text', '')
        srt_text += f"{i}. {text}\n"

    # 3. 构建完整 Prompt
    prompt = f"""# 视频说话人自动分配任务

你是一个专业的视频对话分析专家，需要根据人脸识别结果和对话内容，为每句字幕分配正确的说话人。

{speaker_info_text}

{srt_text}

## 任务要求

请为每句字幕（1-{len(srt_segments)}）分配说话人ID（1-{len(speaker_profiles)}）。

### 分配原则（按优先级）：

1. **人脸出现优先**：
   - 如果某个片段只有一个说话人的人脸出现，优先分配给该说话人
   - 如果多个说话人都出现，需要结合对话内容判断

2. **对话连贯性**：
   - 连续的对话通常是一问一答，说话人会交替
   - 同一个话题的延续通常是同一个说话人

3. **语气和内容匹配**：
   - 根据说话人的角色、性格特点判断
   - 例如：领导发出指令，下属汇报工作

4. **称呼关系**：
   - 注意对话中的称呼（如"经理"、"小李"等）
   - 如果Speaker A的人脸片段中字幕是"经理，..."，那说话人可能不是经理
   - 交叉验证：被称呼的人可能是另一个说话人

5. **特殊情况处理**：
   - 如果实在无法判断，可以标记为"uncertain"
   - 旁白、画外音等可以标记为"narrator"

### 输出格式（JSON）：

```json
{{
  "assignments": [
    {{
      "segment_id": 1,
      "speaker_id": 1,
      "confidence": "high",
      "reason": "片段1中只有Speaker 1的人脸出现，且内容符合其角色特征"
    }},
    {{
      "segment_id": 2,
      "speaker_id": 2,
      "confidence": "medium",
      "reason": "根据对话逻辑，这是对片段1的回应，应该是Speaker 2"
    }}
  ],
  "summary": {{
    "total_segments": {len(srt_segments)},
    "high_confidence": 0,
    "medium_confidence": 0,
    "low_confidence": 0,
    "uncertain": 0
  }}
}}
```

### confidence 等级说明：
- **high**: 人脸唯一出现 + 内容匹配，或对话逻辑非常明确
- **medium**: 有一定依据但不是100%确定
- **low**: 主要靠推测
- **uncertain**: 无法判断

**重要**：只输出 JSON，不要有任何其他说明文字。
"""

    return prompt


def call_minimax_llm(
    prompt: str,
    api_key: str,
    model: str = "abab6.5s-chat",
    temperature: float = 0.1
) -> Optional[Dict]:
    """
    调用 MiniMax LLM API

    Args:
        prompt: 文本提示
        api_key: MiniMax API key
        model: 模型名称（推荐 abab6.5s-chat 用于结构化输出）
        temperature: 温度参数（0-1，越低越确定）

    Returns:
        {'content': '...', 'trace_id': '...'}
    """
    url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的视频对话分析专家，擅长根据上下文、角色特征和对话逻辑分配说话人。你总是输出严格的JSON格式。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 8000  # 需要足够的token来处理长对话
    }

    print(f"\n调用 MiniMax LLM 进行说话人分配...")
    print(f"  API: {url}")
    print(f"  模型: {model}")
    print(f"  Prompt长度: {len(prompt)} 字符")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)

        trace_id = response.headers.get('Trace-ID', 'N/A')
        print(f"  Trace-ID: {trace_id}")
        print(f"  HTTP状态码: {response.status_code}")

        response.raise_for_status()

        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"✓ LLM调用成功")
            print(f"  返回内容长度: {len(content)} 字符")
            print(f"  返回内容预览: {content[:500]}...")
            return {'content': content, 'trace_id': trace_id}
        else:
            print(f"✗ LLM返回格式异常")
            print(f"  返回内容: {result}")
            return None

    except requests.exceptions.Timeout:
        print(f"✗ LLM调用超时（600秒）")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"✗ LLM调用HTTP错误: {e}")
        trace_id = e.response.headers.get('Trace-ID', 'N/A') if hasattr(e, 'response') else 'N/A'
        print(f"  Trace-ID: {trace_id}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"  错误详情: {e.response.text}")
        return None
    except Exception as e:
        print(f"✗ LLM调用失败: {str(e)}")
        import traceback
        print(f"  详细错误:\n{traceback.format_exc()}")
        return None


def assign_speakers_with_llm(
    srt_segments: List[Dict],
    speaker_profiles: Dict[int, Dict],
    api_key: str,
    model: str = "abab6.5s-chat"
) -> Optional[Dict]:
    """
    使用 LLM 自动分配说话人

    Args:
        srt_segments: SRT 字幕片段
        speaker_profiles: 说话人档案（包含VLM命名、人脸出现片段等）
        api_key: MiniMax API key
        model: LLM 模型名称

    Returns:
        分配结果的字典，包含每个片段的说话人ID
    """

    # 构建 clustering_info（从 speaker_profiles 中提取）
    clustering_info = {}
    for speaker_id, profile in speaker_profiles.items():
        clustering_info[f"speaker_{speaker_id}"] = {
            'segments': profile.get('segments', []),
            'face_count': profile.get('face_count', 0)
        }

    # 构建 Prompt
    prompt = build_speaker_assignment_prompt(
        srt_segments=srt_segments,
        speaker_profiles=speaker_profiles,
        clustering_info=clustering_info
    )

    # 调用 LLM
    llm_result = call_minimax_llm(
        prompt=prompt,
        api_key=api_key,
        model=model,
        temperature=0.1  # 低温度保证确定性
    )

    if not llm_result:
        return None

    # 解析 LLM 输出
    try:
        llm_output = llm_result['content']

        # 尝试提取 JSON（可能有额外的文字说明）
        import re
        json_match = re.search(r'\{[\s\S]*\}', llm_output)
        if json_match:
            json_str = json_match.group(0)
            assignment_result = json.loads(json_str)
            print(f"✓ LLM输出解析成功")
            print(f"  分配了 {len(assignment_result.get('assignments', []))} 个片段")
            return assignment_result
        else:
            # 直接尝试解析
            assignment_result = json.loads(llm_output)
            print(f"✓ LLM输出解析成功")
            return assignment_result

    except json.JSONDecodeError as e:
        print(f"✗ LLM输出解析失败: {str(e)}")
        print(f"  LLM输出内容: {llm_output[:2000]}")
        return None
    except Exception as e:
        print(f"✗ 处理LLM输出时出错: {str(e)}")
        return None


def apply_speaker_assignments_to_srt(
    srt_segments: List[Dict],
    assignments: Dict
) -> List[Dict]:
    """
    将说话人分配结果应用到 SRT 字幕

    Args:
        srt_segments: 原始 SRT 片段
        assignments: LLM 分配结果

    Returns:
        添加了说话人信息的 SRT 片段
    """
    # 创建 segment_id 到 speaker_id 的映射
    segment_to_speaker = {}

    if 'assignments' in assignments:
        for item in assignments['assignments']:
            seg_id = item.get('segment_id')
            speaker_id = item.get('speaker_id')
            confidence = item.get('confidence', 'unknown')

            if seg_id and speaker_id:
                segment_to_speaker[seg_id] = {
                    'speaker_id': speaker_id,
                    'confidence': confidence
                }

    # 应用到 SRT 片段
    updated_segments = []
    for i, seg in enumerate(srt_segments, 1):
        updated_seg = seg.copy()

        if i in segment_to_speaker:
            updated_seg['speaker_id'] = segment_to_speaker[i]['speaker_id']
            updated_seg['speaker_confidence'] = segment_to_speaker[i]['confidence']
        else:
            updated_seg['speaker_id'] = None
            updated_seg['speaker_confidence'] = 'unknown'

        updated_segments.append(updated_seg)

    return updated_segments


if __name__ == "__main__":
    # 测试示例
    import os

    # 模拟数据
    test_srt_segments = [
        {'text': '经理，这是本月的销售报告。'},
        {'text': '好的，让我看看。'},
        {'text': '我们这个月增长了15%。'},
        {'text': '不错，继续保持。'}
    ]

    test_speaker_profiles = {
        1: {
            'name': '张经理',
            'gender': '男',
            'role': '主管',
            'segments': [2, 4, 7, 9],
            'face_count': 15,
            'appearance': {
                'clothing': '深色西装',
                'age_estimate': '40-50岁'
            }
        },
        2: {
            'name': '小李',
            'gender': '男',
            'role': '员工',
            'segments': [1, 3, 5, 8],
            'face_count': 12,
            'appearance': {
                'clothing': '白色衬衫',
                'age_estimate': '25-30岁'
            }
        }
    }

    api_key = os.getenv("MINIMAX_API_KEY")
    if api_key:
        result = assign_speakers_with_llm(
            srt_segments=test_srt_segments,
            speaker_profiles=test_speaker_profiles,
            api_key=api_key
        )

        if result:
            print("\n=== 分配结果 ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("请设置 MINIMAX_API_KEY 环境变量")
