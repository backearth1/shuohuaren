"""
核心处理Pipeline
集成人脸检测、聚类、VLM命名
"""

import os
import sys
import numpy as np
import json
from typing import Callable, Optional

# 添加父目录到路径，以便导入已有模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from sklearn.cluster import DBSCAN
from srt_processor import SRTProcessor
from face_detection_poc import (
    extract_frames_for_face_detection,
    detect_faces_with_mtcnn,
    extract_face_embeddings
)


def process_video_pipeline(
    video_path: str,
    srt_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    eps: float = 0.28,
    min_samples: int = 5,
    api_key: Optional[str] = None,
    vlm_provider: str = "qwen"
):
    """
    完整的视频处理Pipeline

    Args:
        video_path: 视频文件路径
        srt_path: SRT文件路径
        output_dir: 输出目录
        progress_callback: 进度回调函数
        eps: DBSCAN参数
        min_samples: DBSCAN参数
        api_key: VLM API Key（用于VLM命名）
        vlm_provider: VLM提供商（qwen或minimax）

    Returns:
        result: {
            "num_speakers": 10,
            "speakers": [
                {
                    "speaker_id": 1,
                    "name": "阮娇",
                    "face_count": 225,
                    "face_images": ["speaker_1_face_1.jpg", ...],
                    "appearance": {...},
                    ...
                }
            ]
        }
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def update_progress(progress: int, message: str):
        if progress_callback:
            progress_callback(progress, message)
        print(f"[{progress}%] {message}")

    try:
        # Step 1: 解析SRT
        update_progress(10, "解析SRT文件...")
        srt_processor = SRTProcessor()
        srt_segments = srt_processor.parse_srt(srt_path)
        print(f"  解析到 {len(srt_segments)} 个字幕片段")

        # Step 2: 抽取帧
        update_progress(20, "抽取视频关键帧...")
        frames_data = extract_frames_for_face_detection(
            video_path,
            srt_segments,
            max_frames_per_segment=5
        )
        print(f"  抽取了 {len(frames_data)} 个片段的帧")

        # Step 3: 人脸检测
        update_progress(30, "检测人脸并进行质量过滤...")
        all_faces = detect_faces_with_mtcnn(frames_data, device=device)
        print(f"  检测到 {len(all_faces)} 张高质量人脸")

        if len(all_faces) == 0:
            raise Exception("未检测到人脸，请检查视频内容")

        # Step 4: 特征提取
        update_progress(45, "提取人脸特征向量...")
        embeddings, valid_faces = extract_face_embeddings(all_faces, device=device)
        print(f"  提取了 {len(embeddings)} 个特征向量")

        # Step 5: 聚类
        update_progress(60, "使用DBSCAN聚类识别说话人...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        num_speakers = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  识别出 {num_speakers} 个说话人")

        # Step 6: 按人脸数量排序，确定speaker_id
        update_progress(70, "统计聚类结果...")
        import cv2
        from collections import defaultdict

        clusters = defaultdict(list)
        for label, face in zip(labels, valid_faces):
            if label != -1:
                clusters[label].append(face)

        # 按人脸数量排序（这个顺序决定了speaker_id）
        sorted_labels = sorted(clusters.keys(), key=lambda x: len(clusters[x]), reverse=True)

        # 创建label到speaker_id的映射
        label_to_speaker_id = {label: rank for rank, label in enumerate(sorted_labels, 1)}

        # Step 7: 保存代表性完整画面（用于web展示，带人脸框标注）
        update_progress(72, "保存代表性完整画面...")
        faces_dir = os.path.join(output_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)

        # 为每个speaker选择2张代表性图片，保存完整画面（带人脸框）
        # 关键：使用speaker_id作为文件名，而不是DBSCAN的label
        all_face_images = {}  # {speaker_id: [image_paths]}
        for label, faces in clusters.items():
            speaker_id = label_to_speaker_id[label]  # 使用映射后的speaker_id
            face_paths = []

            # 选择2张代表图片（按清晰度排序）
            sorted_faces = sorted(faces, key=lambda f: f.get('sharpness', 0), reverse=True)
            selected_faces = sorted_faces[:2]  # 只取前2张

            for i, face in enumerate(selected_faces, 1):
                # 获取原始帧
                frame = face['frame'].copy()  # 复制避免修改原始数据
                if frame is None:
                    continue

                # 在完整画面上画人脸框
                x1, y1, x2, y2 = [int(b) for b in face['box']]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # 画红色矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # 保存完整画面 - 使用speaker_id而不是label
                img_filename = f"speaker_{speaker_id}_fullframe_{i}.jpg"
                img_path = os.path.join(faces_dir, img_filename)
                cv2.imwrite(img_path, frame)
                face_paths.append(f"faces/{img_filename}")

            all_face_images[speaker_id] = face_paths

        # Step 8: 统计聚类结果
        update_progress(75, "构建speaker profiles...")
        speaker_profiles = {}

        # 为VLM准备代表图片（选2张）- 使用排序后的顺序
        from name_speakers_with_vlm import (
            select_representative_frames,
            prepare_annotated_images
        )
        # 只为排序后的labels选择代表帧
        sorted_selected_frames = {}
        selected_frames_all = select_representative_frames(labels, valid_faces, num_frames_per_speaker=2)
        # 按sorted_labels的顺序重新组织
        for rank, label in enumerate(sorted_labels, 1):
            if label in selected_frames_all:
                sorted_selected_frames[rank-1] = selected_frames_all[label]  # rank-1是因为prepare_annotated_images会+1

        image_info = prepare_annotated_images(sorted_selected_frames, output_dir=os.path.join(output_dir, "faces_vlm"))

        for rank, label in enumerate(sorted_labels, 1):
            faces = clusters[label]
            segments = sorted(list(set([f['segment_id'] for f in faces])))

            speaker_profiles[rank] = {
                'speaker_id': rank,
                'original_label': int(label),
                'face_count': len(faces),
                'segment_count': len(segments),
                'segments': segments,
                'face_images': all_face_images.get(rank, [])  # 使用speaker_id获取对应的图片
            }

        # Step 8: VLM命名（可选）
        vlm_result = None
        if api_key:
            update_progress(80, "调用VLM进行人物命名...")
            try:
                from name_speakers_with_vlm import (
                    build_vlm_prompt,
                    call_vlm
                )

                # 收集所有图片路径（VLM代表图片在faces_vlm目录）
                all_image_paths = []
                for speaker_id in sorted(image_info.keys()):
                    for img_info in image_info[speaker_id]:
                        # img_info['path']已经是完整的绝对路径
                        all_image_paths.append(img_info['path'])

                # 构建prompt
                # 需要先构建聚类结果JSON
                temp_clustering_result = {
                    f"speaker_{sp['speaker_id']}": {
                        'face_count': sp['face_count'],
                        'segments': sp['segments']
                    }
                    for sp in speaker_profiles.values()
                }

                temp_json_path = os.path.join(output_dir, "temp_clustering.json")
                with open(temp_json_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_clustering_result, f, ensure_ascii=False, indent=2)

                prompt = build_vlm_prompt(image_info, srt_segments, temp_json_path)

                # 调用VLM（使用统一接口，根据vlm_provider选择）
                vlm_result_dict = call_vlm(all_image_paths, prompt, api_key, provider=vlm_provider)

                if vlm_result_dict:
                    vlm_output = vlm_result_dict.get('content')
                    trace_id = vlm_result_dict.get('trace_id', 'N/A')

                    # 保存trace_id到文件
                    trace_id_file = os.path.join(output_dir, 'vlm_trace_id.txt')
                    with open(trace_id_file, 'w', encoding='utf-8') as f:
                        f.write(f"Trace-ID: {trace_id}\n")
                    print(f"  ✓ Trace-ID已保存到: {trace_id_file}")

                    # 解析VLM输出
                    try:
                        print(f"  VLM输出预览: {vlm_output[:1000]}...")  # 调试输出
                        vlm_result = json.loads(vlm_output)

                        # 合并VLM结果到speaker_profiles
                        if 'speakers' in vlm_result:
                            for vlm_speaker in vlm_result['speakers']:
                                speaker_id = vlm_speaker.get('speaker_id')
                                if speaker_id in speaker_profiles:
                                    speaker_profiles[speaker_id].update({
                                        'name': vlm_speaker.get('name', f'Speaker {speaker_id}'),
                                        'role': vlm_speaker.get('role'),
                                        'gender': vlm_speaker.get('gender'),
                                        'appearance': vlm_speaker.get('appearance'),
                                        'character_analysis': vlm_speaker.get('character_analysis')
                                    })
                    except json.JSONDecodeError as e:
                        print(f"  VLM输出解析失败: {str(e)}")
                        print(f"  VLM输出内容: {vlm_output[:2000]}")  # 显示更多内容以便调试

            except Exception as e:
                print(f"  VLM命名失败: {str(e)}")
                # 继续处理，不中断

        # 如果没有VLM结果，使用默认命名
        for speaker_id, profile in speaker_profiles.items():
            if 'name' not in profile:
                profile['name'] = f'Speaker {speaker_id}'

        # Step 9: 使用 LLM 自动分配说话人
        llm_assignment_result = None
        updated_srt_segments = srt_segments  # 默认使用原始字幕

        if api_key:
            try:
                update_progress(90, "使用LLM自动分配说话人...")

                from llm_speaker_assignment import assign_speakers_with_llm, apply_speaker_assignments_to_srt

                # 调用 LLM 进行说话人分配
                llm_assignment_result = assign_speakers_with_llm(
                    srt_segments=srt_segments,
                    speaker_profiles=speaker_profiles,
                    api_key=api_key,
                    model="abab6.5s-chat"
                )

                if llm_assignment_result:
                    # 将分配结果应用到字幕
                    updated_srt_segments = apply_speaker_assignments_to_srt(
                        srt_segments=srt_segments,
                        assignments=llm_assignment_result
                    )
                    print(f"  ✓ LLM说话人分配完成")

                    # 保存带说话人信息的字幕
                    assigned_srt_path = os.path.join(output_dir, 'subtitle_with_speakers.json')
                    with open(assigned_srt_path, 'w', encoding='utf-8') as f:
                        json.dump(updated_srt_segments, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ 已保存带说话人信息的字幕: {assigned_srt_path}")

            except Exception as e:
                print(f"  LLM说话人分配失败: {str(e)}")
                # 继续处理，不中断

        update_progress(95, "生成最终结果...")

        # 构建最终结果
        result = {
            'num_speakers': num_speakers,
            'speakers': list(speaker_profiles.values()),
            'video_info': {
                'total_segments': len(srt_segments),
                'total_faces_detected': len(all_faces),
                'total_faces_valid': len(valid_faces)
            },
            'clustering_params': {
                'eps': eps,
                'min_samples': min_samples
            },
            'vlm_enabled': api_key is not None,
            'vlm_result': vlm_result,
            'llm_assignment': llm_assignment_result,
            'srt_with_speakers': updated_srt_segments
        }

        update_progress(100, "处理完成！")

        return result

    except Exception as e:
        update_progress(-1, f"处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 测试
    import os

    api_key = os.getenv("MINIMAX_API_KEY")

    result = process_video_pipeline(
        video_path="../../shengying_3.mp4",
        srt_path="../../shengyin3.srt",
        output_dir="./test_output",
        api_key=api_key
    )

    print("\n处理结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
