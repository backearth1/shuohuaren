"""
人脸检测POC - 验证聚类效果

测试目标：
1. 检测视频中的所有人脸
2. 提取人脸特征向量
3. 使用DBSCAN聚类
4. 验证能否正确识别不同的说话人
"""

import cv2
import numpy as np
import os
from collections import defaultdict
import json

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("⚠️  facenet-pytorch未安装")
    print("请运行: pip install facenet-pytorch")

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan
from srt_processor import SRTProcessor


def extract_frames_for_face_detection(video_path, srt_segments, max_frames_per_segment=5):
    """
    为每个segment抽取帧用于人脸检测

    Args:
        max_frames_per_segment: 每个segment最多抽取几帧（默认5帧）
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_data = []

    for seg in srt_segments:
        start_time = seg['start_time']
        end_time = seg['end_time']
        duration = end_time - start_time

        if max_frames_per_segment == 5:
            # 6等分，取中间5个时间点（排除开头和结尾）
            step = duration / 6
            time_points = [
                start_time + step,      # 1/6处
                start_time + 2 * step,  # 2/6处
                start_time + 3 * step,  # 3/6处（中点）
                start_time + 4 * step,  # 4/6处
                start_time + 5 * step   # 5/6处
            ]
        elif max_frames_per_segment == 3:
            # 4等分，取中间3个时间点
            quarter = duration / 4
            time_points = [
                start_time + quarter,
                start_time + 2 * quarter,
                start_time + 3 * quarter
            ]
        else:
            # 只取中点
            time_points = [(start_time + end_time) / 2]

        seg_frames = []
        for time_point in time_points:
            frame_num = int(time_point * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                seg_frames.append({
                    'time': time_point,
                    'frame': frame
                })

        if seg_frames:
            frames_data.append({
                'segment_id': seg['index'],
                'frames': seg_frames
            })

    cap.release()
    return frames_data


def detect_faces_with_mtcnn(frames_data, device='cuda', min_face_size=40):
    """
    使用MTCNN检测所有人脸，并进行质量过滤

    Args:
        min_face_size: 最小人脸尺寸（像素），过滤掉太小的人脸
    """
    if not FACENET_AVAILABLE:
        print("❌ 需要安装 facenet-pytorch")
        return []

    print(f"\n使用设备: {device}")
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False, min_face_size=min_face_size)

    all_faces = []
    filtered_count = {'low_confidence': 0, 'too_small': 0, 'side_face': 0, 'blurry': 0}
    total_frames = sum(len(seg['frames']) for seg in frames_data)

    print(f"开始检测人脸（共{total_frames}帧）...")

    for seg_data in frames_data:
        seg_id = seg_data['segment_id']

        for frame_info in seg_data['frames']:
            frame = frame_info['frame']
            frame_height, frame_width = frame.shape[:2]

            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测人脸（同时获取关键点用于判断是否侧脸）
            boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    # 1. 置信度过滤
                    if prob < 0.95:  # 提高置信度阈值，只保留高质量检测
                        filtered_count['low_confidence'] += 1
                        continue

                    # 2. 人脸尺寸过滤（排除远景小人脸）
                    x1, y1, x2, y2 = box
                    face_width = x2 - x1
                    face_height = y2 - y1

                    # 人脸占画面比例太小（可能是背景群众）
                    face_area_ratio = (face_width * face_height) / (frame_width * frame_height)
                    if face_area_ratio < 0.015:  # 提高到1.5%，过滤远景小人脸
                        filtered_count['too_small'] += 1
                        continue

                    # 3. 侧脸过滤（通过关键点判断）- 影视剧规律：有台词必有正脸
                    if landmarks is not None and i < len(landmarks):
                        landmark = landmarks[i]
                        if landmark is not None:
                            # 计算左右眼到鼻子的距离，判断是否侧脸
                            left_eye = landmark[0]   # 左眼坐标
                            right_eye = landmark[1]  # 右眼坐标
                            nose = landmark[2]       # 鼻子坐标

                            left_to_nose = abs(left_eye[0] - nose[0])
                            right_to_nose = abs(right_eye[0] - nose[0])

                            # 如果一只眼睛到鼻子的距离是另一只的2倍以上，判定为侧脸
                            if left_to_nose > 0 and right_to_nose > 0:
                                ratio = max(left_to_nose, right_to_nose) / min(left_to_nose, right_to_nose)
                                if ratio > 2.0:  # 严格侧脸过滤，只保留正脸
                                    filtered_count['side_face'] += 1
                                    continue

                    # 4. 模糊度过滤（使用Laplacian方差）- 影视剧规律：主角镜头必然清晰
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    face_img = frame[y1:y2, x1:x2]

                    # 转换为灰度图
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    # 计算Laplacian方差（值越大越清晰）
                    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

                    # 模糊阈值：提高到100，只保留清晰人脸
                    if laplacian_var < 100:
                        filtered_count['blurry'] += 1
                        continue

                    # 通过所有过滤条件
                    all_faces.append({
                        'segment_id': seg_id,
                        'time': frame_info['time'],
                        'frame': frame,
                        'box': box,
                        'confidence': prob,
                        'face_area_ratio': face_area_ratio,
                        'sharpness': laplacian_var  # 清晰度指标
                    })

    print(f"✓ 检测到 {len(all_faces)} 张高质量人脸")
    print(f"  过滤统计:")
    print(f"    低置信度: {filtered_count['low_confidence']} 张")
    print(f"    人脸太小: {filtered_count['too_small']} 张")
    print(f"    侧脸: {filtered_count['side_face']} 张")
    print(f"    模糊: {filtered_count['blurry']} 张")

    return all_faces


def extract_face_embeddings(all_faces, device='cuda'):
    """提取人脸特征向量"""
    if not FACENET_AVAILABLE:
        return None

    print(f"\n提取人脸特征向量...")

    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings = []

    for i, face in enumerate(all_faces):
        # 裁剪人脸
        box = face['box']
        frame = face['frame']

        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            embeddings.append(None)
            continue

        # 调整大小到160x160
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (160, 160))

        # 转换为tensor
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        # 提取特征
        with torch.no_grad():
            embedding = facenet(face_tensor)

        embeddings.append(embedding.cpu().numpy()[0])

        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(all_faces)}")

    # 过滤掉None
    valid_embeddings = []
    valid_faces = []
    for emb, face in zip(embeddings, all_faces):
        if emb is not None:
            valid_embeddings.append(emb)
            valid_faces.append(face)

    embeddings_array = np.array(valid_embeddings)
    print(f"✓ 提取了 {len(embeddings_array)} 个有效特征向量 (维度: {embeddings_array.shape[1]})")

    return embeddings_array, valid_faces


def cluster_faces_hdbscan(embeddings, min_cluster_size=5, min_samples=2):
    """
    使用HDBSCAN自动聚类人脸（推荐方法）

    Args:
        min_cluster_size: 最小聚类大小（说话人至少出现多少次才算有效）
        min_samples: HDBSCAN的min_samples参数
    """
    print(f"\n使用HDBSCAN自动聚类...")
    print(f"  参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # HDBSCAN对cosine支持不好，用euclidean
        cluster_selection_method='eom'  # excess of mass方法
    )

    labels = clusterer.fit_predict(embeddings)

    # 统计聚类结果
    unique_labels = set(labels)
    num_clusters = len(unique_labels - {-1})  # 排除噪声(-1)
    num_noise = list(labels).count(-1)

    print(f"✓ 聚类完成:")
    print(f"  识别出 {num_clusters} 个说话人")
    print(f"  噪声点: {num_noise} 个")

    # 统计每个聚类的大小
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1

    print(f"\n  聚类分布:")
    for label in sorted(cluster_sizes.keys()):
        if label == -1:
            print(f"    噪声: {cluster_sizes[label]} 张人脸")
        else:
            print(f"    Speaker {label + 1}: {cluster_sizes[label]} 张人脸")

    return labels, num_clusters, clusterer


def cluster_faces_dbscan_auto(embeddings):
    """
    使用DBSCAN + 轮廓系数自动选择最优eps（降级方法）
    """
    print(f"\n使用DBSCAN + 轮廓系数自动选择eps...")

    eps_values = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_eps = None
    best_score = -1
    best_labels = None
    best_num_clusters = 0

    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 至少需要2个聚类才能计算轮廓系数
        if num_clusters >= 2:
            # 只用非噪声点计算轮廓系数
            non_noise_mask = labels != -1
            if non_noise_mask.sum() >= 2:
                score = silhouette_score(
                    embeddings[non_noise_mask],
                    labels[non_noise_mask],
                    metric='cosine'
                )

                print(f"  eps={eps}: {num_clusters}个聚类, 轮廓系数={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_labels = labels
                    best_num_clusters = num_clusters

    if best_eps is None:
        # 降级到固定eps
        print(f"  ⚠️  无法通过轮廓系数选择，使用默认eps=0.3")
        best_eps = 0.3
        clustering = DBSCAN(eps=best_eps, min_samples=2, metric='cosine')
        best_labels = clustering.fit_predict(embeddings)
        best_num_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)

    print(f"\n✓ 选择最优参数: eps={best_eps}, 轮廓系数={best_score:.3f}")
    print(f"  识别出 {best_num_clusters} 个说话人")

    return best_labels, best_num_clusters


def cluster_faces(embeddings, eps=0.5, min_samples=3):
    """使用DBSCAN聚类人脸（保留用于向后兼容）"""
    print(f"\n开始聚类（eps={eps}, min_samples={min_samples}）...")

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    # 统计聚类结果
    unique_labels = set(labels)
    num_clusters = len(unique_labels - {-1})  # 排除噪声(-1)
    num_noise = list(labels).count(-1)

    print(f"✓ 聚类完成:")
    print(f"  识别出 {num_clusters} 个说话人")
    print(f"  噪声点: {num_noise} 个")

    # 统计每个聚类的大小
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1

    print(f"\n  聚类分布:")
    for label in sorted(cluster_sizes.keys()):
        if label == -1:
            print(f"    噪声: {cluster_sizes[label]} 张人脸")
        else:
            print(f"    Speaker {label + 1}: {cluster_sizes[label]} 张人脸")

    return labels, num_clusters


def validate_clustering_result(labels, valid_faces):
    """
    验证聚类结果的合理性

    Returns:
        is_valid: bool - 是否合理
        reason: str - 原因说明
    """
    clusters = defaultdict(list)
    for label, face in zip(labels, valid_faces):
        if label != -1:
            clusters[label].append(face)

    num_clusters = len(clusters)
    num_noise = list(labels).count(-1)

    # 规则1: 说话人数量应该在合理范围内（2-10人）
    if num_clusters < 2:
        return False, f"说话人数量过少 ({num_clusters}个)，可能欠聚类"
    if num_clusters > 10:
        return False, f"说话人数量过多 ({num_clusters}个)，可能过度聚类"

    # 规则2: 噪声点不应该太多（不超过总数的30%）
    noise_ratio = num_noise / len(labels) if len(labels) > 0 else 0
    if noise_ratio > 0.3:
        return False, f"噪声点过多 ({noise_ratio*100:.1f}%)，聚类质量较差"

    # 规则3: 统计主要角色和次要角色
    major_speakers = sum(1 for faces in clusters.values() if len(faces) >= 10)
    minor_speakers = sum(1 for faces in clusters.values() if len(faces) < 10)

    # 至少应该有1个主要角色
    if major_speakers == 0:
        return False, f"没有检测到主要角色（出现>=10次的说话人）"

    return True, f"聚类合理：{num_clusters}个说话人（{major_speakers}主要+{minor_speakers}次要），噪声{noise_ratio*100:.1f}%"


def analyze_clustering_results(labels, valid_faces, output_dir):
    """分析聚类结果并保存"""
    os.makedirs(output_dir, exist_ok=True)

    # 按聚类分组
    clusters = defaultdict(list)
    for label, face in zip(labels, valid_faces):
        clusters[label].append(face)

    # 为每个聚类生成报告
    speaker_profiles = {}

    for label in sorted(clusters.keys()):
        if label == -1:
            continue  # 跳过噪声

        faces = clusters[label]
        speaker_id = f"speaker_{label + 1}"

        # 统计出现的片段
        segments = sorted(list(set([face['segment_id'] for face in faces])))

        # 找到最大的人脸作为代表
        best_face = max(faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))

        # 保存代表人脸
        box = best_face['box']
        frame = best_face['frame']
        x1, y1, x2, y2 = [int(b) for b in box]
        face_img = frame[y1:y2, x1:x2]

        face_path = os.path.join(output_dir, f"{speaker_id}_sample.jpg")
        cv2.imwrite(face_path, face_img)

        # 保存前3张人脸
        for i, face in enumerate(faces[:3]):
            box = face['box']
            frame = face['frame']
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = frame[y1:y2, x1:x2]

            multi_face_path = os.path.join(output_dir, f"{speaker_id}_face_{i+1}.jpg")
            cv2.imwrite(multi_face_path, face_img)

        speaker_profiles[speaker_id] = {
            'label': int(label),
            'face_count': len(faces),
            'segments': segments,
            'sample_face_path': face_path
        }

        print(f"\n{speaker_id}:")
        print(f"  检测到的人脸数: {len(faces)}")
        print(f"  出现的片段: {segments}")
        print(f"  代表人脸: {face_path}")

    # 保存聚类结果
    result_path = os.path.join(output_dir, "clustering_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_profiles, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 聚类结果已保存: {result_path}")

    return speaker_profiles


def main():
    """主流程"""
    print("="*80)
    print("人脸检测POC - 验证聚类效果")
    print("="*80)

    # 检查依赖
    if not FACENET_AVAILABLE:
        print("\n❌ 缺少必要依赖，请先安装:")
        print("   pip install facenet-pytorch")
        return

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  未检测到CUDA，将使用CPU（速度较慢）")

    # 配置
    video_path = "shortmovie.mp4"
    srt_path = "test2.srt"
    output_dir = "./face_detection_poc_results"

    # 1. 解析SRT
    print(f"\n[1/5] 解析SRT文件")
    print("-"*80)
    srt_processor = SRTProcessor()
    srt_segments = srt_processor.parse_srt(srt_path)
    print(f"✓ 解析到 {len(srt_segments)} 个字幕片段")

    # 2. 抽取帧
    print(f"\n[2/5] 抽取关键帧")
    print("-"*80)
    frames_data = extract_frames_for_face_detection(
        video_path,
        srt_segments,
        max_frames_per_segment=5  # 每个segment抽5帧
    )
    total_frames = sum(len(seg['frames']) for seg in frames_data)
    print(f"✓ 抽取了 {len(frames_data)} 个片段的 {total_frames} 帧")

    # 3. 人脸检测
    print(f"\n[3/5] 人脸检测")
    print("-"*80)
    all_faces = detect_faces_with_mtcnn(frames_data, device=device)

    if len(all_faces) == 0:
        print("❌ 未检测到人脸，POC结束")
        return

    # 4. 特征提取
    print(f"\n[4/5] 特征提取")
    print("-"*80)
    embeddings, valid_faces = extract_face_embeddings(all_faces, device=device)

    # 5. 聚类（使用HDBSCAN自动确定说话人数）
    print(f"\n[5/5] 人脸聚类（自动确定说话人数）")
    print("-"*80)

    # 方法1: HDBSCAN（推荐）
    print(f"\n【方法1】HDBSCAN自动聚类")
    labels_hdbscan, num_clusters_hdbscan, clusterer = cluster_faces_hdbscan(
        embeddings,
        min_cluster_size=5,  # 至少出现5次才算有效说话人
        min_samples=2
    )

    # 验证HDBSCAN结果
    is_valid, reason = validate_clustering_result(labels_hdbscan, valid_faces)
    print(f"\n验证结果: {'✓ ' + reason if is_valid else '✗ ' + reason}")

    final_labels = None
    final_method = None

    if is_valid:
        # HDBSCAN结果合理，直接使用
        final_labels = labels_hdbscan
        final_method = "hdbscan"
        print(f"\n✓ 使用HDBSCAN结果")
    else:
        # HDBSCAN结果不合理，降级到DBSCAN自动选择
        print(f"\n⚠️  HDBSCAN结果不理想，降级到DBSCAN自动选择...")
        print(f"\n【方法2】DBSCAN + 轮廓系数自动选择")

        labels_dbscan, num_clusters_dbscan = cluster_faces_dbscan_auto(embeddings)

        # 验证DBSCAN结果
        is_valid_dbscan, reason_dbscan = validate_clustering_result(labels_dbscan, valid_faces)
        print(f"\n验证结果: {'✓ ' + reason_dbscan if is_valid_dbscan else '✗ ' + reason_dbscan}")

        if is_valid_dbscan:
            final_labels = labels_dbscan
            final_method = "dbscan_auto"
            print(f"\n✓ 使用DBSCAN自动选择结果")
        else:
            # 两种方法都不理想，选择聚类数更多的（宁可多也不能少）
            if num_clusters_hdbscan >= num_clusters_dbscan:
                final_labels = labels_hdbscan
                final_method = "hdbscan_fallback"
                print(f"\n⚠️  两种方法都不理想，选择HDBSCAN结果（聚类数更多）")
            else:
                final_labels = labels_dbscan
                final_method = "dbscan_fallback"
                print(f"\n⚠️  两种方法都不理想，选择DBSCAN结果（聚类数更多）")

    # 保存结果
    print(f"\n{'='*80}")
    print(f"最终聚类方法: {final_method}")
    print(f"{'='*80}")

    output_subdir = os.path.join(output_dir, final_method)
    speaker_profiles = analyze_clustering_results(final_labels, valid_faces, output_subdir)

    print(f"\n{'='*80}")
    print(f"POC完成！聚类结果已保存到: {output_subdir}")
    print(f"{'='*80}")

    print(f"\n请检查以下文件:")
    print(f"  1. 聚类结果JSON: {output_subdir}/clustering_result.json")
    print(f"  2. 代表人脸图片: {output_subdir}/speaker_*_sample.jpg")


if __name__ == "__main__":
    main()
