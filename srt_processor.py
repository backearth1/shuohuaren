#!/usr/bin/env python3
"""
SRT处理模块
负责SRT文件解析、说话人匹配、标注生成等功能
"""

import srt
from datetime import timedelta
from typing import List, Dict, Optional
from pyannote.core import Annotation


class SRTProcessor:
    """SRT处理器"""

    def __init__(self, speaker_format: str = "[说话人{}] {}"):
        """
        初始化SRT处理器

        Args:
            speaker_format: 说话人标注格式，第一个{}是说话人编号，第二个{}是原文本
        """
        self.speaker_format = speaker_format

    def parse_srt(self, srt_path: str) -> List[Dict]:
        """
        解析SRT文件

        Args:
            srt_path: SRT文件路径

        Returns:
            包含时间戳和文本的字典列表
        """
        print(f"解析SRT文件: {srt_path}")

        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        subtitles = list(srt.parse(content))
        segments = []

        for subtitle in subtitles:
            start_time = subtitle.start.total_seconds()
            end_time = subtitle.end.total_seconds()
            duration = end_time - start_time

            segments.append({
                'index': subtitle.index,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': subtitle.content.strip(),
                'start_timedelta': subtitle.start,
                'end_timedelta': subtitle.end
            })

        print(f"解析到 {len(segments)} 个字幕片段")
        return segments

    def match_speakers(self, srt_segments: List[Dict],
                      diarization: Annotation,
                      overlap_threshold: float = 0.5,
                      use_midpoint: bool = False) -> List[Dict]:
        """
        将SRT片段与说话人分离结果匹配

        Args:
            srt_segments: SRT片段列表
            diarization: pyannote说话人分离结果
            overlap_threshold: 重叠阈值（当use_midpoint=False时使用）
            use_midpoint: 是否使用中点匹配（True）或重叠匹配（False）

        Returns:
            包含说话人信息的片段列表
        """
        print(f"\n开始匹配SRT片段与说话人...")
        print(f"匹配策略: {'中点匹配' if use_midpoint else '重叠匹配'}")

        matched_segments = []
        unmatched_count = 0

        for segment in srt_segments:
            if use_midpoint:
                # 使用中点匹配
                midpoint = (segment['start_time'] + segment['end_time']) / 2
                speaker = self._find_speaker_at_time(diarization, midpoint)
            else:
                # 使用重叠匹配
                speaker = self._find_speaker_by_overlap(
                    diarization,
                    segment['start_time'],
                    segment['end_time'],
                    overlap_threshold
                )

            if speaker:
                segment_with_speaker = segment.copy()
                segment_with_speaker['speaker'] = speaker
                segment_with_speaker['speaker_id'] = self._speaker_to_id(speaker)
                matched_segments.append(segment_with_speaker)
            else:
                # 未匹配到说话人，标记为未知
                segment_with_speaker = segment.copy()
                segment_with_speaker['speaker'] = 'UNKNOWN'
                segment_with_speaker['speaker_id'] = -1
                matched_segments.append(segment_with_speaker)
                unmatched_count += 1

        print(f"匹配完成: {len(matched_segments) - unmatched_count}/{len(matched_segments)} 个片段成功匹配")
        if unmatched_count > 0:
            print(f"警告: {unmatched_count} 个片段未能匹配到说话人")

        return matched_segments

    def _find_speaker_at_time(self, diarization: Annotation, timestamp: float) -> Optional[str]:
        """
        查找某个时间点的说话人

        Args:
            diarization: 说话人分离结果
            timestamp: 时间点（秒）

        Returns:
            说话人标签或None
        """
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                return speaker
        return None

    def _find_speaker_by_overlap(self, diarization: Annotation,
                                 start_time: float, end_time: float,
                                 threshold: float = 0.5) -> Optional[str]:
        """
        通过重叠比例查找说话人

        Args:
            diarization: 说话人分离结果
            start_time: 开始时间
            end_time: 结束时间
            threshold: 重叠阈值

        Returns:
            说话人标签或None
        """
        query_duration = end_time - start_time
        if query_duration <= 0:
            return None

        speaker_overlaps = {}

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(segment.start, start_time)
            overlap_end = min(segment.end, end_time)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                if speaker not in speaker_overlaps:
                    speaker_overlaps[speaker] = 0
                speaker_overlaps[speaker] += overlap_duration

        if not speaker_overlaps:
            return None

        best_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])
        overlap_ratio = best_speaker[1] / query_duration

        if overlap_ratio >= threshold:
            return best_speaker[0]
        else:
            return None

    def _speaker_to_id(self, speaker: str) -> int:
        """
        将说话人标签转换为数字ID

        Args:
            speaker: 说话人标签 (如 "SPEAKER_00")

        Returns:
            数字ID
        """
        try:
            # pyannote通常使用 "SPEAKER_00", "SPEAKER_01" 等格式
            if speaker.startswith("SPEAKER_"):
                return int(speaker.split("_")[1])
            else:
                # 如果格式不同，尝试直接转换
                return int(speaker)
        except:
            # 转换失败，返回-1
            return -1

    def generate_annotated_srt(self, matched_segments: List[Dict],
                               output_path: str) -> str:
        """
        生成标注后的SRT文件

        Args:
            matched_segments: 包含说话人信息的片段列表
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        print(f"\n生成标注后的SRT文件: {output_path}")

        annotated_subtitles = []

        for segment in matched_segments:
            # 格式化内容
            speaker = segment.get('speaker', 'UNKNOWN')
            if speaker == 'UNKNOWN':
                content = f"[未识别] {segment['text']}"
            else:
                # 支持两种格式: speaker_id (数字) 或 speaker (字符串如 SPEAKER_00)
                if 'speaker_id' in segment:
                    speaker_num = segment['speaker_id'] + 1
                else:
                    # 从 SPEAKER_00 格式提取数字
                    speaker_num = int(speaker.split('_')[-1]) + 1
                content = self.speaker_format.format(speaker_num, segment['text'])

            # 创建字幕对象
            subtitle = srt.Subtitle(
                index=segment['index'],
                start=segment['start_timedelta'],
                end=segment['end_timedelta'],
                content=content
            )
            annotated_subtitles.append(subtitle)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(annotated_subtitles))

        print(f"成功生成 {len(annotated_subtitles)} 个标注字幕")
        return output_path

    def generate_report(self, matched_segments: List[Dict], output_path: str) -> str:
        """
        生成分析报告

        Args:
            matched_segments: 包含说话人信息的片段列表
            output_path: 报告输出路径

        Returns:
            报告文件路径
        """
        print(f"生成分析报告: {output_path}")

        # 统计信息
        speaker_stats = {}
        total_duration = 0
        unmatched_count = 0

        for segment in matched_segments:
            speaker = segment['speaker']
            duration = segment['duration']
            total_duration += duration

            if speaker == 'UNKNOWN':
                unmatched_count += 1
                continue

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'count': 0,
                    'total_duration': 0,
                    'segments': []
                }

            speaker_stats[speaker]['count'] += 1
            speaker_stats[speaker]['total_duration'] += duration
            speaker_stats[speaker]['segments'].append(segment)

        # 写入报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("说话人标注报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"总片段数: {len(matched_segments)}\n")
            f.write(f"总时长: {total_duration:.2f}秒\n")
            f.write(f"成功匹配: {len(matched_segments) - unmatched_count}\n")
            f.write(f"未匹配: {unmatched_count}\n\n")

            f.write("各说话人统计:\n")
            f.write("-" * 40 + "\n")

            for speaker in sorted(speaker_stats.keys()):
                stats = speaker_stats[speaker]
                speaker_id = self._speaker_to_id(speaker) + 1
                ratio = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0

                f.write(f"\n说话人{speaker_id} ({speaker}):\n")
                f.write(f"  片段数: {stats['count']}\n")
                f.write(f"  总时长: {stats['total_duration']:.2f}秒\n")
                f.write(f"  占比: {ratio:.1f}%\n")

            # 详细列表
            f.write("\n\n详细片段列表:\n")
            f.write("-" * 40 + "\n")

            for segment in matched_segments:
                speaker_label = segment.get('speaker', 'UNKNOWN')
                if speaker_label == 'UNKNOWN':
                    speaker_name = "[未识别]"
                else:
                    # 支持两种格式
                    if 'speaker_id' in segment:
                        speaker_id = segment['speaker_id'] + 1
                    else:
                        speaker_id = int(speaker_label.split('_')[-1]) + 1
                    speaker_name = f"说话人{speaker_id}"

                f.write(f"\n片段 {segment['index']}:\n")
                f.write(f"  时间: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s\n")
                f.write(f"  说话人: {speaker_name}\n")
                f.write(f"  内容: {segment['text']}\n")

        print(f"报告生成完成")
        return output_path

    def create_comparison_view(self, original_srt_path: str,
                              annotated_srt_path: str,
                              output_path: str) -> str:
        """
        创建原始与标注的对比视图

        Args:
            original_srt_path: 原始SRT路径
            annotated_srt_path: 标注后SRT路径
            output_path: 对比文件输出路径

        Returns:
            对比文件路径
        """
        print(f"创建对比视图: {output_path}")

        # 读取文件
        with open(original_srt_path, 'r', encoding='utf-8') as f:
            original_subs = list(srt.parse(f.read()))

        with open(annotated_srt_path, 'r', encoding='utf-8') as f:
            annotated_subs = list(srt.parse(f.read()))

        # 生成对比
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("原始字幕 vs 标注字幕对比\n")
            f.write("=" * 80 + "\n\n")

            for orig, annot in zip(original_subs, annotated_subs):
                f.write(f"片段 {orig.index}:\n")
                f.write(f"时间: {orig.start} --> {orig.end}\n")
                f.write(f"原始: {orig.content}\n")
                f.write(f"标注: {annot.content}\n")
                f.write("-" * 60 + "\n\n")

        print("对比视图生成完成")
        return output_path

    def validate_output(self, output_path: str) -> bool:
        """
        验证输出文件的有效性

        Args:
            output_path: 输出文件路径

        Returns:
            是否有效
        """
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            subtitles = list(srt.parse(content))

            if len(subtitles) == 0:
                print(f"警告: 文件不包含有效的字幕内容")
                return False

            # 检查是否包含说话人标注
            annotated_count = 0
            for subtitle in subtitles:
                if any(marker in subtitle.content for marker in ['[说话人', '[Speaker', '说话人']):
                    annotated_count += 1

            if annotated_count == 0:
                print(f"警告: 文件似乎没有包含说话人标注")
                return False

            print(f"验证通过: {len(subtitles)} 个字幕，{annotated_count} 个包含说话人标注")
            return True

        except Exception as e:
            print(f"验证失败: {e}")
            return False
