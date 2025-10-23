"""
配置文件
用于存储默认参数和设置
"""

# HuggingFace配置
# 获取令牌: https://huggingface.co/settings/tokens
# 首次使用pyannote.audio需要接受模型许可并提供令牌
HUGGINGFACE_CONFIG = {
    'token': None,  # 填入你的HuggingFace访问令牌
}

# LLM API配置
LLM_CONFIG = {
    # MiniMax API
    'minimax_api_key': "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmnZzno4oiLCJVc2VyTmFtZSI6IuadnOejiiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzQ3MTc5MTg3ODQ5OTI0NzU4IiwiUGhvbmUiOiIxMzAyNTQ5MDQyMyIsIkdyb3VwSUQiOiIxNzQ3MTc5MTg3ODQxNTM2MTUwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiZGV2aW5AbWluaW1heGkuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjQtMTItMjMgMTE6NTE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.szVUN2AH7lJ9fQ3EYfzcLcamSCFAOye3Y6yO3Wj_tlNhnhBIYxEEMvZsVgH9mgOe6uhRczOqibmEMbVMUD_1DqtykrbD5klaB4_nhRnDl8fbaAf7m8B1OTRTUIiqgXRVglITenx3K_ugZ6teqiqypByJoLleHbZCSPWvy1-NaDiynb7qAsGzN1V6N4BOTNza1hL5PYdlrXLe2yjQv3YW8nOjQDIGCO1ZqnVBF0UghVaO4V-GZu1Z_0JnkLa7x_2ZXKXAe-LWhk9npwGFzQfLL3aH4oUzlsoEDGnuz3RZdZsFCe95MUiG8dCWfsxhVqlQ5GoFM3LQBAXuLZyqDpmSgg",

    # 阿里通义千问API
    'qwen_api_key': "",  # 填入你的阿里云API Key

    # OpenAI API（可选）
    'openai_api_key': "",

    # Claude API（可选）
    'anthropic_api_key': "",

    # 默认VLM提供商: "minimax" 或 "qwen"
    'default_vlm_provider': "qwen",

    # 默认LLM提供商
    'default_provider': "minimax",

    # LLM权重（0-1之间，建议0.2-0.4）
    'default_llm_weight': 0.3,
}

# SRT匹配配置
MATCHING_CONFIG = {
    'overlap_threshold': 0.5,  # 重叠匹配阈值 (0-1)
    'use_midpoint': False,     # 是否使用中点匹配策略
}

# 输出格式配置
OUTPUT_CONFIG = {
    'speaker_format': '[说话人{}] {}',  # 说话人标注格式
    'output_suffix': '_annotated',      # 输出文件后缀
}

# 输入文件配置（可选，用于脚本模式）
INPUT_CONFIG = {
    'audio_file': None,     # 音频文件路径
    'srt_file': None,       # SRT文件路径
    'num_speakers': None,   # 说话人数量
    'output_file': None,    # 输出文件路径（None表示自动生成）
}
