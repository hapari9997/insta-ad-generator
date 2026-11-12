"""모델 ID, 이미지 규격, 분위기 목록 등 서비스 전체 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
FONT_PATH = ROOT / "assets" / "fonts" / "NotoSansKR-VF.ttf"

# 문구 생성 (4bit nf4, 측정: 로드 후 약 2.8GB, 약 8 tok/s)
LLM_ID = "Qwen/Qwen3-4B-Instruct-2507"
LLM_MAX_NEW_TOKENS = 450
LLM_RETRIES = 2  # JSON 이 깨졌을 때 다시 시도하는 횟수
LLM_TEMPERATURE = 0.6  # 다시 생성하면 다른 문구가 나오도록 샘플링 사용

# 배경 이미지 (fp16, 오프로드 필수: 통째로 올리면 8GB 를 넘겨 매우 느려짐)
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
GEN_SIZE = (896, 1120)  # 4:5, 8의 배수
FEED_SIZE = (1080, 1350)  # 인스타 피드 4:5
STEPS = 25
GUIDANCE = 7.0

# 분위기 → 이미지 프롬프트에 붙는 영어 스타일 문구
MOODS = {
    "따뜻함": "warm golden light, cozy, inviting atmosphere, soft pastel tones",
    "미니멀": "minimalist, clean lines, neutral tones, lots of negative space",
    "레트로": "retro vintage style, film grain, muted warm colors, 1980s mood",
    "모던": "modern, sleek, high contrast, clean professional lighting",
}
