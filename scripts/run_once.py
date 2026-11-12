"""화면 없이 파이프라인을 한 번 돌려 결과를 확인한다.  사용법: python scripts/run_once.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HF_HUB_OFFLINE"] = "1"

from src.pipeline import run  # noqa: E402

FORM = {"업종": "꽃집", "행사명": "봄맞이 꽃다발 할인", "일시": "3월 15일~3월 21일",
        "장소": "서울 마포구 연남동 '라일락 플라워'", "혜택": "꽃다발 20% 할인", "분위기": "따뜻함"}

res = run(FORM, seed=42, on_step=lambda s: print("·", s, flush=True))
print("저장:", res["path"])
print("시간:", res["timings"])
print("--- 헤드라인:", res["copy"]["headline"])
print("--- 서브카피:", res["copy"]["subcopy"])
print("--- 영어 프롬프트:", res["copy"]["image_prompt_en"])
print("--- 캡션 ---")
print(res["caption"])
