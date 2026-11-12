"""배경 이미지 생성: SDXL base(fp16) 를 오프로드로 실행한다. 글자는 넣지 않고 배경만 만든다."""
import gc
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from src.config import GEN_SIZE, GUIDANCE, MOODS, SDXL_ID, STEPS
from src.prompts.templates import IMAGE_SUFFIX, NEGATIVE_PROMPT


def _snapshot_dir() -> str:
    # 안전 필터 등 일부 파일을 받지 않았으므로 저장소 이름 대신 폴더 경로로 직접 연다
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    base = hf_home / "hub" / ("models--" + SDXL_ID.replace("/", "--")) / "snapshots"
    return str(next(base.iterdir()))


def build_prompt(image_prompt_en: str, mood: str) -> str:
    return f"{image_prompt_en}, {MOODS.get(mood, '')}, {IMAGE_SUFFIX}".replace(", ,", ",")


def generate_background(image_prompt_en: str, mood: str, seed: int):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        _snapshot_dir(), torch_dtype=torch.float16, variant="fp16", local_files_only=True
    )
    # 통째로 올리면 최대 10.6GB 가 되어 8GB 를 넘기므로 구성요소를 필요할 때만 GPU 로 옮긴다
    pipe.enable_model_cpu_offload()
    try:
        gen = torch.Generator("cuda").manual_seed(seed)
        return pipe(
            build_prompt(image_prompt_en, mood), negative_prompt=NEGATIVE_PROMPT,
            width=GEN_SIZE[0], height=GEN_SIZE[1], num_inference_steps=STEPS,
            guidance_scale=GUIDANCE, generator=gen,
        ).images[0]
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
