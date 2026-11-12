"""전체 흐름: 문구 생성 → LLM 해제 → 배경 생성 → 글자 합성. 8GB VRAM 때문에 두 모델은 순차로 실행한다."""
import time
from datetime import datetime

from src import image_gen, text_gen
from src.compose import compose
from src.config import OUTPUT_DIR


def build_caption(copy: dict, form: dict) -> str:
    """LLM 이 쓴 캡션 뒤에 입력한 정보를 그대로 붙이고 해시태그를 단다."""
    info = [f"{icon} {label}: {form[label]}" for icon, label in (("📅", "일시"), ("📍", "장소"), ("🎁", "혜택")) if form.get(label)]
    return copy["caption"].strip() + "\n\n" + "\n".join(info) + "\n\n" + " ".join(copy["hashtags"])


def recompose(background, headline: str, subcopy: str, form: dict):
    """배경은 그대로 두고 글자만 다시 얹는다. 모델을 돌리지 않아 바로 끝난다."""
    info = [(label, form[label]) for label in ("일시", "장소", "혜택") if form.get(label)]
    return compose(background, headline, subcopy, info)


def run(form: dict, seed: int, on_step=lambda label: None) -> dict:
    timings = {}

    on_step("문구를 만드는 중")
    t = time.time()
    copy = text_gen.generate_copy(form, seed)
    timings["문구 생성"] = round(time.time() - t, 1)

    on_step("모델을 바꾸는 중")
    t = time.time()
    text_gen.unload()
    timings["LLM 해제"] = round(time.time() - t, 1)

    on_step("배경 이미지를 그리는 중")
    t = time.time()
    bg = image_gen.generate_background(copy["image_prompt_en"], form["분위기"], seed)
    timings["배경 생성"] = round(time.time() - t, 1)

    on_step("글자를 얹는 중")
    t = time.time()
    final = recompose(bg, copy["headline"], copy["subcopy"], form)
    timings["글자 합성"] = round(time.time() - t, 1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"post_{datetime.now():%Y%m%d_%H%M%S}.png"
    final.save(path)

    return {"image": final, "bg": bg, "path": path, "copy": copy, "caption": build_caption(copy, form),
            "hashtags": " ".join(copy["hashtags"]), "timings": timings, "seed": seed}
