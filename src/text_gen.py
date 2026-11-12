"""문구 생성: 로컬 LLM(Qwen3-4B, 4bit)으로 헤드라인·서브카피·캡션·해시태그·영어 이미지 프롬프트를 만든다."""
import gc
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import LLM_ID, LLM_MAX_NEW_TOKENS, LLM_RETRIES, LLM_TEMPERATURE
from src.prompts.templates import SYSTEM, user_message

REQUIRED = ("headline", "subcopy", "caption", "hashtags", "image_prompt_en")
_state = {}


def load():
    if _state:
        return
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(LLM_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_ID, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16, local_files_only=True
    )
    model.eval()
    _state.update(tok=tok, model=model)


def unload():
    """이미지 모델에 VRAM 을 넘기기 위해 LLM 을 내린다."""
    _state.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in data for k in REQUIRED) or not isinstance(data["hashtags"], list):
        return None
    return data


def normalize_hashtags(tags) -> list[str]:
    """'#' 유무를 통일하고 공백·중복을 제거한다 (모델이 어떤 때는 붙이고 어떤 때는 안 붙임)."""
    out, seen = [], set()
    for t in tags:
        t = str(t).strip().lstrip("#").replace(" ", "")
        if t and t not in seen:
            seen.add(t)
            out.append("#" + t)
    return out


def ensure_empty_space(prompt: str) -> str:
    """글자를 얹을 빈 공간 문구가 빠졌으면 코드가 붙인다 (모델이 가끔 빼먹음)."""
    if "empty space for text" in prompt.lower():
        return prompt
    return prompt.rstrip(" .,") + ", empty space for text"


def copy_problems(data: dict) -> list[str]:
    """서브카피에 숫자(날짜·시간·할인율)가 있거나 글에 '분위기'가 있으면 문제로 본다. 이 정보는 코드가 따로 붙인다."""
    found = []
    if re.search(r"\d", data["subcopy"]):
        found.append("서브카피에 숫자")
    if "분위기" in data["subcopy"] or "분위기" in data["caption"]:
        found.append("분위기 단어")
    return found


def generate_copy(form: dict, seed: int) -> dict:
    load()
    tok, model = _state["tok"], _state["model"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(form)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
    n_in = ids["input_ids"].shape[1]
    best = None  # (문제 수, 결과): 재시도를 다 써도 못 고치면 문제가 가장 적은 것을 쓴다
    for attempt in range(LLM_RETRIES + 1):
        torch.manual_seed(seed + attempt)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=LLM_MAX_NEW_TOKENS, do_sample=True,
                                 temperature=LLM_TEMPERATURE, top_p=0.9)
        data = _parse(tok.decode(out[0][n_in:], skip_special_tokens=True))
        if not data:
            continue
        data["hashtags"] = normalize_hashtags(data["hashtags"])
        data["attempts"] = attempt + 1
        data["image_prompt_en"] = ensure_empty_space(data["image_prompt_en"])
        problems = copy_problems(data)
        if not problems:
            return data
        if best is None or len(problems) < best[0]:
            best = (len(problems), data)
    if best:
        return best[1]
    raise RuntimeError("문구 생성 결과가 JSON 형식을 지키지 못했습니다. 다시 시도해 주세요.")
