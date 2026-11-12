"""문구 모델(Qwen3-4B, 4bit) 검증. 이미지 모델은 불러오지 않는다.

사용법 (프로젝트 폴더에서):  python -m evaluation.eval_text [--seeds 2]
정답 문장이 없으므로 BLEU 같은 지표 대신, 서비스가 요구하는 형식과 규칙을 지켰는지를 자동으로 센다.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402

from evaluation.cases import CASES  # noqa: E402
from src import text_gen  # noqa: E402
from src.pipeline import build_caption  # noqa: E402

RESULT_DIR = ROOT / "evaluation" / "results"

CHECKS = {
    "json_first_try": "첫 시도에 JSON 형식 성공",
    "json_final": "재시도 포함 JSON 성공",
    "headline_len": "헤드라인 5~22자",
    "subcopy_len": "서브카피 10~40자",
    "caption_len": "캡션 40~300자",
    "hashtag_count": "해시태그 6~8개",
    "hashtag_format": "해시태그가 #로 시작하고 공백 없음",
    "caption_no_leak": "캡션에 일시·혜택 숫자가 없음",
    "image_prompt_ok": "영어 프롬프트 (영문, no text, empty space 포함)",
    "info_preserved": "최종 캡션에 일시·장소·혜택이 입력 그대로 있음",
    "event_ref": "행사명 단어가 문구에 반영됨",
}


def check(form: dict, copy: dict) -> dict:
    head, sub, cap, tags, prompt = copy["headline"], copy["subcopy"], copy["caption"], copy["hashtags"], copy["image_prompt_en"]
    digits = set(re.findall(r"\d+", form.get("일시", "") + " " + form.get("혜택", "")))
    non_ascii = sum(ord(c) > 127 for c in prompt) / max(len(prompt), 1)
    final = build_caption(copy, form)
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", form["행사명"])
    return {
        "json_first_try": copy["attempts"] == 1,
        "json_final": True,
        "headline_len": 5 <= len(head) <= 22,
        "subcopy_len": 10 <= len(sub) <= 40,
        "caption_len": 40 <= len(cap) <= 300,
        "hashtag_count": 6 <= len(tags) <= 8,
        "hashtag_format": all(re.fullmatch(r"#\S+", t) for t in tags),
        "caption_no_leak": not any(d in cap for d in digits),
        "image_prompt_ok": non_ascii <= 0.05 and "no text" in prompt.lower() and "empty space" in prompt.lower(),
        "info_preserved": all(form[k] in final for k in ("일시", "장소", "혜택") if form.get(k)),
        "event_ref": any(t in head + sub + cap for t in tokens),
    }


def pct(n, total):
    return f"{n}/{total} ({100 * n / total:.0f}%)" if total else "-"


def write_report(path: Path, meta: dict, runs: list) -> None:
    total = len(runs)
    L = [f"# 문구 모델 검증 결과 ({meta['when']})", "",
         f"- 모델: {meta['model']}  /  케이스 {meta['cases']}개 × 시드 {meta['seeds']}개 = {total}회",
         f"- 로드 {meta['load_s']}초, 최대 VRAM {meta['peak_vram_gb']}GB, 1회 평균 {meta['mean_s']}초 (최대 {meta['max_s']}초)", "",
         "## 자동 점검 요약", "", "| 점검 | 통과 |", "|---|---|"]
    for key, label in CHECKS.items():
        n = sum(r["checks"].get(key, False) for r in runs)
        L.append(f"| {label} | {pct(n, total)} |")
    L += ["", f"- 같은 입력에서 시드를 바꿨을 때 헤드라인이 서로 다른 비율: {meta['diversity']}", "",
          "## 실패한 점검", "", "| 케이스 | 시드 | 실패 항목 |", "|---|---|---|"]
    bad = [(r, [CHECKS[k] for k, v in r["checks"].items() if not v]) for r in runs]
    bad = [(r, f) for r, f in bad if f]
    L += [f"| {r['case_id']} | {r['seed']} | {', '.join(f)} |" for r, f in bad] or ["| (없음) | | |"]
    L += ["", "## 생성 결과 전문 (눈으로 읽어 보는 용도)", ""]
    for r in runs:
        L.append(f"### {r['case_id']} / seed {r['seed']} ({r['seconds']}초)")
        if r["copy"]:
            c = r["copy"]
            L += [f"- 헤드라인: {c['headline']}", f"- 서브카피: {c['subcopy']}", f"- 캡션: {c['caption']}",
                  f"- 해시태그: {' '.join(c['hashtags'])}", f"- 영어 프롬프트: {c['image_prompt_en']}", ""]
        else:
            L += [f"- 실패: {r['error']}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2, help="케이스마다 시드 몇 개로 돌릴지")
    args = ap.parse_args()
    seeds = [100 + k for k in range(args.seeds)]

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    text_gen.load()
    load_s = round(time.time() - t, 1)

    runs = []
    for case in CASES:
        for seed in seeds:
            print(f"· {case['id']} seed {seed}", flush=True)
            t = time.time()
            try:
                copy = text_gen.generate_copy(case["form"], seed)
                checks, err = check(case["form"], copy), None
            except RuntimeError as e:  # 재시도해도 JSON 이 안 나온 경우
                copy, err = None, str(e)
                checks = {k: False for k in CHECKS}
            runs.append({"case_id": case["id"], "seed": seed, "seconds": round(time.time() - t, 1),
                         "copy": copy, "error": err, "checks": checks})

    diverse = [len({r["copy"]["headline"] for r in runs if r["case_id"] == c["id"] and r["copy"]}) == len(seeds) for c in CASES]
    secs = [r["seconds"] for r in runs]
    meta = {"when": f"{datetime.now():%Y-%m-%d %H:%M}", "model": text_gen.LLM_ID, "cases": len(CASES), "seeds": len(seeds),
            "load_s": load_s, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "mean_s": round(sum(secs) / len(secs), 1), "max_s": max(secs), "diversity": pct(sum(diverse), len(diverse))}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    (RESULT_DIR / f"text_eval_{stamp}.json").write_text(
        json.dumps({"meta": meta, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(RESULT_DIR / f"text_eval_{stamp}.md", meta, runs)
    print("저장:", RESULT_DIR / f"text_eval_{stamp}.md")


if __name__ == "__main__":
    main()
