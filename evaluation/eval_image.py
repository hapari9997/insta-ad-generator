"""이미지 모델(SDXL) 검증. 문구 모델은 불러오지 않는다.

사용법 (프로젝트 폴더에서):  python -m evaluation.eval_image [--seeds 1] [--source auto|text|fixed]
프롬프트와 문구는 가장 최근 문구 검증 결과(text_eval_*.json)에서 가져오고, 없으면 cases.py 의 고정값을 쓴다.
자동으로 재는 것: 시간, VRAM, 글자 대비, 글자 배치.  사람이 봐야 하는 것: review.html 의 체크리스트.
"""
import argparse
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation.cases import CASES  # noqa: E402
from src.compose import BOTTOM, MARGIN_X, compose  # noqa: E402
from src.config import FEED_SIZE, SDXL_ID  # noqa: E402
from src.image_gen import generate_background  # noqa: E402

RESULT_DIR = ROOT / "evaluation" / "results"
CONTRAST_MIN = 4.5   # WCAG AA 일반 글자 기준 (흰 글자 기준으로 계산)
BLOCK_MAX = 0.5      # 글자 영역이 이미지 높이의 절반을 넘으면 배경이 가려진다고 본다

HUMAN_CHECKS = ["글자가 읽힌다", "배경에 글자·로고가 없다", "업종·분위기에 어울린다", "피사체가 글자에 가려지지 않는다"]


def latest_text_result():
    files = sorted(RESULT_DIR.glob("text_eval_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def pick_inputs(source: str):
    """케이스마다 (헤드라인, 서브카피, 영어 프롬프트, 출처)를 정한다."""
    res = latest_text_result() if source != "fixed" else None
    if source == "text" and not res:
        sys.exit("text_eval 결과가 없습니다. 먼저 python -m evaluation.eval_text 를 실행하세요.")
    out = {}
    for c in CASES:
        run = next((r for r in (res["runs"] if res else []) if r["case_id"] == c["id"] and r["copy"]), None)
        if run:
            cp = run["copy"]
            out[c["id"]] = (cp["headline"], cp["subcopy"], cp["image_prompt_en"], "문구 모델 출력")
        else:
            f = c["fixed"]
            out[c["id"]] = (f["headline"], f["subcopy"], f["image_prompt_en"], "고정값")
    return out


def measure_text_area(final, base):
    """글자 없이 합성한 이미지와 비교해 글자가 놓인 곳을 찾고, 그 아래 배경과 흰 글자의 대비를 잰다."""
    a, b = np.asarray(final, dtype=np.int16), np.asarray(base, dtype=np.int16)
    mask = np.abs(a - b).sum(axis=2) > 40
    ys, xs = np.where(mask)
    W, H = FEED_SIZE
    under = np.asarray(base, dtype=np.float64)[mask] / 255
    lin = np.where(under <= 0.03928, under / 12.92, ((under + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    contrast = 1.05 / (np.percentile(lum, 90) + 0.05)  # 가장 밝은 쪽 10%를 기준으로 한 최악에 가까운 값
    left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
    return {
        "contrast": round(float(contrast), 2),
        "contrast_ok": bool(contrast >= CONTRAST_MIN),
        "inside_margin": bool(left >= MARGIN_X - 2 and right <= W - MARGIN_X + 6 and bottom <= H - BOTTOM + 20),
        "block_ratio": round(float((bottom - top) / H), 2),
        "block_ok": bool((bottom - top) / H <= BLOCK_MAX),
    }


def write_review_html(path: Path, rows: list) -> None:
    cards = []
    for i, r in enumerate(rows):
        boxes = "".join(f'<label><input type="checkbox" data-i="{i}" data-k="{k}"> {html.escape(h)}</label>'
                        for k, h in enumerate(HUMAN_CHECKS))
        cards.append(f'<section><img src="{r["file"]}" loading="lazy"><div><h3>{html.escape(r["case_id"])} / seed {r["seed"]}</h3>'
                     f'<p>대비 {r["contrast"]} · 글자 영역 {r["block_ratio"]}</p>{boxes}</div></section>')
    page = """<!doctype html><meta charset="utf-8"><title>이미지 검증 체크리스트</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
body{margin:0;padding:16px;max-width:1100px;margin-inline:auto}
section{display:flex;gap:16px;margin:16px 0;flex-wrap:wrap;align-items:flex-start}
img{width:320px;max-width:100%;border-radius:8px}
label{display:block;margin:6px 0}
#bar{position:sticky;top:0;padding:8px 0;background:Canvas}
</style>
<div id="bar"><b id="sum"></b> <button id="copy">결과 복사 (마크다운)</button></div>
""" + "".join(cards) + """
<script>
const names = __NAMES__, rows = __ROWS__, KEY = "image-review:" + location.pathname;
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const boxes = [...document.querySelectorAll("input[type=checkbox]")];
function sum() {
  const n = boxes.filter(b => b.checked).length;
  document.getElementById("sum").textContent = `체크 ${n} / ${boxes.length}`;
  try { localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(boxes.map(b => [b.dataset.i + "_" + b.dataset.k, b.checked])))); } catch (e) {}
}
boxes.forEach(b => { b.checked = !!st[b.dataset.i + "_" + b.dataset.k]; b.onchange = sum; });
document.getElementById("copy").onclick = () => {
  const head = "| 케이스 | 시드 | " + names.join(" | ") + " |\\n|---|---|" + names.map(() => "---").join("|") + "|\\n";
  const body = rows.map((r, i) => `| ${r[0]} | ${r[1]} | ` + names.map((_, k) => boxes.find(b => b.dataset.i == i && b.dataset.k == k).checked ? "O" : "X").join(" | ") + " |").join("\\n");
  navigator.clipboard.writeText(head + body).catch(() => {});
};
sum();
</script>"""
    page = (page.replace("__NAMES__", json.dumps(HUMAN_CHECKS, ensure_ascii=False))
                .replace("__ROWS__", json.dumps([[r["case_id"], r["seed"]] for r in rows], ensure_ascii=False)))
    path.write_text(page, encoding="utf-8")


def write_report(path: Path, meta: dict, rows: list) -> None:
    n = len(rows)
    L = [f"# 이미지 모델 검증 결과 ({meta['when']})", "",
         f"- 모델: {meta['model']} (fp16, cpu offload)  /  {meta['images']}장, 입력 출처: {meta['source']}",
         f"- 1장 평균 {meta['mean_s']}초 (최대 {meta['max_s']}초, 모델 로드 포함), 최대 VRAM {meta['peak_vram_gb']}GB", "",
         "## 자동 점검 요약", "", "| 점검 | 통과 |", "|---|---|",
         f"| 글자 대비 {CONTRAST_MIN} 이상 | {sum(r['contrast_ok'] for r in rows)}/{n} |",
         f"| 글자가 여백 안에 배치됨 | {sum(r['inside_margin'] for r in rows)}/{n} |",
         f"| 글자 영역이 높이의 {int(BLOCK_MAX * 100)}% 이하 | {sum(r['block_ok'] for r in rows)}/{n} |", "",
         "## 이미지별 수치", "", "| 케이스 | 시드 | 시간(초) | VRAM(GB) | 대비 | 글자 영역 비율 | 여백 | 파일 |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['case_id']} | {r['seed']} | {r['seconds']} | {r['peak_vram_gb']} | {r['contrast']} | {r['block_ratio']} | "
          f"{'O' if r['inside_margin'] else 'X'} | {r['file']} |" for r in rows]
    L += ["", "## 사람이 확인하는 항목", "", "같은 폴더의 `review.html` 을 브라우저로 열어 이미지마다 아래 항목을 체크한다.", ""]
    L += [f"- {h}" for h in HUMAN_CHECKS]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="케이스마다 시드 몇 개로 그릴지")
    ap.add_argument("--source", choices=["auto", "text", "fixed"], default="auto")
    args = ap.parse_args()
    seeds = [100 + k for k in range(args.seeds)]
    inputs = pick_inputs(args.source)

    out_dir = RESULT_DIR / "images" / f"image_eval_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        head, sub, prompt, _ = inputs[case["id"]]
        info = [(k, case["form"][k]) for k in ("일시", "장소", "혜택") if case["form"].get(k)]
        for seed in seeds:
            print(f"· {case['id']} seed {seed}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            bg = generate_background(prompt, case["form"]["분위기"], seed)
            sec = round(time.time() - t, 1)
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            final = compose(bg, head, sub, info)
            base = compose(bg, "", "", [])
            name = f"{case['id']}_s{seed}.png"
            final.save(out_dir / name)
            rows.append({"case_id": case["id"], "seed": seed, "seconds": sec, "peak_vram_gb": peak, "file": name,
                         **measure_text_area(final, base)})

    secs = [r["seconds"] for r in rows]
    used = sorted({v[3] for v in inputs.values()})
    meta = {"when": f"{datetime.now():%Y-%m-%d %H:%M}", "model": SDXL_ID, "images": len(rows), "source": ", ".join(used),
            "mean_s": round(sum(secs) / len(secs), 1), "max_s": max(secs), "peak_vram_gb": max(r["peak_vram_gb"] for r in rows)}
    (out_dir / "result.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", meta, rows)
    write_review_html(out_dir / "review.html", rows)
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
