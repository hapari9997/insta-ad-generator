"""글자 합성: AI가 그린 배경 위에 헤드라인·서브카피와 사용자가 입력한 일시·장소·혜택을 Pillow로 얹는다.

일시·장소·혜택은 LLM 을 거치지 않고 입력값 그대로 쓴다 (LLM 은 이 값을 문장으로 풀어 써서 글자가 달라졌다).
"""
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import FEED_SIZE, FONT_PATH

MARGIN_X = 84
BOTTOM = 96
ACCENT = (255, 214, 102)


def _font(size: int, weight: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_PATH), size)
    try:
        f.set_variation_by_axes([weight])  # 가변 폰트의 굵기 축
    except Exception:  # noqa: BLE001  굵기 조절이 안 되는 환경에서는 기본 굵기를 쓴다
        pass
    return f


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = word if not cur else f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
            cur = ""
        if draw.textlength(word, font=font) <= max_w:
            cur = word
            continue
        for ch in word:  # 공백 없이 너무 긴 단어는 글자 단위로 자른다
            if draw.textlength(cur + ch, font=font) <= max_w:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


def compose(background: Image.Image, headline: str, subcopy: str, info: list[tuple[str, str]]) -> Image.Image:
    W, H = FEED_SIZE
    img = ImageOps.fit(background.convert("RGB"), (W, H), Image.LANCZOS)

    # 아래쪽을 어둡게 눌러 글자가 읽히게 한다
    alpha = Image.new("L", (1, H))
    start = int(H * 0.38)
    for y in range(H):
        alpha.putpixel((0, y), 0 if y < start else int(215 * ((y - start) / (H - start)) ** 1.2))
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shade.putalpha(alpha.resize((W, H)))
    img = Image.alpha_composite(img.convert("RGBA"), shade)

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    max_w = W - MARGIN_X * 2

    # 헤드라인이 3줄을 넘으면 글자 크기를 줄인다
    size = 96
    while True:
        f_head = _font(size, 800)
        head_lines = _wrap(d, headline, f_head, max_w)
        if len(head_lines) <= 3 or size <= 56:
            break
        size -= 6
    f_sub, f_label, f_val = _font(44, 400), _font(34, 700), _font(34, 400)
    sub_lines = _wrap(d, subcopy, f_sub, max_w)
    label_w = max((d.textlength(label + "  ", font=f_label) for label, _ in info), default=0)
    info_lines = [(label, _wrap(d, value, f_val, max_w - label_w)) for label, value in info]

    lh = lambda f, k=1.32: int(f.size * k)  # noqa: E731
    total = lh(f_head, 1.25) * len(head_lines) + 22 + lh(f_sub) * len(sub_lines)
    if info_lines:
        total += 40 + sum(lh(f_val) * len(v) for _, v in info_lines) + 10 * (len(info_lines) - 1)

    y = H - BOTTOM - total

    def text(x, yy, s, font, fill):
        d.text((x + 3, yy + 3), s, font=font, fill=(0, 0, 0, 150))  # 그림자
        d.text((x, yy), s, font=font, fill=fill)

    for line in head_lines:
        text(MARGIN_X, y, line, f_head, (255, 255, 255, 255))
        y += lh(f_head, 1.25)
    y += 22
    for line in sub_lines:
        text(MARGIN_X, y, line, f_sub, (235, 235, 235, 255))
        y += lh(f_sub)
    if info_lines:
        y += 40
        for label, vlines in info_lines:
            text(MARGIN_X, y, label, f_label, ACCENT + (255,))
            for v in vlines:
                text(MARGIN_X + label_w, y, v, f_val, (255, 255, 255, 255))
                y += lh(f_val)
            y += 10

    return Image.alpha_composite(img, layer).convert("RGB")
