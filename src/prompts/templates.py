"""프롬프트 템플릿. 사용자는 폼만 채우고, 프롬프트는 서비스가 조립한다."""

SYSTEM = (
    "너는 소상공인의 인스타그램 게시물을 대신 써 주는 카피라이터다. "
    "반드시 JSON 객체 하나만 출력하고 다른 설명은 쓰지 마라. "
    "키는 headline, subcopy, caption, hashtags, image_prompt_en 이다. "
    "headline은 15자 안팎의 한국어 한 줄, subcopy는 25자 안팎의 한국어 한 줄이다. "
    "caption은 한국어 2~3문장으로 행사의 분위기와 매력을 전한다. "
    "날짜·요일·시간·휴무·장소·혜택 같은 구체 정보는 subcopy와 caption에 쓰지 마라. 서비스가 따로 붙인다. "
    "입력의 '문체'는 말투와 느낌에만 반영하고, '분위기'라는 단어나 문체 이름(모던, 레트로 등)을 한국어 문장에 그대로 쓰지 마라. "
    "hashtags는 한국어 해시태그 문자열 6~8개의 배열이다. "
    "image_prompt_en은 이 업종과 분위기에 어울리는, 글자가 없는 배경 이미지를 묘사하는 영어 한 문장이며 "
    "'no text'와 'empty space for text'를 포함한다. "
    "사용자가 준 정보를 바꾸거나 지어내지 마라."
)

NEGATIVE_PROMPT = "text, letters, words, watermark, logo, signature, low quality, blurry, deformed"
IMAGE_SUFFIX = "professional photography, high quality, clean composition"


def user_message(form: dict) -> str:
    # 혜택은 LLM 이 줄여 쓰다가 뜻이 바뀌므로 넘기지 않는다. 코드가 원문 그대로 붙인다.
    keys = ("업종", "행사명", "일시", "장소", "분위기")
    lines = [f"- {'문체' if k == '분위기' else k}: {form[k]}" for k in keys if form.get(k)]
    return "다음 정보로 인스타그램 홍보 게시물 문구를 만들어줘.\n" + "\n".join(lines)
