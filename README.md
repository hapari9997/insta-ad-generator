# 인스타 홍보 게시물 생성기

업종·행사명·일시·장소·혜택을 입력하면 인스타그램 피드 이미지(4:5), 캡션, 해시태그를 한 번에 만들어 주는 Streamlit 앱입니다. 자세한 설계와 검증은 [`docs/report_ad-gen.pdf`](docs/report_ad-gen.pdf)를 참고하세요.

## 동작 흐름

```
입력 폼 → 문구 생성(Qwen3-4B) → LLM 해제 → 배경 생성(SDXL) → 글자 합성(Pillow) → 이미지 + 캡션 + 해시태그
```

- 8GB VRAM에서 돌도록 두 모델을 한 번에 하나씩 올리고 내립니다.
- 일시·장소·혜택은 LLM이 쓰지 않고 입력한 글자 그대로 이미지와 캡션에 붙습니다.
- 모델을 학습하지 않고 사전학습 모델을 그대로 사용합니다.

## 요구 사양

| 항목 | 값 |
|---|---|
| GPU | NVIDIA, VRAM 8GB 이상 (RTX 4060 8GB에서 개발·측정, 최대 사용 5.6GB) |
| Python | 3.13 |
| 디스크 | 모델 약 14GB (Qwen3-4B 7.6GB + SDXL 6.5GB) |
| 생성 시간 | 1건당 약 1분 (문구 30초 안팎 + 이미지 25초 안팎) |

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`torch`는 CUDA 빌드(개발 환경: `2.12.0+cu132`)가 필요합니다. 자신의 CUDA 버전에 맞는 빌드를 [PyTorch 안내](https://pytorch.org/get-started/locally/)로 먼저 설치하세요.

## 모델 준비

모델은 오프라인으로 읽으므로 처음 한 번 Hugging Face에서 내려받아 캐시에 둡니다. 이후에는 인터넷 없이 실행됩니다.

| 용도 | 모델 |
|---|---|
| 문구 생성 | `Qwen/Qwen3-4B-Instruct-2507` (4bit 양자화 로드) |
| 배경 이미지 | `stabilityai/stable-diffusion-xl-base-1.0` (fp16, CPU 오프로드) |

```bash
hf download Qwen/Qwen3-4B-Instruct-2507
hf download stabilityai/stable-diffusion-xl-base-1.0
```

## 실행

```bash
python -m streamlit run app.py
```

브라우저에서 폼을 채우고 생성하기를 누릅니다.

- 다시 생성: 같은 입력으로 새 결과를 만듭니다.
- 서브카피 수정: 결과 이미지 아래 입력칸(최대 40자)에서 고치고 글자 다시 입히기를 누르면 배경은 그대로 두고 글자만 다시 얹습니다.
- 동시 접속: 요청은 대기번호 순서대로 처리하고, 대기 중에는 앞사람 수와 예상 시간을 보여 줍니다.
- 내보내기: 이미지 PNG 다운로드, 캡션 복사.

화면 없이 한 번만 돌려 보려면:

```bash
python scripts/run_once.py
```

## 폴더 구조

```
app.py                 Streamlit 화면
src/
  pipeline.py          전체 흐름 (문구 → 배경 → 글자 합성)
  text_gen.py          문구 생성, JSON 파싱, 출력 검증·재시도
  image_gen.py         SDXL 배경 생성
  compose.py           Pillow 글자 합성
  ticket_queue.py      대기번호 큐
  config.py            모델 ID, 이미지 규격, 분위기 목록
  prompts/templates.py 시스템 프롬프트와 입력 조립
evaluation/            문구·이미지 검증 스크립트, 케이스, 결과
scripts/run_once.py    화면 없이 1회 실행
assets/fonts/          Noto Sans KR
docs/                  보고서
```

## 출력 품질 관리

| 문제 | 대응 |
|---|---|
| LLM이 JSON 형식을 어김 | 시드를 바꿔 최대 3회 재시도 |
| 서브카피에 날짜·시간·할인율 등 숫자가 섞임 | 숫자가 있으면 재시도, 3회로 못 고치면 문제가 가장 적은 결과 사용 |
| 이미지 프롬프트에 "empty space for text" 누락 | 코드가 덧붙임 |
| 혜택 문구를 LLM이 줄여 쓰다 뜻이 바뀜 | LLM에 전달하지 않고 코드가 원문을 붙임 |

12개 업종 × 시드 2개 = 24회 검증 결과: JSON 형식 24/24, 서브카피 숫자 2/24, "분위기" 단어 5/24 (`evaluation/results/`).

## 알려진 한계

- 피사체가 글자 뒤에 놓이는 이미지가 있고, 밝은 배경에서는 글자 대비가 약합니다.
- 대기 중 같은 사용자가 화면을 다시 실행하면 대기번호를 잃습니다.

## 라이선스

| 구성 요소 | 라이선스 |
|---|---|
| Qwen3-4B-Instruct-2507 | Apache-2.0 |
| Stable Diffusion XL base 1.0 | CreativeML Open RAIL++-M |
| Noto Sans KR (`assets/fonts`) | SIL Open Font License 1.1 |
