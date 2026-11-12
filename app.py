"""Streamlit 진입점: 폼을 채우면 인스타 피드 이미지 + 캡션 + 해시태그를 한 번에 만든다."""
import io
import random
import time

import streamlit as st

from src.config import MOODS
from src.pipeline import recompose, run
from src.ticket_queue import TicketQueue

SUBCOPY_MAX = 40  # 이미지에 3줄 안팎으로 들어가는 길이 (문구 검증의 서브카피 상한과 같음)

st.set_page_config(page_title="인스타 홍보 게시물 만들기", page_icon="📸", layout="wide")


@st.cache_resource
def tickets() -> TicketQueue:
    # GPU 하나에 모델 두 개를 번갈아 올리므로 동시에 두 요청이 들어오면 번호표를 주고 순서대로 처리한다
    return TicketQueue()


def generate(form: dict, seed: int):
    queue = tickets()
    ticket = queue.take()
    started = None
    with st.status("순서를 확인하는 중입니다.", expanded=True) as status:
        try:
            queue.wait_turn(ticket, on_wait=lambda ahead, serving, eta: status.update(
                label=f"내 번호 {ticket}번 · 현재 {serving}번 처리 중 · 앞에 {ahead}명 · 예상 약 {max(1, round(eta / 60))}분"))
            status.update(label="만드는 중입니다. 보통 1분 안팎이 걸립니다.")
            started = time.time()
            result = run(form, seed, on_step=lambda label: status.write(label))
        except Exception as e:  # noqa: BLE001  사용자에게는 원인만 보여준다
            status.update(label="실패했습니다", state="error")
            st.error(str(e))
            return
        finally:
            # 실패하거나 기다리다 페이지를 떠나도 번호를 반납해야 뒷사람이 멈추지 않는다
            queue.finish(ticket, time.time() - started if started else None)
        status.update(label="완성되었습니다", state="complete", expanded=False)
    st.session_state["result"] = result
    st.session_state["form"] = form


def apply_subcopy():
    """고친 서브카피로 글자만 다시 얹는다. 모델을 쓰지 않으므로 대기번호가 필요 없다."""
    res = st.session_state["result"]
    new = st.session_state[f"sub_{res['seed']}"].strip()
    if not new or new == res["copy"]["subcopy"]:
        return
    res["copy"]["subcopy"] = new
    res["image"] = recompose(res["bg"], res["copy"]["headline"], new, st.session_state["form"])
    res["image"].save(res["path"])


st.title("📸 인스타 홍보 게시물 만들기")
st.caption("정보만 입력하면 피드 이미지, 캡션, 해시태그를 한 번에 만들어 드립니다. 모든 생성은 이 PC에서 실행됩니다.")

left, right = st.columns([2, 3], gap="large")

with left:
    with st.form("input"):
        업종 = st.text_input("업종", placeholder="예: 꽃집, 카페, 자동차 정비소")
        행사명 = st.text_input("행사명", placeholder="예: 봄맞이 꽃다발 할인")
        일시 = st.text_input("일시", placeholder="예: 3월 15일~3월 21일")
        장소 = st.text_input("장소", placeholder="예: 서울 마포구 연남동 '라일락 플라워'")
        혜택 = st.text_input("혜택 (선택)", placeholder="예: 꽃다발 20% 할인")
        분위기 = st.selectbox("분위기", list(MOODS))
        submitted = st.form_submit_button("생성하기", type="primary", use_container_width=True)

    if submitted:
        form = {"업종": 업종.strip(), "행사명": 행사명.strip(), "일시": 일시.strip(), "장소": 장소.strip(),
                "혜택": 혜택.strip(), "분위기": 분위기}
        missing = [k for k in ("업종", "행사명", "일시", "장소") if not form[k]]
        if missing:
            st.warning("필수 항목을 입력해 주세요: " + ", ".join(missing))
        else:
            generate(form, random.randint(0, 2**31 - 1))

    if "result" in st.session_state and st.button("마음에 안 들어요 · 다시 생성", use_container_width=True):
        generate(st.session_state["form"], random.randint(0, 2**31 - 1))

with right:
    res = st.session_state.get("result")
    if res is None:
        st.info("왼쪽 폼을 채우고 '생성하기'를 눌러 주세요.")
    else:
        img_col, text_col = st.columns([1, 1], gap="medium")
        with img_col:
            st.image(res["image"], use_container_width=True)
            with st.form("edit_subcopy", border=False):
                # 새로 생성하면 seed 가 바뀌어 입력칸도 새 문구로 채워진다
                st.text_input("서브카피 (고쳐서 다시 입힐 수 있습니다)", value=res["copy"]["subcopy"],
                              max_chars=SUBCOPY_MAX, key=f"sub_{res['seed']}")
                st.form_submit_button("글자 다시 입히기", on_click=apply_subcopy, use_container_width=True)
            buf = io.BytesIO()
            res["image"].save(buf, format="PNG")
            st.download_button("이미지 다운로드", buf.getvalue(), file_name=res["path"].name, mime="image/png",
                               use_container_width=True)
        with text_col:
            st.subheader("캡션")
            st.code(res["caption"], language=None, wrap_lines=True)  # 우측 상단 아이콘으로 복사
            st.caption("일시·장소·혜택은 입력하신 글자 그대로 들어갑니다.")
            with st.expander("생성 정보"):
                st.write(res["timings"])
                st.write({"seed": res["seed"], "문구 생성 시도 횟수": res["copy"].get("attempts")})
