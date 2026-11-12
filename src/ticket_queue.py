"""대기번호 큐: GPU 하나를 여러 요청이 쓸 때 번호표를 뽑고 순서대로 처리한다."""
import threading

DEFAULT_SECONDS = 60  # 처리 시간 기록이 없을 때 예상 대기에 쓰는 값 (화면 안내문과 같은 1분)
KEEP_RECENT = 5       # 예상 대기 시간은 최근 처리 시간 몇 건의 평균으로 낸다


class TicketQueue:
    def __init__(self):
        self._cond = threading.Condition()
        self._issued = 0       # 지금까지 발급한 번호
        self._serving = 1      # 지금 차례인 번호
        self._finished = set()  # 끝났거나 기다리다 포기한 번호
        self._recent = []      # 최근 처리 시간(초)

    def take(self) -> int:
        with self._cond:
            self._issued += 1
            return self._issued

    def wait_turn(self, ticket: int, on_wait) -> None:
        """내 차례가 올 때까지 기다린다. 기다리는 동안 1초마다 on_wait(앞사람 수, 현재 번호, 예상 초)를 부른다."""
        while True:
            with self._cond:
                ahead = self._ahead(ticket)
                if ahead == 0:
                    return
                serving, eta = self._serving, ahead * self._average()
            on_wait(ahead, serving, eta)
            with self._cond:
                self._cond.wait(timeout=1.0)

    def finish(self, ticket: int, seconds: float | None = None) -> None:
        """처리를 마쳤거나 기다리다 떠난 번호를 반납한다. 반납하지 않으면 뒷사람이 영원히 기다린다."""
        with self._cond:
            self._finished.add(ticket)
            if seconds is not None:
                self._recent = (self._recent + [seconds])[-KEEP_RECENT:]
            while self._serving in self._finished:
                self._finished.discard(self._serving)
                self._serving += 1
            self._cond.notify_all()

    def _ahead(self, ticket: int) -> int:
        return sum(1 for t in range(self._serving, ticket) if t not in self._finished)

    def _average(self) -> float:
        return sum(self._recent) / len(self._recent) if self._recent else DEFAULT_SECONDS
