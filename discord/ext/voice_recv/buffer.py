# -*- coding: utf-8 -*-

from __future__ import annotations

import heapq
import logging
import threading

from .utils import gap_wrapped, add_wrapped


from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
)

from .rtp import _PacketCmpMixin

if TYPE_CHECKING:
    from typing import Optional, List
    from .rtp import AudioPacket

__all__ = [
    'HeapJitterBuffer',
    'ReorderJitterBuffer',
]


_T = TypeVar('_T')
PacketT = TypeVar('PacketT', bound=_PacketCmpMixin)


log = logging.getLogger(__name__)


class Buffer(Protocol[_T]):
    """The base class representing a simple buffer with no extra features."""

    # fmt: off
    def __len__(self) -> int: ...
    def push(self, item: _T) -> None: ...
    def pop(self) -> Optional[_T]: ...
    def peek(self) -> Optional[_T]: ...
    def flush(self) -> List[_T]: ...
    def reset(self) -> None: ...
    # fmt: on


class BaseBuffer(Buffer[PacketT]):
    """A basic buffer."""

    def __init__(self):
        self._buffer: List[PacketT] = []

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, item: PacketT) -> None:
        self._buffer.append(item)

    def pop(self) -> Optional[PacketT]:
        return self._buffer.pop()

    def peek(self) -> Optional[PacketT]:
        return self._buffer[-1] if self._buffer else None

    def flush(self) -> List[PacketT]:
        buf = self._buffer.copy()
        self._buffer.clear()
        return buf

    def reset(self) -> None:
        self._buffer.clear()


class HeapJitterBuffer(BaseBuffer[PacketT]):
    """Push item in, pop items out"""

    _threshold: int = 10000

    def __init__(self, maxsize: int = 10, *, prefsize: int = 1, prefill: int = 1):
        if maxsize < 1:
            raise ValueError(f'maxsize ({maxsize}) must be greater than 0')

        if not 0 <= prefsize <= maxsize:
            raise ValueError(f'prefsize must be between 0 and maxsize ({maxsize})')

        self.maxsize: int = maxsize
        self.prefsize: int = prefsize
        self.prefill: int = prefill
        self._prefill: int = prefill

        self._last_tx_seq: int = -1

        self._has_item: threading.Event = threading.Event()
        # I sure hope I dont need to add a lock to this
        self._buffer: List[AudioPacket] = []

    def _push(self, packet: AudioPacket) -> None:
        heapq.heappush(self._buffer, packet)

    def _pop(self) -> AudioPacket:
        return heapq.heappop(self._buffer)

    def _get_packet_if_ready(self) -> Optional[AudioPacket]:
        return self._buffer[0] if len(self._buffer) > self.prefsize else None

    def _pop_if_ready(self) -> Optional[AudioPacket]:
        return self._pop() if len(self._buffer) > self.prefsize else None

    def _update_has_item(self) -> None:
        prefilled = self._prefill == 0
        packet_ready = len(self._buffer) > self.prefsize

        if not prefilled or not packet_ready:
            self._has_item.clear()
            return

        next_packet = self._buffer[0]
        sequential = add_wrapped(self._last_tx_seq, 1) == next_packet.sequence
        positive_seq = self._last_tx_seq >= 0

        # We have the next packet ready
        # OR we havent sent a packet out yet
        # OR the buffer is full
        if (sequential and positive_seq) or not positive_seq or len(self._buffer) >= self.maxsize:
            self._has_item.set()
        else:
            self._has_item.clear()

    def _cleanup(self) -> None:
        # Logging this is pointless until I fix the stale remote buffer issue
        # if len(self._buffer) > self.maxsize:
            # log.debug("Buffer overfilled: %s > %s", len(self._buffer), self.maxsize)

        # drop oldest packets if buffer overfilled
        while len(self._buffer) > self.maxsize:
            packet = heapq.heappop(self._buffer)
            # log.debug("Dropped extra packet %s", packet)

    def push(self, packet: AudioPacket) -> bool:
        """
        Push a packet into the buffer.  If the packet would make the buffer
        exceed its maxsize, the oldest packet will be dropped.
        """

        seq = packet.sequence

        # for the gap between _last_tx_seq and the current one, a large gap is old, a small gap is new
        # the gap for old packets will generally be very large since they wrap all the way around
        if gap_wrapped(self._last_tx_seq, seq) > self._threshold and self._last_tx_seq != -1:
            log.debug("Dropping old packet %s", packet)
            return False

        self._push(packet)

        if self._prefill > 0:
            self._prefill -= 1

        self._cleanup()
        self._update_has_item()

        return True

    def pop(self, *, timeout: float | None = 0) -> Optional[AudioPacket]:
        """
        If timeout is a positive number, wait as long as timeout for a packet
        to be ready and return that packet, otherwise return None.
        """

        ok = self._has_item.wait(timeout)
        if not ok:
            return None

        if self._prefill > 0:
            return None

        # This function should actually be redundant but i'll leave it for now
        packet = self._pop_if_ready()

        if packet is not None:
            self._last_tx_seq = packet.sequence

        self._update_has_item()
        return packet

    def peek(self, *, all: bool = False) -> Optional[AudioPacket]:
        """
        Returns the next packet in the buffer only if it is ready, meaning it can
        be popped. When `all` is set to True, it returns the next packet, if any.
        """

        if not self._buffer:
            return None

        if all:
            return self._buffer[0]
        else:
            return self._get_packet_if_ready()

    def peek_next(self) -> Optional[AudioPacket]:
        """
        Returns the next packet in the buffer only if it is sequential.
        """

        packet = self.peek(all=True)

        if packet is None:
            return

        if packet.sequence == add_wrapped(self._last_tx_seq, 1) or self._last_tx_seq < 0:
            return packet

    def gap(self) -> int:
        """
        Returns the number of missing packets between the last packet to be
        popped and the currently held next packet.  Returns 0 otherwise.
        """

        if self._buffer and self._last_tx_seq > 0:
            return gap_wrapped(self._last_tx_seq, self._buffer[0].sequence)

        return 0

    def flush(self) -> List[AudioPacket]:
        """
        Return all remaining packets.
        """

        packets = sorted(self._buffer)
        self._buffer.clear()

        if packets:
            self._last_tx_seq = packets[-1].sequence

        self._prefill = self.prefill
        self._has_item.clear()

        return packets

    def reset(self) -> None:
        """
        Clear buffer and reset internal counters.
        """

        self._buffer.clear()
        self._has_item.clear()
        self._prefill = self.prefill
        self._last_tx_seq = -1


class ReorderJitterBuffer:
    """Per-SSRC jitter buffer with real reordering tolerance (0.5.6a).

    HeapJitterBuffer released a packet as soon as two were queued. A gap
    (one lost or late packet) made the decoder flush the buffer and keep
    only the first packet, discarding the rest, and the late packet was then
    dropped as "old": every loss cost an extra packet, multi-packet
    reordering discarded 6-49% of ARRIVED audio at 25-80 ms jitter, and
    nothing was ever concealed (the FakePacket/FEC path was unreachable).

    Here:
      * the next in-order packet is released at once (no added latency on a
        clean path);
      * behind a gap, packets wait until `reorder_depth` are queued or the
        oldest has waited `max_wait` seconds (time-based, so the tail of an
        utterance is never stranded) — then the gap is declared lost;
      * the first `max_conceal` missing packets of a gap become FakePackets,
        so the decoder runs FEC (from the next packet) or packet-loss
        concealment; the rest of a longer gap is skipped;
      * packets behind the release point (truly late) and duplicates are
        dropped; sequence numbers are unwrapped, so 65535 -> 0 orders
        correctly.

    pop() never blocks. peek() is truthy exactly when pop() would return
    something, so the router's ready-registration and pops always agree.
    """

    # a packet this many sequence numbers behind the release point (~2 s of
    # audio) cannot be reordering; it is a restarted stream
    RESTART_BEHIND = 100

    def __init__(self, *, reorder_depth: int = 3, max_wait: float = 0.06,
                 max_conceal: int = 3, maxsize: int = 50, clock=None):
        import time as _time
        self.reorder_depth = reorder_depth
        self.max_wait = max_wait
        self.max_conceal = max_conceal
        self.maxsize = maxsize
        self._clock = clock or _time.monotonic
        self._heap: list = []           # (ext_seq, counter, arrival, packet)
        self._counter = 0
        self._next: Optional[int] = None  # next expected extended sequence
        self._run = 0                     # frames concealed in the current gap
        self._ref: Optional[int] = None   # extended seq used to unwrap
        self.stats = {'released': 0, 'concealed': 0, 'skipped': 0,
                      'late': 0, 'duplicate': 0, 'overflow': 0, 'restarts': 0}

    def __len__(self) -> int:
        return len(self._heap)

    def _unwrap(self, seq: int) -> int:
        if self._ref is None:
            return seq
        d = (seq - (self._ref & 0xFFFF) + 0x8000) % 0x10000 - 0x8000
        return self._ref + d

    def push(self, packet: AudioPacket) -> bool:
        ext = self._unwrap(packet.sequence)
        if self._next is not None and ext < self._next:
            if self._next - ext > self.RESTART_BEHIND:
                # far behind anything plausible for reordering: the sender
                # restarted its sequence on the same SSRC. Start over
                # instead of dropping every packet as "late" forever.
                self.stats['restarts'] += 1
                self.reset()
                ext = packet.sequence
            else:
                self.stats['late'] += 1
                return False
        if any(e == ext for e, _, _, _ in self._heap):
            self.stats['duplicate'] += 1
            return False
        if self._ref is None or ext > self._ref:
            self._ref = ext
        self._counter += 1
        heapq.heappush(self._heap, (ext, self._counter, self._clock(), packet))
        while len(self._heap) > self.maxsize:      # router stalled: advance
            e, _, _, _ = heapq.heappop(self._heap)
            self.stats['overflow'] += 1
            self._next = e + 1
        return True

    def _ready(self) -> bool:
        if not self._heap:
            return False
        head = self._heap[0][0]
        if self._next is None or head == self._next:
            return True
        if len(self._heap) >= self.reorder_depth:
            return True
        oldest = min(t for _, _, t, _ in self._heap)
        return self._clock() - oldest >= self.max_wait

    def peek(self, *, all: bool = False) -> Optional[AudioPacket]:
        if not self._heap:
            return None
        if all or self._ready():
            return self._heap[0][3]
        return None

    def peek_next(self) -> Optional[AudioPacket]:
        """The next packet only if it is exactly the next in order (FEC)."""
        if self._heap and self._next is not None and self._heap[0][0] == self._next:
            return self._heap[0][3]
        return None

    def pop(self, *, timeout: Optional[float] = 0) -> Optional[AudioPacket]:
        if not self._ready():
            return None
        ext, _, _, packet = self._heap[0]
        if self._next is None or ext == self._next:
            heapq.heappop(self._heap)
            self._next = ext + 1
            self._run = 0
            self.stats['released'] += 1
            return packet
        missing = ext - self._next
        if self._run < self.max_conceal:
            # conceal the first few frames of any gap (FEC for the last one
            # before the head, PLC fade for the rest), then skip the rest
            from .rtp import FakePacket
            self._run += 1
            ts = (packet.timestamp - missing * 960) % 2**32
            fake = FakePacket(packet.ssrc, self._next & 0xFFFF, ts)
            self._next += 1
            self.stats['concealed'] += 1
            return fake
        self.stats['skipped'] += missing
        self._next = ext
        return self.pop()

    def gap(self) -> int:
        if self._heap and self._next is not None:
            return max(0, self._heap[0][0] - self._next)
        return 0

    def flush(self) -> List[AudioPacket]:
        packets = [p for _, _, _, p in sorted(self._heap)]
        if self._heap:
            self._next = max(e for e, _, _, _ in self._heap) + 1
        self._heap.clear()
        return packets

    def reset(self) -> None:
        self._heap.clear()
        self._next = None
        self._ref = None
        self._run = 0
