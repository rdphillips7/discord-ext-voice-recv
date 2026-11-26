"""Tests for the 0.5.6a reorder jitter buffer and the decoder/router changes.

Run from the repo root (NOT from inside discord/ext/voice_recv, whose
types.py shadows the stdlib):  python -m pytest tests -q
Needs libopus (the FEC tests decode real Opus).
"""
import struct
import threading
import time
import types
import unittest

import discord.opus as dopus
from discord.ext.voice_recv import rtp, utils
from discord.ext.voice_recv.buffer import ReorderJitterBuffer
from discord.ext.voice_recv import opus as vopus
from discord.ext.voice_recv.router import PacketRouter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def pkt(seq, ssrc=1, payload=b"\0" * 60, ts=None):
    ts = (seq * 960) % 2**32 if ts is None else ts
    p = rtp.decode_rtp(struct.pack(">BBHII", 0x80, 0x78, seq & 0xFFFF, ts, ssrc) + payload)
    p.decrypted_data = payload
    return p


def drain(b):
    out = []
    while True:
        p = b.pop()
        if p is None:
            return out
        out.append(("F" if not p else "P", p.sequence))


class BufferTests(unittest.TestCase):
    def setUp(self):
        self.clk = Clock()
        self.b = ReorderJitterBuffer(clock=self.clk)

    def test_in_order_released_immediately(self):
        for s in range(10, 15):
            self.b.push(pkt(s))
            self.assertTrue(self.b.peek())
            self.assertEqual(drain(self.b), [("P", s)])

    def test_reorder_within_depth_loses_nothing(self):
        # 10, 12, 13, 11, 14: the old buffer flushed and dropped 11 and 13
        out = []
        for s in (10, 12, 13, 11, 14):
            self.b.push(pkt(s))
            out += drain(self.b)
        self.assertEqual(out, [("P", s) for s in range(10, 15)])
        self.assertEqual(self.b.stats["late"], 0)
        self.assertEqual(self.b.stats["concealed"], 0)

    def test_held_behind_gap_until_depth(self):
        self.b.push(pkt(10)); drain(self.b)
        self.b.push(pkt(12))
        self.assertIsNone(self.b.peek())          # waiting for 11
        self.b.push(pkt(13))
        self.assertIsNone(self.b.peek())
        self.b.push(pkt(14))                      # depth 3 reached: 11 is lost
        self.assertTrue(self.b.peek())
        self.assertEqual(drain(self.b), [("F", 11), ("P", 12), ("P", 13), ("P", 14)])
        self.assertEqual(self.b.stats["concealed"], 1)

    def test_tail_released_by_time(self):
        # end of an utterance: 10, (11 lost), 12 and then nothing more
        self.b.push(pkt(10)); drain(self.b)
        self.b.push(pkt(12))
        self.clk.t = 0.059
        self.assertIsNone(self.b.peek())
        self.clk.t = 0.0601
        self.assertTrue(self.b.peek())
        self.assertEqual(drain(self.b), [("F", 11), ("P", 12)])

    def test_late_packet_dropped_after_conceal(self):
        self.b.push(pkt(10)); drain(self.b)
        for s in (12, 13, 14):
            self.b.push(pkt(s))
        drain(self.b)
        self.assertFalse(self.b.push(pkt(11)))
        self.assertEqual(self.b.stats["late"], 1)
        self.assertEqual(drain(self.b), [])

    def test_duplicates_dropped(self):
        self.b.push(pkt(10)); drain(self.b)
        self.b.push(pkt(12))
        self.assertFalse(self.b.push(pkt(12)))
        self.assertFalse(self.b.push(pkt(10)))
        self.assertEqual(self.b.stats["duplicate"], 1)
        self.assertEqual(self.b.stats["late"], 1)

    def test_long_gap_conceals_three_then_skips(self):
        self.b.push(pkt(10)); drain(self.b)
        for s in (20, 21, 22):
            self.b.push(pkt(s))
        out = drain(self.b)
        self.assertEqual(out, [("F", 11), ("F", 12), ("F", 13), ("P", 20), ("P", 21), ("P", 22)])
        self.assertEqual(self.b.stats["skipped"], 6)

    def test_second_gap_concealed_after_first(self):
        self.b.push(pkt(10)); drain(self.b)
        for s in (13, 14, 15):                    # gap of 2
            self.b.push(pkt(s))
        for s in (18, 19, 20):                    # another gap of 2
            self.b.push(pkt(s))
        out = drain(self.b)
        self.assertEqual([k for k, _ in out].count("F"), 4)
        self.assertEqual(self.b.stats["skipped"], 0)

    def test_fec_source_only_when_adjacent(self):
        self.b.push(pkt(10)); drain(self.b)
        for s in (20, 21, 22):
            self.b.push(pkt(s))
        self.assertFalse(self.b.pop())            # fake 11
        self.assertIsNone(self.b.peek_next())     # 20 holds FEC for 19, not 12
        self.b.pop(); self.b.pop()                # fakes 12, 13; rest skipped
        self.assertEqual(self.b.pop().sequence, 20)
        b = ReorderJitterBuffer(clock=self.clk)
        b.push(pkt(10)); b.pop()
        for s in (12, 13, 14):
            b.push(pkt(s))
        self.assertFalse(b.pop())                 # fake 11
        # after the fake, 12 is next in order -> its FEC is used
        self.assertEqual(b.peek_next().sequence, 12)

    def test_sequence_wraparound(self):
        out = []
        for s in (65533, 65534, 0, 65535, 1, 2):
            self.b.push(pkt(s))
            out += drain(self.b)
        self.assertEqual([s for _, s in out], [65533, 65534, 65535, 0, 1, 2])
        self.assertEqual(self.b.stats["late"] + self.b.stats["concealed"], 0)

    def test_wraparound_gap_conceals_correct_seq(self):
        self.b.push(pkt(65534)); drain(self.b)
        for s in (1, 2, 3):
            self.b.push(pkt(s))
        self.assertEqual(drain(self.b), [("F", 65535), ("F", 0), ("P", 1), ("P", 2), ("P", 3)])

    def test_fake_packet_timestamp(self):
        self.b.push(pkt(10)); drain(self.b)
        self.b.push(pkt(12)); self.clk.t = 1
        f = self.b.pop()
        self.assertFalse(f)
        self.assertEqual((f.sequence, f.timestamp, f.ssrc), (11, 11 * 960, 1))

    def test_stream_restart_on_same_ssrc(self):
        for s in range(5000, 5005):
            self.b.push(pkt(s)); drain(self.b)
        out = []
        for s in range(3, 8):                     # sender restarted at 3
            self.b.push(pkt(s)); out += drain(self.b)
        self.assertEqual([s for _, s in out], list(range(3, 8)))
        self.assertEqual(self.b.stats["restarts"], 1)

    def test_small_backwards_step_is_late_not_restart(self):
        for s in range(5000, 5005):
            self.b.push(pkt(s)); drain(self.b)
        self.b.push(pkt(4990))
        self.assertEqual((self.b.stats["late"], self.b.stats["restarts"]), (1, 0))

    def test_peek_agrees_with_pop(self):
        import random
        rnd = random.Random(1)
        seqs = list(range(100, 400))
        for i in range(0, len(seqs) - 3, 3):
            if rnd.random() < 0.5:
                seqs[i], seqs[i + 2] = seqs[i + 2], seqs[i]
        seqs = [s for s in seqs if rnd.random() > 0.05]
        for s in seqs:
            self.b.push(pkt(s))
            self.clk.t += 0.02
            while True:
                ready = bool(self.b.peek())
                p = self.b.pop()
                self.assertEqual(ready, p is not None)
                if p is None:
                    break

    def test_maxsize_bounds_memory(self):
        b = ReorderJitterBuffer(clock=self.clk, maxsize=50, reorder_depth=10**6, max_wait=10**6)
        b.push(pkt(0)); b.pop()
        for s in range(2, 200):
            b.push(pkt(s))
        self.assertLessEqual(len(b), 50)
        self.assertGreater(b.stats["overflow"], 0)

    def test_flush_and_reset(self):
        self.b.push(pkt(10)); drain(self.b)
        self.b.push(pkt(13)); self.b.push(pkt(12))
        self.assertEqual([p.sequence for p in self.b.flush()], [12, 13])
        self.assertEqual(len(self.b), 0)
        self.b.reset()
        self.b.push(pkt(7))
        self.assertEqual(drain(self.b), [("P", 7)])


class MultiDataEventTests(unittest.TestCase):
    def test_register_idempotent_and_unregister_all(self):
        ev = utils.MultiDataEvent()
        a = object()
        ev.register(a); ev.register(a); ev.register(a)
        self.assertEqual(ev.items, [a])
        ev.unregister(a)
        self.assertEqual(ev.items, [])
        self.assertFalse(ev.is_ready())


# ----------------------------------------------------------- decoder/router
class FakeDave:
    ready = True

    def __init__(self):
        self.calls = []

    def set_passthrough_mode(self, *a):
        pass

    def decrypt(self, uid, media, data):
        self.calls.append(bytes(data))
        if not data.startswith(b"DAVE"):
            raise ValueError("not dave")
        return data[4:]


class Member:
    id = 42


def make_env(dave=None, wants_opus=False):
    written = []
    conn = types.SimpleNamespace(dave_session=dave)
    vc = types.SimpleNamespace(
        _connection=conn,
        guild=types.SimpleNamespace(get_member=lambda uid: Member() if uid == 42 else None),
        client=types.SimpleNamespace(get_user=lambda uid: None),
        _get_id_from_ssrc=lambda ssrc: 42,
        stop_listening=lambda: None,
    )
    sink = types.SimpleNamespace(
        voice_client=vc,
        wants_opus=lambda: wants_opus,
        write=lambda user, data: written.append(data),
    )
    reader = types.SimpleNamespace(dave_failures=0, voice_client=vc, error=None)
    router = PacketRouter(sink, reader)
    return router, written, reader


def opus_frames(n, fec=True):
    enc = dopus.Encoder()
    enc.set_fec(fec)
    enc.set_expected_packet_loss_percent(0.15)
    import math
    frames = []
    for i in range(n):
        pcm = b"".join(
            struct.pack("<hh", v, v)
            for v in (int(8000 * math.sin(2 * math.pi * 440 * (i * 960 + k) / 48000)) for k in range(960))
        )
        frames.append(enc.encode(pcm, 960))
    return frames


@unittest.skipUnless(dopus.is_loaded() or dopus._load_default(), "libopus not available")
class DecoderRouterTests(unittest.TestCase):
    def run_router(self, router, feed, written=None, expect=None):
        """Feed, then wait. With `expect`, also record how long it took
        (while the router is still running) to write that many items."""
        router.start()
        took = None
        try:
            t0 = time.monotonic()
            feed()
            deadline = t0 + 1.0
            while time.monotonic() < deadline:
                if expect is not None and took is None and len(written) >= expect:
                    took = time.monotonic() - t0
                time.sleep(0.005)
        finally:
            router.stop()
            router.join(1)
        return took

    def test_loss_concealed_with_fec_through_real_decoder(self):
        router, written, _ = make_env()
        frames = opus_frames(12)

        def feed():
            for s in (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11):   # 3 lost
                router.feed_rtp(pkt(s, ssrc=7, payload=frames[s]))
                time.sleep(0.005)
        self.run_router(router, feed)
        seqs = [d.packet.sequence for d in written]
        self.assertEqual(seqs, list(range(12)))
        fake = written[3]
        self.assertFalse(fake.packet)
        self.assertEqual(len(fake.pcm), 3840)               # 20 ms stereo PCM
        self.assertTrue(any(fake.pcm))                      # not silence
        # and it is the FEC reconstruction from frame 4, not plain PLC
        ref = dopus.Decoder()
        for s in range(3):
            ref.decode(frames[s], fec=False)
        self.assertEqual(fake.pcm, ref.decode(frames[4], fec=True))
        plc = dopus.Decoder()
        for s in range(3):
            plc.decode(frames[s], fec=False)
        self.assertNotEqual(fake.pcm, plc.decode(None, fec=False))
        self.assertTrue(all(len(d.pcm) == 3840 for d in written))

    def test_tail_after_gap_delivered_without_more_packets(self):
        router, written, _ = make_env()
        frames = opus_frames(4)

        def feed():
            for s in (0, 1, 3):                              # 2 lost, then silence
                router.feed_rtp(pkt(s, ssrc=7, payload=frames[s]))
        took = self.run_router(router, feed, written, expect=4)
        self.assertEqual([d.packet.sequence for d in written], [0, 1, 2, 3])
        # released by the router's own timed re-check (~60-80 ms), not by
        # shutting down
        self.assertIsNotNone(took)
        self.assertLess(took, 0.25)

    def test_router_does_not_spin(self):
        router, written, _ = make_env()
        frames = opus_frames(3)
        calls = [0]
        orig = vopus.PacketDecoder.pop_data

        def counting(self, **kw):
            calls[0] += 1
            return orig(self, **kw)
        vopus.PacketDecoder.pop_data = counting
        try:
            def feed():
                for s in range(3):
                    router.feed_rtp(pkt(s, ssrc=7, payload=frames[s]))
            self.run_router(router, feed)
        finally:
            vopus.PacketDecoder.pop_data = orig
        self.assertEqual(len(written), 3)
        self.assertLess(calls[0], 20)                        # ~1 s idle afterwards

    def test_dave_fec_decrypts_next_packet_once(self):
        dave = FakeDave()
        router, written, reader = make_env(dave=dave)
        frames = opus_frames(8)

        def feed():
            for s in (0, 1, 3, 4, 5, 6, 7):                  # 2 lost
                router.feed_rtp(pkt(s, ssrc=7, payload=b"DAVE" + frames[s]))
                time.sleep(0.005)
        self.run_router(router, feed)
        self.assertEqual([d.packet.sequence for d in written], list(range(8)))
        self.assertEqual(len(dave.calls), 7)                 # one per real packet
        self.assertIsNone(router.decoders[7]._dave_plain)    # cache consumed
        self.assertEqual(reader.dave_failures, 0)
        self.assertTrue(any(written[2].pcm))
        self.assertTrue(all(len(d.pcm) == 3840 for d in written))

    def test_fake_packet_never_sent_to_dave(self):
        dave = FakeDave()
        router, written, reader = make_env(dave=dave)
        frames = opus_frames(4)

        def feed():
            for s in (0, 3):                                 # 1, 2 lost; tail by time
                router.feed_rtp(pkt(s, ssrc=7, payload=b"DAVE" + frames[s]))
        self.run_router(router, feed)
        self.assertEqual([d.packet.sequence for d in written], [0, 1, 2, 3])
        self.assertTrue(all(c.startswith(b"DAVE") for c in dave.calls))
        self.assertEqual(reader.dave_failures, 0)

    def test_buffer_stats_survive_decoder_destroy(self):
        router, written, _ = make_env()
        frames = opus_frames(6)

        def feed():
            for s in (0, 1, 3, 4, 5):
                router.feed_rtp(pkt(s, ssrc=7, payload=frames[s]))
        self.run_router(router, feed)
        router.destroy_decoder(7)
        st = router.buffer_stats()
        self.assertEqual((st["released"], st["concealed"]), (5, 1))


if __name__ == "__main__":
    unittest.main()
