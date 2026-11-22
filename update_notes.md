# Update notes
Notably, not a changelog, just notes.

## 0.5.6a
- New default jitter buffer, `ReorderJitterBuffer`. The old `HeapJitterBuffer` released a packet as soon as two were queued. Any gap made the decoder flush the buffer and keep only the first packet, so each loss cost extra packets and reordered packets were dropped. FEC/PLC never ran.
  - The next in-order packet is released at once (no added delay on a clean path).
  - Behind a gap, packets wait until 3 are queued or the oldest has waited 60 ms, then the gap is declared lost.
  - The first 3 missing frames of a gap are decoded with FEC (from the next packet) or PLC; the rest of a longer gap is skipped.
  - Sequence numbers are unwrapped (65535 -> 0); a stream restarted on the same SSRC is detected.
- The packet router re-checks held packets every 20 ms, so the tail of an utterance after a gap is still delivered.
- With DAVE, the packet used as the FEC source is decrypted once and cached.
- `MultiDataEvent.register` is idempotent (fixes a possible busy loop).
- `AudioReader.health()['jitter']`: released / concealed / skipped / late / duplicate / overflow / restarts.

## 0.5.2
- Adds `extras.localplayback` module
- Adds info about the extras modules to the readme
- Adds `WavSink` as an alias to `WaveSink`
- Fixed a member cleanup error in SpeechRecognitionSink
- Changes the optional dependency format
  - Previously it was a single optional dep, `extras`.  Now there is a dependency per module, with `extras` installing all of them.  See the readme for details.

## 0.5.1
- Fixes a build process related error
- Changes `voice_recv.extras` import semantics
  - The `__all__` contents of the extras modules are no longer `*` imported into `voice_recv.extras` (this was only `extras.SpeechRecognitionSink`).  You will have to access them directly, or import that specific extra module.  Example:
    ```py
    from discord.ext.voice_recv.extras.speechrecognition import SpeechRecognitionSink
    # or
    from discord.ext.voice_recv.extras import speechrecognition
    sink = speechrecognition.SpeechRecognitionSink(...)
    ```
