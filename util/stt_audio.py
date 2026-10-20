"""Keep container headers across utterance boundaries in one recorder stream."""

import io

from faster_whisper.audio import decode_audio


class STTAudioBuffer:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.data = bytearray()
        self.decoded_samples = 0
        self.pending = False

    def append(self, chunk: bytes) -> None:
        # A restarted recorder/file carries a new container header. A timeslice
        # from the same recorder usually does not; it needs the earlier bytes.
        if not self.pending and chunk.startswith((b"\x1a\x45\xdf\xa3", b"RIFF", b"fLaC", b"ID3")):
            self.clear()
        if len(self.data) + len(chunk) > self.max_bytes:
            raise BufferError("Audio stream exceeds buffer limit")
        self.data.extend(chunk)
        self.pending = True

    def take_audio(self):
        if not self.pending:
            return None
        self.pending = False
        # Decode with the original header, but never infer old samples again.
        # Retain bytes on incomplete input so the next chunk can complete it.
        audio = decode_audio(io.BytesIO(self.data), sampling_rate=16000)
        new_audio = audio[self.decoded_samples:]
        self.decoded_samples = len(audio)
        return new_audio if new_audio.size else None

    def clear(self) -> None:
        self.data.clear()
        self.decoded_samples = 0
        self.pending = False
