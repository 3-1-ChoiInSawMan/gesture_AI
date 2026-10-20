import io
import unittest

import av
import numpy as np
from faster_whisper.audio import decode_audio

from util.stt_audio import STTAudioBuffer


def make_webm():
    output = io.BytesIO()
    with av.open(output, mode="w", format="webm", options={"cluster_time_limit": "250"}) as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        for _ in range(75):
            samples = (np.sin(np.arange(960) * .05) * 10000).astype(np.int16)[None, :]
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


class STTAudioTests(unittest.TestCase):
    def test_headerless_continuation_decodes_without_repeating_samples(self):
        data = make_webm()
        first_cluster = data.index(b"\x1f\x43\xb6\x75")
        boundary = data.index(b"\x1f\x43\xb6\x75", first_cluster + 4)
        buffer = STTAudioBuffer(1024 * 1024)
        buffer.append(data[:boundary])
        first = buffer.take_audio()
        self.assertGreater(len(first), 0)
        self.assertIsNone(buffer.take_audio())
        buffer.append(data[boundary:])
        second = buffer.take_audio()
        complete = decode_audio(io.BytesIO(data))
        self.assertEqual(len(first) + len(second), len(complete))
        np.testing.assert_array_equal(second, complete[len(first):])

    def test_restarted_recorder_resets_sample_offset(self):
        data = make_webm()
        buffer = STTAudioBuffer(1024 * 1024)
        buffer.append(data)
        first = buffer.take_audio()
        buffer.append(data)
        second = buffer.take_audio()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(buffer.data), len(data))

    def test_incomplete_header_is_retained_until_more_audio_arrives(self):
        data = make_webm()
        buffer = STTAudioBuffer(1024 * 1024)
        buffer.append(data[:10])
        with self.assertRaises(Exception):
            buffer.take_audio()
        self.assertIsNone(buffer.take_audio())
        buffer.append(data[10:])
        self.assertGreater(len(buffer.take_audio()), 0)


if __name__ == "__main__":
    unittest.main()
