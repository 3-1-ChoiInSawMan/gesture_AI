import asyncio
import html
import json
import time
from dataclasses import dataclass
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
import websockets
from websockets.exceptions import ConnectionClosed


@dataclass
class ReceiveEvent:
    elapsed: float
    message: str


@dataclass
class SendResult:
    ok: bool
    error: str | None
    url: str
    bytes_sent: int
    chunks_sent: int
    elapsed: float
    events: list[ReceiveEvent]
    trace: list[str]


def _build_ws_url(base_url: str, session_id: str, debug: bool) -> str:
    url = base_url.strip()
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}session_id={session_id}"
    if debug:
        url = f"{url}&debug=1"
    return url


async def _receive_until_idle(ws, idle_timeout: float, total_timeout: float, trace: list[str]) -> list[ReceiveEvent]:
    events: list[ReceiveEvent] = []
    started_at = time.monotonic()

    while True:
        remaining_total = total_timeout - (time.monotonic() - started_at)
        if remaining_total <= 0:
            trace.append("receive: total timeout reached")
            return events

        timeout = min(idle_timeout, remaining_total)
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except TimeoutError:
            trace.append("receive: idle timeout reached")
            return events
        except ConnectionClosed as exc:
            trace.append(f"receive: connection closed code={exc.code} reason={exc.reason!r}")
            return events

        elapsed = time.monotonic() - started_at
        if isinstance(message, bytes):
            text = f"<binary message: {len(message)} bytes>"
        else:
            text = message
        events.append(ReceiveEvent(elapsed=elapsed, message=text))
        trace.append(f"receive: {text!r}")


async def _send_audio(
    url: str,
    audio_bytes: bytes,
    chunk_size: int,
    chunk_delay: float,
    receive_idle_timeout: float,
    receive_total_timeout: float,
    ping_interval: float | None,
) -> SendResult:
    trace: list[str] = []
    chunks_sent = 0
    started_at = time.monotonic()

    try:
        trace.append(f"connect: {url}")
        async with websockets.connect(url, ping_interval=ping_interval) as ws:
            trace.append("connect: ok")

            for offset in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[offset : offset + chunk_size]
                await ws.send(chunk)
                chunks_sent += 1
                trace.append(f"send: chunk={chunks_sent} bytes={len(chunk)} offset={offset}")
                if chunk_delay > 0:
                    await asyncio.sleep(chunk_delay)

            trace.append("send: complete")
            events = await _receive_until_idle(
                ws,
                idle_timeout=receive_idle_timeout,
                total_timeout=receive_total_timeout,
                trace=trace,
            )

        elapsed = time.monotonic() - started_at
        return SendResult(
            ok=True,
            error=None,
            url=url,
            bytes_sent=len(audio_bytes),
            chunks_sent=chunks_sent,
            elapsed=elapsed,
            events=events,
            trace=trace,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        trace.append(f"error: {type(exc).__name__}: {exc}")
        return SendResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            url=url,
            bytes_sent=len(audio_bytes),
            chunks_sent=chunks_sent,
            elapsed=elapsed,
            events=[],
            trace=trace,
        )


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def _render_live_browser_stream(url: str, timeslice_ms: int, mime_type: str) -> None:
    escaped_url = json.dumps(url)
    escaped_mime_type = json.dumps(mime_type)
    title = html.escape(url)
    component_html = f"""
<!doctype html>
<html>
<head>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      color: #172026;
      background: #ffffff;
    }}
    .wrap {{
      display: grid;
      gap: 12px;
      padding: 2px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    button {{
      border: 1px solid #b7c0c7;
      border-radius: 6px;
      background: #ffffff;
      color: #172026;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      min-height: 36px;
      padding: 0 12px;
    }}
    button.primary {{
      border-color: #176b87;
      background: #176b87;
      color: #ffffff;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    .status {{
      border: 1px solid #cdd5dc;
      border-radius: 6px;
      min-height: 38px;
      padding: 9px 10px;
      font-size: 14px;
      background: #f7f9fa;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      border: 1px solid #d6dde2;
      border-radius: 6px;
      padding: 10px;
      min-width: 0;
    }}
    .metric span {{
      display: block;
      color: #5f6f7a;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
      overflow-wrap: anywhere;
    }}
    .panel {{
      border: 1px solid #d6dde2;
      border-radius: 6px;
      min-height: 140px;
      padding: 10px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #ffffff;
      font-size: 14px;
      line-height: 1.45;
    }}
    .log {{
      background: #111820;
      color: #d7e2ea;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      min-height: 190px;
    }}
    @media (max-width: 720px) {{
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <button id="start" class="primary">Start mic stream</button>
      <button id="stop" disabled>Stop</button>
      <button id="clear">Clear</button>
    </div>
    <div id="status" class="status">Ready: {title}</div>
    <div class="grid">
      <div class="metric"><span>Chunks sent</span><strong id="chunks">0</strong></div>
      <div class="metric"><span>Bytes sent</span><strong id="bytes">0</strong></div>
      <div class="metric"><span>Messages</span><strong id="messages">0</strong></div>
      <div class="metric"><span>Recorder MIME</span><strong id="mime">-</strong></div>
    </div>
    <div>
      <h4>Received text</h4>
      <div id="received" class="panel"></div>
    </div>
    <div>
      <h4>Trace</h4>
      <div id="log" class="panel log"></div>
    </div>
  </div>

  <script>
    const wsUrl = {escaped_url};
    const requestedMimeType = {escaped_mime_type};
    const timesliceMs = {int(timeslice_ms)};

    let ws = null;
    let recorder = null;
    let mediaStream = null;
    let chunksSent = 0;
    let bytesSent = 0;
    let messages = 0;

    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    const clearButton = document.getElementById("clear");
    const statusEl = document.getElementById("status");
    const chunksEl = document.getElementById("chunks");
    const bytesEl = document.getElementById("bytes");
    const messagesEl = document.getElementById("messages");
    const mimeEl = document.getElementById("mime");
    const receivedEl = document.getElementById("received");
    const logEl = document.getElementById("log");

    function setStatus(text) {{
      statusEl.textContent = text;
    }}

    function log(text) {{
      const now = new Date().toLocaleTimeString();
      logEl.textContent += `[${{now}}] ${{text}}\\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }}

    function refreshMetrics() {{
      chunksEl.textContent = String(chunksSent);
      bytesEl.textContent = String(bytesSent);
      messagesEl.textContent = String(messages);
    }}

    function pickMimeType() {{
      const candidates = [];
      if (requestedMimeType) candidates.push(requestedMimeType);
      candidates.push("audio/webm;codecs=opus");
      candidates.push("audio/webm");
      candidates.push("audio/ogg;codecs=opus");
      candidates.push("");
      for (const candidate of candidates) {{
        if (!candidate || MediaRecorder.isTypeSupported(candidate)) {{
          return candidate;
        }}
      }}
      return "";
    }}

    async function stopAll() {{
      if (recorder && recorder.state !== "inactive") {{
        recorder.stop();
      }}
      if (mediaStream) {{
        for (const track of mediaStream.getTracks()) track.stop();
      }}
      if (ws && ws.readyState === WebSocket.OPEN) {{
        ws.close(1000, "client stop");
      }}
      recorder = null;
      mediaStream = null;
      ws = null;
      startButton.disabled = false;
      stopButton.disabled = true;
      setStatus("Stopped");
    }}

    startButton.addEventListener("click", async () => {{
      chunksSent = 0;
      bytesSent = 0;
      messages = 0;
      refreshMetrics();
      receivedEl.textContent = "";
      logEl.textContent = "";
      startButton.disabled = true;
      stopButton.disabled = false;

      try {{
        setStatus("Requesting microphone...");
        mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        log("microphone: ok");

        setStatus("Connecting WebSocket...");
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {{
          const mimeType = pickMimeType();
          const options = mimeType ? {{ mimeType }} : undefined;
          recorder = new MediaRecorder(mediaStream, options);
          mimeEl.textContent = recorder.mimeType || "-";

          recorder.ondataavailable = async (event) => {{
            if (!event.data || event.data.size === 0) return;
            if (!ws || ws.readyState !== WebSocket.OPEN) {{
              log(`skip: websocket not open, blob bytes=${{event.data.size}}`);
              return;
            }}
            const buffer = await event.data.arrayBuffer();
            ws.send(buffer);
            chunksSent += 1;
            bytesSent += buffer.byteLength;
            refreshMetrics();
            log(`send: chunk=${{chunksSent}} bytes=${{buffer.byteLength}}`);
          }};

          recorder.onstart = () => {{
            setStatus("Streaming microphone audio...");
            log(`recorder: start timeslice=${{timesliceMs}}ms mime=${{recorder.mimeType || "-"}}`);
          }};

          recorder.onstop = () => {{
            log("recorder: stop");
          }};

          recorder.onerror = (event) => {{
            log(`recorder error: ${{event.error && event.error.message ? event.error.message : event.error}}`);
          }};

          recorder.start(timesliceMs);
          log("websocket: open");
        }};

        ws.onmessage = (event) => {{
          messages += 1;
          refreshMetrics();
          const text = typeof event.data === "string" ? event.data : `<binary message: ${{event.data.byteLength}} bytes>`;
          receivedEl.textContent += text + "\\n";
          receivedEl.scrollTop = receivedEl.scrollHeight;
          log(`receive: ${{JSON.stringify(text)}}`);
        }};

        ws.onerror = () => {{
          log("websocket: error");
          setStatus("WebSocket error");
        }};

        ws.onclose = (event) => {{
          log(`websocket: close code=${{event.code}} reason=${{JSON.stringify(event.reason)}}`);
          stopAll();
        }};
      }} catch (error) {{
        log(`error: ${{error.name || "Error"}}: ${{error.message || error}}`);
        setStatus(`Error: ${{error.message || error}}`);
        await stopAll();
      }}
    }});

    stopButton.addEventListener("click", stopAll);

    clearButton.addEventListener("click", () => {{
      receivedEl.textContent = "";
      logEl.textContent = "";
      chunksSent = 0;
      bytesSent = 0;
      messages = 0;
      refreshMetrics();
    }});
  </script>
</body>
</html>
"""
    components.html(component_html, height=690, scrolling=True)


st.set_page_config(page_title="CC STT WebSocket Debug", layout="wide")
st.title("CC STT WebSocket Debug")

with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("WebSocket URL", value="ws://localhost:8000/cc_stt")
    session_id = st.text_input("session_id", value=f"st-debug-{uuid4()}")
    debug_param = st.checkbox("Append debug=1", value=False)

    st.header("Streaming")
    test_mode = st.radio("Mode", ["Live microphone", "Audio file"], index=0)
    chunk_size = st.number_input("Chunk size bytes", min_value=256, max_value=262_144, value=16_384, step=1024)
    chunk_delay_ms = st.number_input("Delay per chunk ms", min_value=0, max_value=2000, value=50, step=10)
    live_timeslice_ms = st.number_input("Live chunk ms", min_value=100, max_value=5000, value=1000, step=100)
    live_mime_type = st.selectbox(
        "Live MIME type",
        ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "Browser default"],
    )
    receive_idle_timeout = st.number_input("Receive idle timeout sec", min_value=0.5, max_value=60.0, value=5.0, step=0.5)
    receive_total_timeout = st.number_input("Receive total timeout sec", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    ping_interval_enabled = st.checkbox("Enable websocket ping", value=True)

live_url = _build_ws_url(base_url, session_id, debug_param)

if test_mode == "Live microphone":
    st.caption("Browser microphone audio is sent directly from this page to the FastAPI WebSocket as binary frames.")
    mime_type = "" if live_mime_type == "Browser default" else live_mime_type
    _render_live_browser_stream(live_url, int(live_timeslice_ms), mime_type)
    st.stop()

audio_source = st.radio("Audio source", ["Upload file", "Record in browser"], horizontal=True)
uploaded_file = None

if audio_source == "Upload file":
    uploaded_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "webm", "ogg", "flac"])
else:
    audio_input = getattr(st, "audio_input", None)
    if audio_input is None:
        st.warning("This Streamlit version does not have st.audio_input. Use file upload instead.")
    else:
        uploaded_file = audio_input("Record audio")

if uploaded_file is None:
    st.info("Upload an audio file, then click Send audio.")
    st.stop()

audio_bytes = uploaded_file.getvalue()
st.audio(audio_bytes)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("File", uploaded_file.name)
col_b.metric("Size", _format_bytes(len(audio_bytes)))
col_c.metric("Chunk size", f"{int(chunk_size)} B")
estimated_chunks = (len(audio_bytes) + int(chunk_size) - 1) // int(chunk_size)
col_d.metric("Estimated chunks", estimated_chunks)

send_clicked = st.button("Send audio", type="primary", use_container_width=True)

if send_clicked:
    url = _build_ws_url(base_url, session_id, debug_param)
    ping_interval = 20.0 if ping_interval_enabled else None

    with st.spinner("Streaming audio to WebSocket..."):
        result = asyncio.run(
            _send_audio(
                url=url,
                audio_bytes=audio_bytes,
                chunk_size=int(chunk_size),
                chunk_delay=float(chunk_delay_ms) / 1000.0,
                receive_idle_timeout=float(receive_idle_timeout),
                receive_total_timeout=float(receive_total_timeout),
                ping_interval=ping_interval,
            )
        )

    if result.ok:
        st.success("WebSocket test finished.")
    else:
        st.error(result.error)

    summary_a, summary_b, summary_c, summary_d = st.columns(4)
    summary_a.metric("Sent bytes", _format_bytes(result.bytes_sent))
    summary_b.metric("Sent chunks", result.chunks_sent)
    summary_c.metric("Elapsed", f"{result.elapsed:.2f}s")
    summary_d.metric("Messages", len(result.events))

    st.subheader("Received text")
    if result.events:
        transcript = "\n".join(event.message for event in result.events)
        st.text_area("Messages joined", value=transcript, height=180)
        st.dataframe(
            [{"elapsed_sec": round(event.elapsed, 3), "message": event.message} for event in result.events],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No messages received before timeout.")

    st.subheader("Trace")
    st.code("\n".join(result.trace), language="text")
