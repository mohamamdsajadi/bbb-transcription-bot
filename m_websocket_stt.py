# m_websocket_stt.py
import json
from typing import List, Optional

from websocket import create_connection

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule, Status
from stream_pipeline.module_classes import Module, ModuleOptions

import data
import logger

log = logger.get_logger()

class WebSocket_STT(Module):
    def __init__(self, ws_url: str = "ws://localhost:2700") -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="WebSocket-STT-Module",
        )
        self.ws_url: str = ws_url

    def init_module(self) -> None:
        pass

    def execute(self, dp: DataPackage[data.AudioData], dpc: DataPackageController, dpp: DataPackagePhase, dpm: DataPackageModule) -> None:
        if not dp.data:
            raise Exception("No data found")
        if dp.data.raw_audio_data is None:
            raise Exception("No audio data found")

        log.debug("Sending audio data to external STT")
        try:
            ws = create_connection(self.ws_url, timeout=self.timeout)
            ws.send(dp.data.raw_audio_data, opcode=0x2)  # binary
            message = ws.recv()
            ws.close()
        except Exception as e:
            dpm.status = Status.EXIT
            dpm.message = f"WebSocket error: {e}"
            log.error(f"WebSocket error: {e}")
            return

        try:
            response = json.loads(message)
        except json.JSONDecodeError as e:
            dpm.status = Status.EXIT
            dpm.message = f"Invalid JSON response: {e}"
            log.error("Invalid JSON response from STT server")
            return

        segments: List[data.TextSegment] = []
        start_offset = dp.data.audio_buffer_start_after or 0.0
        for seg in response.get("segments", []):
            words: Optional[List[data.Word]] = None
            word_items = seg.get("words")
            if isinstance(word_items, list):
                words = []
                for w in word_items:
                    words.append(
                        data.Word(
                            word=w.get("word", ""),
                            start=float(w.get("start", 0.0)) + start_offset,
                            end=float(w.get("end", 0.0)) + start_offset,
                            probability=float(w.get("probability", 1.0)),
                        )
                    )
            ts = data.TextSegment(
                text=seg.get("text", ""),
                start=float(seg.get("start", 0.0)) + start_offset,
                end=float(seg.get("end", 0.0)) + start_offset,
                words=words,
            )
            segments.append(ts)

        dp.data.transcribed_segments = segments
