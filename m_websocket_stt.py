# m_websocket_stt.py
import json
import os
from typing import List, Optional

import websocket

from stream_pipeline.data_package import DataPackage, DataPackageController, DataPackagePhase, DataPackageModule
from stream_pipeline.module_classes import Module, ModuleOptions

import logger
import data

log = logger.get_logger()


class WebSocket_STT(Module):
    """Stream buffered audio to an external STT engine over WebSocket."""

    def __init__(self, url: Optional[str] = None) -> None:
        super().__init__(
            ModuleOptions(
                use_mutex=True,
                timeout=5,
            ),
            name="WebSocket-STT-Module",
        )
        self.url: str = url or os.getenv("STT_WS_URL", "")

    def init_module(self) -> None:
        if not self.url:
            raise Exception("STT_WS_URL not configured")

    def execute(
        self,
        dp: DataPackage[data.AudioData],
        dpc: DataPackageController,
        dpp: DataPackagePhase,
        dpm: DataPackageModule,
    ) -> None:
        if not dp.data or dp.data.raw_audio_data is None:
            raise Exception("No audio data found")
        if dp.data.audio_buffer_start_after is None:
            raise Exception("No audio buffer start time found")

        audio_buffer_start_after = dp.data.audio_buffer_start_after

        try:
            ws = websocket.create_connection(self.url)
            ws.send_binary(dp.data.raw_audio_data)
            response = ws.recv()
            ws.close()
        except Exception as e:
            raise Exception(f"WebSocket STT error: {e}")

        try:
            payload = json.loads(response)
        except Exception as e:
            raise Exception(f"Invalid STT response: {e}")

        segments: List[data.TextSegment] = []
        if isinstance(payload, dict) and "segments" in payload:
            for seg in payload["segments"]:
                words: List[data.Word] = []
                if seg.get("words"):
                    for w in seg["words"]:
                        words.append(
                            data.Word(
                                word=str(w.get("word", "")),
                                start=float(w.get("start", 0.0)) + audio_buffer_start_after,
                                end=float(w.get("end", 0.0)) + audio_buffer_start_after,
                                probability=float(w.get("probability", 1.0)),
                            )
                        )
                ts = data.TextSegment(
                    text=str(seg.get("text", "")),
                    start=float(seg.get("start", 0.0)) + audio_buffer_start_after,
                    end=float(seg.get("end", 0.0)) + audio_buffer_start_after,
                    words=words,
                )
                segments.append(ts)
        else:
            log.warning("STT response did not contain segments")

        dp.data.transcribed_segments = segments
