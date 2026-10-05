"""Per-speaker Ogg Opus tracks: container writing, RTP timeline placement, and decoding for STT.

Discord delivers each speaker as a stream of 20 ms Opus packets with a 48 kHz RTP timestamp.
The packets are written to an Ogg Opus file as they arrive (no re-encoding), and gaps in the
timestamp are filled with Opus silence frames, so position N seconds in the file is N seconds
after the speaker's first packet. Nothing here needs libopus: packets are copied, not decoded.
"""
from __future__ import annotations

import logging
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable

log = logging.getLogger(__name__)

OPUS_SAMPLE_RATE = 48000
OPUS_CHANNELS = 2
FRAME_SAMPLES = 960  # 20 ms, what Discord clients send
# Each track starts with this much silence, declared as Opus pre-skip (RFC 7845 recommends at
# least 80 ms so the decoder converges), so decoded sample N is still N/48000 s into the track.
PRE_SKIP_SAMPLES = 4 * FRAME_SAMPLES
# A valid 20 ms Opus frame that decodes to silence (the same frame Discord clients send).
OPUS_SILENCE = b"\xf8\xff\xfe"
# Re-anchor a track to wall clock when RTP time and arrival time disagree by more than this
# (new SSRC after a reconnect, a client restart, or a client that pauses its RTP clock).
RTP_TOLERANCE_SECONDS = 2.0

TRANSCRIPTION_SAMPLE_RATE = 16000

LEGACY_OPUS_EXTENSION = ".skopus"
_LEGACY_PACKET_HEADER = struct.Struct("<HIH")  # sequence, RTP timestamp, payload size

_OGG_HEADER = struct.Struct("<4sBBqIIIB")
_OGG_CONTINUED, _OGG_BOS, _OGG_EOS = 0x01, 0x02, 0x04
_MAX_PAGE_SEGMENTS = 255


def _crc_table() -> list[int]:
    table = []
    for index in range(256):
        value = index << 24
        for _ in range(8):
            value = ((value << 1) ^ 0x04C11DB7) if value & 0x80000000 else value << 1
        table.append(value & 0xFFFFFFFF)
    return table


_CRC_TABLE = _crc_table()


def ogg_crc(data: bytes) -> int:
    """CRC-32 as Ogg defines it (polynomial 0x04C11DB7, no reflection, zero init)."""
    crc = 0
    table = _CRC_TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ byte) & 0xFF]
    return crc


def opus_packet_samples(packet: bytes) -> int:
    """Samples (at 48 kHz) in one Opus packet, from its TOC byte (RFC 6716 section 3.1); 0 if invalid."""
    if not packet:
        return 0
    toc = packet[0]
    config = toc >> 3
    if config < 12:  # SILK
        frame = (480, 960, 1920, 2880)[config % 4]
    elif config < 16:  # hybrid
        frame = (480, 960)[config % 2]
    else:  # CELT
        frame = (120, 240, 480, 960)[config % 4]
    code = toc & 0x03
    if code == 0:
        count = 1
    elif code in (1, 2):
        count = 2
    else:
        if len(packet) < 2:
            return 0
        count = packet[1] & 0x3F
    samples = frame * count
    return samples if 0 < samples <= 5760 else 0  # 120 ms is the most one packet may hold


class OggOpusWriter:
    """Writes Opus packets into an Ogg Opus stream (RFC 7845) on an open binary file.

    The stream opens with PRE_SKIP_SAMPLES of silence that decoders drop, so `samples_written`
    and decoded positions both count from the first real packet. Packets are grouped into pages;
    call `flush` to end the current page (the caller decides how often, which bounds what a
    crash can lose).
    """

    def __init__(self, handle: BinaryIO, serial: int, channels: int = OPUS_CHANNELS) -> None:
        self._handle = handle
        self._serial = serial & 0xFFFFFFFF
        self._sequence = 0
        self._granule = 0
        self._packets: list[bytes] = []
        self._segments = 0
        self._closed = False
        head = struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, PRE_SKIP_SAMPLES, OPUS_SAMPLE_RATE, 0, 0)
        vendor = b"ScrollKeeper"
        tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
        self._write_page([head], granule=0, flags=_OGG_BOS)
        self._write_page([tags], granule=0, flags=0)
        self.write_silence(PRE_SKIP_SAMPLES // FRAME_SAMPLES)

    @property
    def samples_written(self) -> int:
        """Samples after the pre-skip, i.e. the track position the next packet starts at."""
        return self._granule - PRE_SKIP_SAMPLES

    def write_packet(self, packet: bytes) -> None:
        samples = opus_packet_samples(packet)
        if samples == 0:
            # A packet the decoder can't size would break the timeline; keep time with silence.
            packet, samples = OPUS_SILENCE, FRAME_SAMPLES
        needed = len(packet) // 255 + 1
        if self._segments + needed > _MAX_PAGE_SEGMENTS:
            self.flush()
        self._packets.append(packet)
        self._segments += needed
        self._granule += samples

    def write_silence(self, frames: int) -> None:
        for _ in range(frames):
            self.write_packet(OPUS_SILENCE)

    def flush(self) -> None:
        """End the current page (if it has packets) and hand it to the file."""
        if self._packets:
            self._write_page(self._packets, granule=self._granule, flags=0)
            self._packets, self._segments = [], 0

    def close(self) -> None:
        if self._closed:
            return
        self._write_page(self._packets, granule=self._granule, flags=_OGG_EOS)
        self._packets, self._segments = [], 0
        self._closed = True

    def _write_page(self, packets: list[bytes], granule: int, flags: int) -> None:
        lacing = bytearray()
        for packet in packets:
            lacing.extend(b"\xff" * (len(packet) // 255))
            lacing.append(len(packet) % 255)
        body = b"".join(packets)
        header = _OGG_HEADER.pack(b"OggS", 0, flags, granule, self._serial, self._sequence, 0, len(lacing))
        page = bytearray(header + bytes(lacing) + body)
        struct.pack_into("<I", page, 22, ogg_crc(bytes(page)))
        self._handle.write(page)
        self._sequence += 1


def ogg_duration_seconds(path: Path) -> float:
    """Duration of an Ogg Opus file written by OggOpusWriter, from the last page's granule."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 65536))
        tail = handle.read()
    index = tail.rfind(b"OggS")
    while index >= 0:
        if index + _OGG_HEADER.size <= len(tail):
            granule = _OGG_HEADER.unpack_from(tail, index)[3]
            if granule >= 0:
                return max(0, granule - PRE_SKIP_SAMPLES) / OPUS_SAMPLE_RATE
        index = tail.rfind(b"OggS", 0, index)
    return 0.0


class TrackTimeline:
    """Decides where each packet of one speaker goes on that speaker's track.

    Positions are in 48 kHz samples from the track's first packet. Within one RTP stream the
    timestamp is trusted (it keeps advancing across silence). The track re-anchors to arrival
    time when the SSRC changes or when RTP time drifts from arrival time by more than
    RTP_TOLERANCE_SECONDS, so a bad timestamp can never insert more silence than really passed.
    """

    def __init__(self, tolerance_seconds: float = RTP_TOLERANCE_SECONDS) -> None:
        self._tolerance = int(tolerance_seconds * OPUS_SAMPLE_RATE)
        self._start_clock: float | None = None
        self._ssrc: int | None = None
        self._anchor_ts = 0
        self._anchor_pos = 0

    def target(self, ssrc: int, rtp_timestamp: int, clock: float, written: int) -> int:
        """Track position (samples) where this packet belongs; `clock` is its arrival time in seconds."""
        if self._start_clock is None:
            self._start_clock, self._ssrc = clock, ssrc
            self._anchor_ts, self._anchor_pos = rtp_timestamp, 0
            return 0
        wall_pos = int(round((clock - self._start_clock) * OPUS_SAMPLE_RATE))
        position = self._anchor_pos + _rtp_delta(rtp_timestamp, self._anchor_ts)
        if ssrc != self._ssrc or abs(position - wall_pos) > self._tolerance:
            self._ssrc = ssrc
            self._anchor_ts = rtp_timestamp
            self._anchor_pos = max(wall_pos, written)
            position = self._anchor_pos
        return position


def _rtp_delta(timestamp: int, anchor: int) -> int:
    """Signed difference of two 32-bit RTP timestamps, allowing for wraparound."""
    delta = (timestamp - anchor) % (1 << 32)
    return delta - (1 << 32) if delta >= (1 << 31) else delta


class SpeakerTrack:
    """One speaker's Ogg Opus file plus its timeline. Not thread-safe: one writer thread owns it."""

    def __init__(self, handle: BinaryIO, serial: int) -> None:
        self.handle = handle
        self.writer = OggOpusWriter(handle, serial)
        self.timeline = TrackTimeline()
        self.packets = 0

    def add(self, ssrc: int, rtp_timestamp: int, payload: bytes, clock: float) -> None:
        written = self.writer.samples_written
        gap = self.timeline.target(ssrc, rtp_timestamp, clock, written) - written
        if gap >= FRAME_SAMPLES:
            self.writer.write_silence(gap // FRAME_SAMPLES)
        self.writer.write_packet(payload)
        self.packets += 1

    def flush(self) -> None:
        self.writer.flush()
        self.handle.flush()

    def close(self) -> None:
        self.writer.close()
        self.handle.flush()
        self.handle.close()


# --- Legacy `.skopus` clips (before per-speaker tracks) ------------------------------------


@dataclass(slots=True)
class LegacyPacket:
    sequence: int
    timestamp: int
    payload: bytes


def read_legacy_packets(path: Path) -> list[LegacyPacket]:
    """Read a `.skopus` clip: repeated (sequence u16, RTP timestamp u32, size u16, Opus payload)."""
    packets: list[LegacyPacket] = []
    with path.open("rb") as handle:
        while header := handle.read(_LEGACY_PACKET_HEADER.size):
            if len(header) != _LEGACY_PACKET_HEADER.size:
                raise RuntimeError(f"Corrupted Opus packet header in {path}")
            sequence, timestamp, size = _LEGACY_PACKET_HEADER.unpack(header)
            payload = handle.read(size)
            if len(payload) != size:
                raise RuntimeError(f"Corrupted Opus packet payload in {path}")
            packets.append(LegacyPacket(sequence, timestamp, payload))
    return packets


def convert_legacy_clips(clips: Iterable[tuple[datetime, Path]], out_path: Path, serial: int) -> datetime | None:
    """Join one speaker's legacy clips into one Ogg Opus track, each clip at its recorded start time.

    Returns the track's start time (the first readable clip's), or None if no clip was readable.
    Unreadable clips are skipped with a warning.
    """
    readable: list[tuple[datetime, list[LegacyPacket]]] = []
    for started_at, path in sorted(clips, key=lambda clip: clip[0]):
        try:
            packets = read_legacy_packets(path)
        except (OSError, RuntimeError) as exc:
            log.warning("Skipping unreadable legacy clip %s: %s", path, exc)
            continue
        if packets:
            readable.append((started_at, packets))
    if not readable:
        return None
    track_start = readable[0][0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    track = SpeakerTrack(out_path.open("wb"), serial)
    try:
        for clip_index, (started_at, packets) in enumerate(readable):
            clip_offset = (started_at - track_start).total_seconds()
            first_ts = packets[0].timestamp
            for packet in packets:
                clock = clip_offset + _rtp_delta(packet.timestamp, first_ts) / OPUS_SAMPLE_RATE
                # Each clip gets its own pseudo-SSRC so the track re-anchors to the clip's start time.
                track.add(clip_index, packet.timestamp, packet.payload, clock)
    finally:
        track.close()
    return track_start


# --- Decoding for speech-to-text -----------------------------------------------------------


def decode_for_transcription(source: Path, destination: Path) -> None:
    """Decode a track to 16 kHz mono FLAC with ffmpeg (soxr resampling, stereo averaged to mono).

    FLAC keeps the upload small (long silent stretches compress to almost nothing) and is lossless.
    """
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
        "-i", str(source),
        "-af", "aresample=resampler=soxr",
        "-ac", "1", "-ar", str(TRANSCRIPTION_SAMPLE_RATE),
        "-c:a", "flac",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {source.name}: {result.stderr.strip()[-500:]}")
