"""
This module runs the Snes9x libretro core without a display window and exposes controller input, frames, memory, and save states.

The wrapper retains native callbacks and ROM buffers, advances controller input, copies video
frames, reads memory snapshots and serializes compatible save states. Memory reads return
independent bytes; poke performs bounded native writes. Each captured frame also retains SRAM,
WRAM and VRAM snapshots from its video callback, so pixel checks do not pair an image with
memory read later in emulation. GameSession manages ROM loading, core shutdown and the live
memory views used by gameplay verification. Direct users must unload the ROM and deinitialize
the core.

Usage from Python:
    from emulator.snes9x_core import Emu
    e = Emu('build/Solid Runner (English).sfc')
    try:
        e.run(60)
        e.press('start', 5)
        e.save_screenshot('out.png')
        wram, vram = e.read_wram_bytes(), e.read_vram_bytes()
    finally:
        e.lib.retro_unload_game()
        e.lib.retro_deinit()
"""

from __future__ import annotations

import array
import ctypes as C
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, cast

from core.graphics import write_png as png
from emulator.validate_controller_input import BTN, validate_frame_count

# Default core DLL resolved at import time; the constructor can override this path.
CORE = os.environ.get(
    "SOLID_RUNNER_CORE",
    str(Path(__file__).resolve().parents[2] / "external/snes9x/snes9x_libretro.dll"),
)

# Pixel-format ID -> pixel-to-RGB conversion table. Callers must not mutate it.
TABLES: dict[int, list[bytes]] = {}


def _build_rgb_lookup(fmt: int) -> list[bytes]:
    """
    Build or reuse a native-pixel conversion table for screenshot RGB output.

    The decoder indexes the table by each 16-bit pixel value, converting RGB565 or 0RGB1555
    channels to packed eight-bit RGB bytes.

    Args:
        fmt (int): Libretro pixel format: 2 selects RGB565; other values select
            0RGB1555.

    Returns:
        list[bytes]: Cached 65,536-entry table mapping a native 16-bit pixel value
            to its three-byte RGB representation. Callers must not modify it.
    """
    if fmt not in TABLES:
        # Libretro format 2 is RGB565; the other supported 16-bit format is
        # 0RGB1555. Expand component bits to full 0-255 screenshot samples.
        if fmt == 2:
            TABLES[fmt] = [
                bytes(((p >> 11 & 31) * 255 // 31, (p >> 5 & 63) * 255 // 63, (p & 31) * 255 // 31))
                for p in range(65536)
            ]
        else:
            TABLES[fmt] = [
                bytes(((p >> 10 & 31) * 255 // 31, (p >> 5 & 31) * 255 // 31, (p & 31) * 255 // 31))
                for p in range(65536)
            ]

    return TABLES[fmt]


def read_snapshot_blocks(raw: bytes) -> dict[str, bytes]:
    """
    Read named blocks from the pinned Snes9x core's version-12 save-state format.

    Args:
        raw (bytes): Complete uncompressed state returned by Snes9x 1.63's
            retro_serialize API. Other formats and snapshot versions are rejected.

    Returns:
        dict[str, bytes]: Three-letter block tags mapped to independent byte payloads.

    Raises:
        TypeError: If raw is not bytes.
        ValueError: If the snapshot version is unsupported, a block header is malformed, a block
            is truncated, or a block tag occurs more than once.
    """
    if not isinstance(cast(object, raw), bytes):
        raise TypeError("Snes9x snapshot must be bytes")
    magic = b"#!s9xsnp:0012\n"
    if not raw.startswith(magic):
        raise ValueError("Expected a Snes9x version-12 snapshot")
    blocks: dict[str, bytes] = {}
    offset = len(magic)
    while offset < len(raw):
        header = raw[offset : offset + 11]
        if (
            len(header) != 11
            or header[3:4] != b":"
            or header[10:11] != b":"
            or not header[4:10].isdigit()
        ):
            raise ValueError("Invalid Snes9x snapshot block header")
        if not all(65 <= value <= 90 for value in header[:3]):
            raise ValueError("Invalid Snes9x snapshot block tag")
        tag = header[:3].decode("ascii")
        size = int(header[4:10])
        offset += 11
        if tag in blocks or offset + size > len(raw):
            raise ValueError("Duplicate or truncated Snes9x snapshot block")
        blocks[tag] = raw[offset : offset + size]
        offset += size
    return blocks


def decode_dma_configuration(raw: bytes) -> bytes:
    """
    Decode the pinned core's saved DMA channels into eight register-sized slots.

    Snes9x serializes DMA state separately from FillRAM. Its version-12 DMA block has eight
    nineteen-byte records; two-byte integers are big-endian. Each returned sixteen-byte slot
    supplies configuration registers $43n0-$43n7. The remaining eight bytes are zero and do not
    represent live transfer counters.

    Args:
        raw (bytes): Complete 152-byte DMA block from a version-12 core snapshot.

    Returns:
        bytes: Eight sixteen-byte slots in channel order, with register-order
            little-endian addresses and the original control-bit assignments.

    Raises:
        ValueError: If the DMA block does not contain eight 19-byte records or a record
            specifies a transfer mode outside 0 through 7.
    """
    if len(raw) != 152:
        raise ValueError("Expected eight nineteen-byte Snes9x DMA records")
    out = bytearray(128)
    for channel in range(8):
        record = raw[channel * 19 : (channel + 1) * 19]
        if record[5] > 7:
            raise ValueError("Invalid Snes9x DMA transfer mode")
        control = record[5]
        for index, bit in ((0, 7), (1, 6), (2, 5), (3, 3), (4, 4)):
            control |= bool(record[index]) << bit
        start = channel * 16
        out[start : start + 8] = bytes(
            (
                control,
                record[6],
                record[8],
                record[7],
                record[9],
                record[11],
                record[10],
                record[12],
            )
        )
    return bytes(out)


class GameInfo(C.Structure):
    """
    ctypes layout of retro_game_info.

    The backing path and content buffers must outlive their native use.

    Attributes:
        path (Optional[bytes]): C-compatible encoded content path passed to
            libretro.
        data (Optional[int]): Native pointer to retained ROM-content storage.
        size (int): Length of the ROM-content buffer in bytes.
        meta (Optional[bytes]): Optional C-compatible content metadata string.
    """

    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


# libretro callback ABI signatures. Emu retains each callback object while the
# core stores its function pointer, preventing garbage collection of native hooks.
ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIOB_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL_CB = C.CFUNCTYPE(None)
STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class Emu:
    """
    This class controls a Snes9x libretro instance without opening a display window.

    It keeps ctypes callbacks and loaded ROM storage alive while the core uses them. Direct
    callers must unload the game and deinitialize the core; GameSession performs that cleanup
    for verification sessions.

    Attributes:
        lib (ctypes.CDLL): Loaded libretro core; direct calls require the
            registered callback lifetimes.
        held (set[int]): Button IDs reported by the input callback on each
            emulated frame.
        fmt (int): Current libretro pixel format: 0 is 0RGB1555, 1 XRGB8888,
            and 2 RGB565. Each captured frame stores its own format value.
        frame_memory (dict[int, bytes]): Independent SRAM (0), WRAM (2) and VRAM (3)
            snapshots copied with the retained video frame, not live emulator memory.
        capture_registers (bool): Enable version-12 core snapshots during video
            callbacks to retain register memory and DMA configuration without emulated reads.
        frame_registers (bytes | None): Captured register memory with decoded DMA slots;
            None until a captured frame supplies register data.
        frame (tuple[bytes, int, int, int, int] | None): Retained raw pixel bytes,
            width, height, byte pitch, and format; None before the first video callback.
    """

    def __init__(self, rom: str | os.PathLike[str], core: str | os.PathLike[str] = CORE) -> None:
        """
        Load a ROM path through the core DLL.

        Args:
            rom (str | os.PathLike[str]): ROM file path loaded by libretro;
                the file must remain readable for loading.
            core (str | os.PathLike[str], optional): Snes9x libretro DLL
                path; defaults to the installed pinned core.

        Raises:
            RuntimeError: If the core rejects the ROM; it is deinitialized before raising.
            OSError: If the ROM cannot be read or the native core library cannot be loaded.
        """
        self._rom = Path(rom).read_bytes()
        self.lib = C.CDLL(os.fspath(core))
        self.fmt = 0  # 0=0RGB1555, 1=XRGB8888, 2=RGB565
        self.frame: tuple[bytes, int, int, int, int] | None = None
        self.frame_memory: dict[int, bytes] = {}
        self.capture_registers = False
        self.frame_registers: bytes | None = None
        self._frame_capture_error: str | None = None
        self.held: set[int] = set()
        self._sysdir = C.c_char_p(os.path.dirname(os.path.abspath(rom)).encode())

        # libretro retains raw function pointers; keep the ctypes callbacks alive
        # for the entire core lifetime rather than letting Python collect them.
        discard_sample: Callable[[int, int], None] = lambda _left, _right: None
        discard_batch: Callable[[int | None, int], int] = lambda _data, count: count
        self._cbs = [
            ENV_CB(self._handle_environment),
            VIDEO_CB(self._receive_video),
            AUDIO_CB(discard_sample),
            AUDIOB_CB(discard_batch),
            POLL_CB(lambda: None),
            STATE_CB(self._read_controller_state),
        ]
        L = self.lib
        L.retro_get_memory_data.restype = C.c_void_p
        L.retro_get_memory_size.restype = C.c_size_t
        L.retro_serialize_size.restype = C.c_size_t
        L.retro_set_environment(self._cbs[0])
        L.retro_set_video_refresh(self._cbs[1])
        L.retro_set_audio_sample(self._cbs[2])
        L.retro_set_audio_sample_batch(self._cbs[3])
        L.retro_set_input_poll(self._cbs[4])
        L.retro_set_input_state(self._cbs[5])
        L.retro_init()

        # A core may retain the content pointer after retro_load_game returns.
        self._buf = C.create_string_buffer(self._rom, len(self._rom))
        gi = GameInfo(
            os.path.abspath(rom).encode(), C.cast(self._buf, C.c_void_p), len(self._rom), None
        )
        if not L.retro_load_game(C.byref(gi)):
            L.retro_deinit()
            raise RuntimeError("Emulator could not load the ROM")

    def reload_rom(self, image: bytes) -> None:
        """
        Replace the loaded cartridge while keeping its native buffer alive.

        The input is validated and its backing buffer is allocated before unloading the current
        cartridge. Preparation failures leave the current cartridge and retained buffers
        unchanged. Once unloading starts, existing memory views become invalid and must be
        reacquired by the caller. The retained video frame and its memory snapshots are cleared
        when the old ROM is unloaded.

        Args:
            image (bytes): Complete replacement ROM contents retained for the cartridge
                lifetime.

        Raises:
            TypeError: If image is not bytes; the current cartridge remains loaded.
            MemoryError: If memory cannot be allocated for the replacement ROM buffer before the
                current ROM is unloaded.
            RuntimeError: If the core rejects the replacement image; the previous cartridge has
                already been unloaded.
        """
        if not isinstance(cast(object, image), bytes):
            raise TypeError("Replacement ROM image must be bytes")

        # Prepare everything that can fail before releasing the current cartridge.
        # Its existing ROM buffers must stay alive until native unloading finishes.
        buffer = C.create_string_buffer(image, len(image))
        info = GameInfo(b"scratch.sfc", C.cast(buffer, C.c_void_p), len(image), None)
        self.lib.retro_unload_game()
        self.frame = None
        self.frame_memory = {}
        self.frame_registers = None
        self._frame_capture_error = None

        # The core may retain the replacement pointer even after loading returns.
        self._rom, self._buf = image, buffer
        if not self.lib.retro_load_game(C.byref(info)):
            raise RuntimeError("The emulator refused the ROM image")

    def _handle_environment(self, cmd: int, data: int | None) -> bool:
        """
        Handle supported libretro environment requests.

        Args:
            cmd (int): Environment command ID; high flag bits are masked before
                dispatch.
            data (int | None): Command-specific native payload pointer;
                supported commands read or write through it.

        Returns:
            bool: True when the command is supported, otherwise False.

        Raises:
            AssertionError: If a supported environment request has no payload pointer.
        """
        cmd &= 0xFFFF

        if data is None and cmd in (3, 9, 10, 31):
            raise AssertionError("Environment request requires a payload")

        if cmd == 10:  # SET_PIXEL_FORMAT
            fmt = C.cast(data or 0, C.POINTER(C.c_int))[0]
            if fmt not in (0, 1, 2):
                return False
            self.fmt = fmt
            return True

        if cmd in (9, 31):  # system / save dir
            # c_char_p.value creates a temporary Python bytes object. Return the
            # existing buffer, which this Emu owns for the core's whole lifetime.
            C.cast(data or 0, C.POINTER(C.c_void_p))[0] = C.cast(self._sysdir, C.c_void_p).value
            return True

        if cmd == 3:  # GET_CAN_DUPE
            C.cast(data or 0, C.POINTER(C.c_bool))[0] = True
            return True

        return False

    def _receive_video(self, data: int | None, w: int, h: int, pitch: int) -> None:
        """
        Copy a video frame and its SRAM, WRAM, and VRAM snapshots into Python storage.

        Snapshot errors are caught because exceptions cannot cross the native callback.
        The frame remains stored, its memory snapshots are cleared, and later snapshot
        readers report the saved error.

        Args:
            data (int | None): Frame pixel pointer; null means a duplicate
                frame and leaves the stored frame and paired memory unchanged.
            w (int): Frame width in pixels.
            h (int): Frame height in scanlines.
            pitch (int): Native byte stride between scanlines; h * pitch bytes are
                copied.
        """
        if data:  # keep the raw frame; convert only when a screenshot is taken
            self.frame = (C.string_at(data, h * pitch), w, h, pitch, self.fmt)
            # Copy memory at the video callback: the CPU can change VRAM
            # and SRAM again before retro_run returns to the caller.
            self._frame_capture_error = None
            self.frame_registers = None
            try:
                if self.capture_registers:
                    # Serialization copies the core's register mirrors without
                    # executing any emulated CPU instructions or reading MMIO.
                    blocks = read_snapshot_blocks(self.save_state())
                    self.frame_memory = {
                        0: blocks["SRA"][: self.lib.retro_get_memory_size(0)],
                        2: blocks["RAM"],
                        3: blocks["VRA"],
                    }
                    registers = bytearray(blocks["FIL"])
                    if len(registers) != 0x8000:
                        raise ValueError("Incomplete Snes9x register snapshot")
                    registers[0x4300:0x4380] = decode_dma_configuration(blocks["DMA"])
                    self.frame_registers = bytes(registers)
                else:
                    self.frame_memory = {
                        region: self.read_memory_bytes(region) for region in (0, 2, 3)
                    }
            except Exception as error:
                # Exceptions cannot propagate through a ctypes callback. Revoke
                # the evidence and raise from the public snapshot readers instead.
                self.frame_memory = {}
                self._frame_capture_error = str(error)

    def read_rgb_rows(self) -> list[bytes]:
        """
        Return the last captured frame as packed 8-bit RGB scanlines.

        Returns:
            list[bytes]: Independent packed RGB rows for the most recent video frame,
                excluding native scanline padding.

        Raises:
            AssertionError: If no emulator frame has been captured yet.
        """
        if self.frame is None:
            raise AssertionError("No emulator frame has been captured")

        raw, w, h, pitch, fmt = self.frame
        rows: list[bytes] = []

        # XRGB8888 contains a padding byte per pixel; remove it and reverse the
        # in-memory BGR channel order when building RGB screenshot rows.
        if fmt == 1:  # XRGB8888, little-endian: B G R X in memory
            for y in range(h):
                line = raw[y * pitch : y * pitch + w * 4]
                out = bytearray(w * 3)
                out[0::3], out[1::3], out[2::3] = line[2::4], line[1::4], line[0::4]
                rows.append(bytes(out))

            return rows

        table = _build_rgb_lookup(fmt)
        for y in range(h):
            pixels = array.array("H", raw[y * pitch : y * pitch + w * 2])
            if sys.byteorder != "little":
                pixels.byteswap()

            rows.append(b"".join(map(table.__getitem__, pixels)))

        return rows

    def _read_controller_state(self, port: int, device: int, index: int, id_: int) -> int:
        """
        Report whether a libretro button is held on controller port zero.

        Args:
            port (int): Controller port index; ports other than zero return
                released.
            device (int): Requested device type; accepted but not used by this
                joypad callback.
            index (int): Device sub-index; accepted but not used.
            id_ (int): Libretro button ID tested against held.

        Returns:
            int: One for a held button on port zero, otherwise zero.
        """
        return 1 if port == 0 and id_ in self.held else 0

    def run(self, n: int = 1) -> None:
        """
        Advance n frames using the current held button-ID set.

        Args:
            n (int, optional): Emulated frames to advance.
                Defaults to 1.

        Raises:
            ValueError: If the frame count is not a nonnegative integer.
        """
        validate_frame_count(n)
        for _ in range(n):
            self.lib.retro_run()

    def press(self, btn: str, frames: int = 4, after: int = 10) -> None:
        """
        Hold named buttons, then release all input for the requested delay.

        This replaces the current held set rather than adding to or restoring it. The hold phase
        clears input in a finally block even if emulation fails; the release delay runs only
        after a successful hold phase.

        Args:
            btn (str): Button name or + separated button names from BTN.
            frames (int, optional): Emulated frames to hold the button.
                Defaults to 4.
            after (int, optional): Released-input frames after the press.
                Defaults to 10.

        Raises:
            ValueError: If a hold or release duration is not a nonnegative integer.
            KeyError: If a + separated button name is not a key in BTN.
        """
        validate_frame_count(frames)
        validate_frame_count(after)
        self.held = {BTN[b] for b in btn.split("+")}
        try:
            self.run(frames)
        finally:
            self.held = set()
        self.run(after)

    def save_screenshot(self, path: str | os.PathLike[str]) -> None:
        """
        Save the last frame as a PNG.

        Args:
            path (str | os.PathLike[str]): Destination PNG for the most
                recently captured frame. An existing file is overwritten; the parent
                directory must already exist.

        Raises:
            AssertionError: If no video frame has been captured.
            OSError: If the PNG destination cannot be opened or written.
        """
        png.write_png(path, self.read_rgb_rows(), 3)

    def read_memory_bytes(self, id_: int) -> bytes:
        """
        Copy one libretro memory region into an independent byte snapshot.

        Args:
            id_ (int): Libretro memory-region ID: 0 is SRAM, 2 is WRAM and 3 is
                VRAM.

        Returns:
            bytes: Independent copy of the selected memory region, or empty bytes
                when it has no native data pointer.
        """
        p = self.lib.retro_get_memory_data(id_)
        n = self.lib.retro_get_memory_size(id_)
        return C.string_at(p, n) if p else b""

    def read_frame_memory(self, id_: int) -> bytes:
        """
        Return memory copied at the same callback as the retained screenshot.

        These snapshots pair memory with the retained image; read_memory_bytes instead
        reads the core's current state, which may have advanced past the video callback.
        Save-state restoration retains the old screenshot and its snapshots until a new
        frame arrives. ROM reload clears both.

        Args:
            id_ (int): Captured libretro memory region: zero for SRAM, two for WRAM or three for VRAM.

        Returns:
            bytes: Independent, immutable memory snapshot paired with the current image.

        Raises:
            ValueError: If the region is not SRAM/WRAM/VRAM, or is not an integer.
            RuntimeError: If no video callback has captured the requested snapshot yet.
        """
        if type(id_) is not int or id_ not in (0, 2, 3):
            raise ValueError("Frame memory is available only for SRAM (0), WRAM (2) and VRAM (3)")
        if id_ not in self.frame_memory:
            raise RuntimeError(self._frame_capture_error or "No frame memory has been captured")
        return self.frame_memory[id_]

    def read_frame_registers(self) -> bytes:
        """
        Return frame register memory with decoded DMA-channel configuration.

        Register data is available only after a video callback with capture_registers
        enabled. It comes from the pinned core's serialization format; the returned
        bytes are register mirrors, not values obtained by reading write-only emulated
        PPU registers. The $4300-$437F range is synthesized from the snapshot's separate
        DMA block; FillRAM itself does not contain the DMA channel configuration.

        Returns:
            bytes: A 32 KiB FillRAM copy paired with the image, with $43n0-$43n7
                populated from saved DMA state and each channel's remaining bytes zero.

        Raises:
            RuntimeError: If the retained frame has no register data, or capturing it failed.
        """
        if self.frame_registers is None:
            raise RuntimeError(self._frame_capture_error or "No frame registers have been captured")
        return self.frame_registers

    def read_wram_bytes(self) -> bytes:
        """
        Return an independent bytes snapshot of work RAM (libretro region 2).

        Returns:
            bytes: Independent snapshot of libretro memory region 2. Later emulation
                does not change these bytes.
        """
        return self.read_memory_bytes(2)

    def poke(self, addr: object, data: bytes | Iterable[int], id_: int = 2) -> None:
        """
        Write bytes inside a memory region (default WRAM).

        Args:
            addr (object): Integer byte offset relative to the selected memory region;
                booleans and other types are rejected.
            data (bytes | Iterable[int]): Replacement bytes; the complete
                span must fit inside the native region.
            id_ (int, optional): Libretro memory-region ID; poke defaults to WRAM
                (2).

        Raises:
            ValueError: If the region is unavailable or the requested span is outside it, or a
                payload element is outside the byte range.
            TypeError: If data is not bytes or an iterable of integer byte values.
        """
        p = self.lib.retro_get_memory_data(id_)
        size = self.lib.retro_get_memory_size(id_)
        # bytes(integer) means a zero-filled buffer of that length. Require an
        # iterable so a scalar payload cannot silently erase native memory.
        payload = data if isinstance(data, bytes) else bytes(iter(data))

        # Validate the complete write against the native region before pointer
        # arithmetic; ctypes.memmove cannot enforce Python buffer bounds.
        if not p or type(addr) is not int or not 0 <= addr <= size or len(payload) > size - addr:
            raise ValueError(
                f"Write outside emulator memory region {id_}: offset {addr}, size {len(payload)}, capacity {size}"
            )

        if payload:
            C.memmove(p + addr, payload, len(payload))

    def read_vram_bytes(self) -> bytes:
        """
        Return an independent bytes snapshot of video RAM (libretro region 3).

        Returns:
            bytes: Independent snapshot of libretro memory region 3, suitable for
                tilemap inspection without retaining a native pointer.
        """
        return self.read_memory_bytes(3)

    def save_state(self) -> bytes:
        """
        Return a serialized native save state.

        Returns:
            bytes: Opaque serialized state for a compatible ROM and this core
                version.

        Raises:
            RuntimeError: If the core has no serialization support or fails to serialize its
                current state.
        """
        # Serialization size belongs to the loaded core and ROM. Allocate exactly
        # that size, then return independent bytes rather than a view of core memory.
        n = self.lib.retro_serialize_size()
        if not n:
            raise RuntimeError("Emulator does not provide save states")

        buf = C.create_string_buffer(n)
        if not self.lib.retro_serialize(buf, n):
            raise RuntimeError("Emulator could not save state")

        return buf.raw

    def load_state(self, raw: bytes) -> None:
        """
        Restore serialized state bytes in the currently loaded cartridge.

        The state does not include this wrapper's held-input set or retained video
        frame. The previous frame remains available until a later video callback
        supplies new pixel data.

        Args:
            raw (bytes): Opaque serialized state produced by this core and a
                compatible ROM.

        Raises:
            RuntimeError: If the core rejects the serialized state bytes.
        """
        buf = C.create_string_buffer(raw, len(raw))
        if not self.lib.retro_unserialize(buf, len(raw)):
            raise RuntimeError("Emulator could not load state")
