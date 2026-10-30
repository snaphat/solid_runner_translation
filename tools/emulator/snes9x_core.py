"""
This module runs the Snes9x libretro core without a display window and exposes controller input, frames, memory, and save states.

The wrapper retains native callbacks and ROM buffers, advances controller input,
copies video frames, reads memory snapshots and serializes compatible save states.
Memory reads return independent bytes; poke performs bounded native writes.
GameSession provides the higher-level lifecycle and live memory views used by
gameplay verification. Direct users must unload the ROM and deinitialize the core.

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

    The decoder indexes the table by each 16-bit pixel value, converting RGB565 or
    0RGB1555 channels to packed eight-bit RGB bytes.

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
    Low-level, headless Snes9x libretro instance with retained ctypes callbacks.

    Retains ctypes callbacks and ROM storage for the core lifetime. Callers
    must unload the game and deinitialize the core, or use GameSession to own
    that lifecycle.

    Attributes:
        lib (ctypes.CDLL): Loaded libretro core; direct calls require the
            registered callback lifetimes.
        held (set[int]): Button IDs reported by the input callback on each
            emulated frame.
        fmt (int): Current libretro pixel format: 0 is 0RGB1555, 1 XRGB8888,
            and 2 RGB565. Each captured frame stores its own format value.
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
            RuntimeError: The core rejects the ROM; it is deinitialized before
                raising.
            OSError: The ROM cannot be read or the native core library cannot be loaded.
        """
        self._rom = Path(rom).read_bytes()
        self.lib = C.CDLL(os.fspath(core))
        self.fmt = 0  # 0=0RGB1555, 1=XRGB8888, 2=RGB565
        self.frame: tuple[bytes, int, int, int, int] | None = None
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

        L.retro_get_memory_data.restype = C.c_void_p
        L.retro_get_memory_size.restype = C.c_size_t
        L.retro_serialize_size.restype = C.c_size_t

    def reload_rom(self, image: bytes) -> None:
        """
        Replace the loaded cartridge while keeping its native buffer alive.

        Validate the input and allocate its backing buffer before unloading the
        current cartridge. Preparation failures leave the current cartridge and
        retained buffers unchanged. Once unloading starts, existing memory views
        become invalid and must be reacquired by the caller.

        Args:
            image (bytes): Complete replacement ROM contents retained for the
                cartridge lifetime.

        Raises:
            TypeError: image is not bytes; the current cartridge remains loaded.
            MemoryError: Replacement-buffer allocation fails before unloading.
            RuntimeError: The core rejects the replacement image; the previous
                cartridge has already been unloaded.
        """
        if not isinstance(cast(object, image), bytes):
            raise TypeError("Replacement ROM image must be bytes")

        # Prepare everything that can fail before releasing the current cartridge.
        # Its existing ROM buffers must stay alive until native unloading finishes.
        buffer = C.create_string_buffer(image, len(image))
        info = GameInfo(b"scratch.sfc", C.cast(buffer, C.c_void_p), len(image), None)
        self.lib.retro_unload_game()

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
            AssertionError: A supported environment request has no payload pointer.
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
        Copy a native video frame into retained Python storage.

        Args:
            data (int | None): Frame pixel pointer; null means a duplicate
                frame and leaves the stored frame unchanged.
            w (int): Frame width in pixels.
            h (int): Frame height in scanlines.
            pitch (int): Native byte stride between scanlines; h * pitch bytes are
                copied.
        """
        if data:  # keep the raw frame; convert only when a screenshot is taken
            self.frame = (C.string_at(data, h * pitch), w, h, pitch, self.fmt)

    def read_rgb_rows(self) -> list[bytes]:
        """
        Return the last captured frame as packed 8-bit RGB scanlines.

        Returns:
            list[bytes]: Independent packed RGB rows for the most recent video frame,
                excluding native scanline padding.

        Raises:
            AssertionError: No emulator frame has been captured yet.
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

        Raises:
            ValueError: The frame count is not a nonnegative integer.
        """
        validate_frame_count(n)
        for _ in range(n):
            self.lib.retro_run()

    def press(self, btn: str, frames: int = 4, after: int = 10) -> None:
        """
        Hold named buttons, then release all input for the requested delay.

        This replaces the current held set rather than adding to or restoring it.
        The hold phase clears input in a finally block even if emulation fails; the
        release delay runs only after a successful hold phase.

        Args:
            btn (str): Button name or + separated button names from BTN.
            frames (int, optional): Emulated frames to hold the button.
            after (int, optional): Released-input frames after the press.

        Raises:
            ValueError: A hold or release duration is not a nonnegative integer.
            KeyError: A + separated button name is not a key in BTN.
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
            AssertionError: No video frame has been captured.
            OSError: The PNG destination cannot be opened or written.
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
            ValueError: The region is unavailable or the requested span is outside
                it, or a payload element is outside the byte range.
            TypeError: data is not bytes or an iterable of integer byte values.
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
            RuntimeError: The core has no serialization support or fails to
                serialize its current state.
        """
        # Serialization size belongs to the loaded core and ROM. Allocate exactly
        # that size, and return an owned byte copy rather than a temporary native view.
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
        frame. Advance emulation after restoring when a fresh screenshot is needed.

        Args:
            raw (bytes): Opaque serialized state produced by this core and a
                compatible ROM.

        Raises:
            RuntimeError: The core rejects the serialized state bytes.
        """
        buf = C.create_string_buffer(raw, len(raw))
        if not self.lib.retro_unserialize(buf, len(raw)):
            raise RuntimeError("Emulator could not load state")
