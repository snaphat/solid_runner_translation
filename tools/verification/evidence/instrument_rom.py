"""
This module creates an instrumented ROM copy that records text commands, terminators, and glyph draws in additional cartridge SRAM.

The builder reads the supplied ROM and writes an instrumented scratch copy.
The hooks execute the instructions they replace as well as recording trace
data; this does not by itself establish that instrumentation preserves every
aspect of game behavior.

All trace data lives in added cartridge SRAM: the scratch copy declares 128 KiB
instead of the game's 8 KiB ($70:0000-$70:1FFF). This keeps trace writes out of
game-owned memory. Scenes use much of WRAM, including $7E8000-$7EAFFF, for buffers;
menus also reuse $7E1F00-$7E1F7F. Offsets in save RAM:

  TRACE $FF00 ($71:7F00), counters:
    +04-06  text pointer when the event interpreter starts a text command ($81:C960)
    +08-0A  event-script position of that command;  +0B  the interpreter's X (actor slot)
    +0C     count of text commands started
    +20-21  bytes used in the glyph log (reset by each text command unless accumulating)
    +30     count of terminators read ($84:A953);  +32-34  pointer at that terminator
    +70-71  bytes used in the terminator ring
  RING $10000 ($72:0000), 16 bytes per terminator read: terminator (3), reader start $18B2 (3),
    column, line, $18B0 flags, $179C screen mode, $18D0 speaker side, glyph-log bytes used (2),
    script position (3)
  GLYPHS $18000 ($73:0000), 12 bytes per glyph: font offset (2), VWF x before (2), cell column,
    line, text bank, VWF x after (2), cell column after, line after
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import ctypes
from pathlib import Path

from core.rom import assemble_rom as assemble


class GlyphRecord(TypedDict):
    """
    One instrumented glyph draw, with native cursor positions before and after it.

    Attributes:
        font_offset (int): Byte offset of the glyph within its font table.
        before (int): VWF horizontal position in pixels before drawing.
        col (int): Native character-column state before drawing.
        row (int): Native text-row state before drawing.
        bank (int): Source text bus bank; 7E/7F identifies a dynamic insertion.
        after (int): VWF horizontal position in pixels after drawing.
        after_col (int): Native character-column state after drawing.
        after_row (int): Native text-row state after drawing.
    """

    font_offset: int
    before: int
    col: int
    row: int
    bank: int
    after: int
    after_col: int
    after_row: int


class TerminatorRecord(TypedDict):
    """
    Native reader state captured when a text terminator is consumed.

    Attributes:
        term (int): Terminator's 24-bit SNES bus address.
        start (int): Text reader's 24-bit starting bus address.
        column (int): Native character-column state at the terminator.
        row (int): Native text-row state at the terminator.
        flags (int): Text-engine flags from WRAM 18B0.
        ui_mode (int): Native screen mode from WRAM 179C.
        speaker_side (int): Dialogue portrait side from WRAM 18D0.
        draw_end (int): Exclusive byte offset in the instrumented glyph log.
        script (int): Event interpreter's 24-bit script bus address.
    """

    term: int
    start: int
    column: int
    row: int
    flags: int
    ui_mode: int
    speaker_side: int
    draw_end: int
    script: int


# Byte offsets in the expanded scratch SRAM; bus addresses and record layouts above.
TRACE, RING, GLYPHS = 0xFF00, 0x10000, 0x18000

# Buffer capacities in bytes: 1,536 terminator records and 1,024 glyph records.
RING_SIZE, LOG_SIZE = 0x6000, 0x3000

# The scratch hooks preserve native registers and resume replaced instructions.
# scan_log records text starts; end_log appends 16-byte terminator records;
# draw_log appends 12-byte glyph records around the actual renderer call.
# Both buffers stop accepting records when full rather than overwriting unread history.
ASM = """lorom
; Hooks run in the native CPU state at each patched instruction. Direct-page
; references ($32-$34, $50-$52, $3B-$3C) use the game's current D register.
; Absolute WRAM references use data bank $80, whose low addresses mirror WRAM.
; Trace stores use explicit long addresses in the expanded cartridge SRAM.
; Logging saves P, full A/X/Y and DB before changing widths or bank, then restores
; them before executing the replaced instruction. D is neither changed nor saved.
org $00FFD8
    db $07                      ; 128 KB SRAM in the scratch copy (the game uses the first 8 KB)
; Text-command entry: replace the four-byte JSL with a long-call wrapper.
org $81C960
    JSL scan_log
; Terminator reader: replace LDA $18B0 / BIT #$40 (five bytes) with JSL / NOP.
org $84A953
    JSL end_log
    NOP
; Glyph dispatch: replace a bank-local JSR; omitted when glyph logging is off.
org $84A9CD
    JSR draw_log
org $84F000
; scan_log: capture the text pointer and event-script position before the native
; text-command handler runs. X identifies the actor; Y is the script-relative
; cursor and $32-$34 is its script base. Only the low byte of X is recorded.
; Returns with RTL after invoking the original JSL $80:D938 with restored state.
scan_log:
    PHP
    REP #$30
    PHA
    PHX
    PHY
    PHB
    ; Use byte-sized copies for pointers and counters; select mirrored WRAM.
    SEP #$30
    LDA #$80
    PHA
    PLB
    ; Reader start pointer ($18B2-$18B4) -> TRACE +04..+06.
    LDA $18B2
    STA $717F04
    LDA $18B3
    STA $717F05
    LDA $18B4
    STA $717F06
    ; Script base plus cursor, minus one, identifies the text-command opcode.
    ; The addition carries into the high offset byte; the bank is copied as-is.
    TYA
    CLC
    ADC $32
    STA $717F08
    LDA $33
    ADC #$00
    STA $717F09
    LDA $34
    STA $717F0A
    REP #$20
    LDA $717F08
    DEC A
    STA $717F08
    SEP #$20
    ; Record the actor slot and increment the byte counter (wraps at 256).
    TXA
    STA $717F0B
    LDA $717F0C
    INC A
    STA $717F0C
    ; The optional reset fragment clears only the glyph byte count, not its data.
{reset}    PLB
    REP #$30
    PLY
    PLX
    PLA
    PLP
    JSL $80D938                 ; replaced instruction
    RTL
; end_log: capture the terminator cursor and append one 16-byte reader record.
; Y is the reader-relative cursor; $50-$52 is its text base. The intercepted
; LDA/BIT runs after state restoration, so its A and flag effects reach the caller.
; This hook expects the native eight-bit accumulator at the LDA/BIT site.
end_log:
    PHP
    REP #$30
    PHA
    PHX
    PHY
    PHB
    ; Use byte-sized copies for pointers and counters; select mirrored WRAM.
    SEP #$30
    LDA #$80
    PHA
    PLB
    ; Text base plus cursor -> TRACE +32..+34; preserve the source bank.
    TYA
    CLC
    ADC $50
    STA $717F32
    LDA $51
    ADC #$00
    STA $717F33
    LDA $52
    STA $717F34
    ; Count every terminator, including those observed after the ring is full.
    LDA $717F30
    INC A
    STA $717F30
    REP #$30
    ; X is the next byte offset; stop appending at the 24 KiB capacity.
    LDA $717F70
    TAX
    CPX #$6000
    BCC .ring_room
    JMP .ring_full
.ring_room:
    ; +00..+02 terminator pointer; +03..+05 reader-start pointer.
    SEP #$20
    LDA $717F32
    STA $720000,X
    LDA $717F33
    STA $720001,X
    LDA $717F34
    STA $720002,X
    LDA $18B2
    STA $720003,X
    LDA $18B3
    STA $720004,X
    LDA $18B4
    STA $720005,X
    ; +06..+0A column, row, engine flags, screen mode and speaker side.
    LDA $18BA
    STA $720006,X
    LDA $18BB
    STA $720007,X
    LDA $18B0
    STA $720008,X
    LDA $179C
    STA $720009,X
    LDA $18D0
    STA $72000A,X
    ; +0B..+0C glyph-log end; +0D..+0F most recent text-command position.
    LDA $717F20
    STA $72000B,X
    LDA $717F21
    STA $72000C,X
    LDA $717F08
    STA $72000D,X
    LDA $717F09
    STA $72000E,X
    LDA $717F0A
    STA $72000F,X
    REP #$20
    TXA
    CLC
    ; Publish the completed record by advancing the ring byte count last.
    ADC #$0010
    STA $717F70
.ring_full:
    PLB
    REP #$30
    PLY
    PLX
    PLA
    PLP
    LDA $18B0                   ; replaced instructions
    BIT #$40
    RTL
; draw_log: wrap the current bank-$84 glyph routine with before/after capture.
; The native glyph path uses 16-bit index registers; log offsets can reach $3000.
; $3B-$3C holds the font byte offset, $18F0-$18F1 the VWF pixel cursor,
; $18BA/$18BB the cell column/row, and $52 the source text bank.
; Restore entry state before JSR, then preserve the renderer's resulting state
; around the after-capture. RTS returns with the renderer's A/X/Y/DB/P effects.
draw_log:
    PHP
    REP #$30
    PHA
    PHX
    PHY
    PHB
    SEP #$20
    LDA #$80
    PHA
    PLB
    REP #$20
    ; Use the unpublished log offset for both halves of this glyph record.
    LDA $717F20
    TAX
    CPX #$3000
    BCS .skip_before
    SEP #$20
    ; +00..+06: font offset, VWF cursor, cell column/row and source bank.
    LDA $3B
    STA $730000,X
    LDA $3C
    STA $730001,X
    LDA $18F0
    STA $730002,X
    LDA $18F1
    STA $730003,X
    LDA $18BA
    STA $730004,X
    LDA $18BB
    STA $730005,X
    LDA $52
    STA $730006,X
; Even with a full log, restore state and execute the actual glyph draw.
.skip_before:
    PLB
    REP #$30
    PLY
    PLX
    PLA
    PLP
    JSR ${draw:04X}                 ; the build's glyph routine (replaced instruction)
    PHP
    REP #$30
    PHA
    PHX
    PHY
    PHB
    SEP #$20
    LDA #$80
    PHA
    PLB
    REP #$20
    ; Use the unpublished log offset for both halves of this glyph record.
    LDA $717F20
    TAX
    CPX #$3000
    BCS .skip_after
    SEP #$20
    ; +07..+0A: resulting VWF cursor and cell column/row; +0B is padding.
    LDA $18F0
    STA $730007,X
    LDA $18F1
    STA $730008,X
    LDA $18BA
    STA $730009,X
    LDA $18BB
    STA $73000A,X
    REP #$20
    TXA
    CLC
    ; Commit only after both halves are written, advancing by twelve bytes.
    ADC #$000C
    STA $717F20
.skip_after:
    PLB
    REP #$30
    PLY
    PLX
    PLA
    PLP
    RTS
"""

# Assembly fragment omitted in accumulating mode so glyph history survives text starts.
RESET = """    LDA #$00
    STA $717F20
    STA $717F21
"""

# Unheadered ROM offsets -> required original bytes at text-start and terminator hooks.
SITES = {0x00C960: bytes.fromhex("2238D980"), 0x022953: bytes.fromhex("ADB0188940")}
DRAW = 0x0229CD  # File offset of the English VWF glyph-draw hook.
FREE = (0x027000, 0x027400)  # Half-open ROM file range: $84:F000-$84:F3FF.


def build_instrumented_rom(
    release: bytes | bytearray, glyph_log: bool = True, accumulate: bool = False
) -> bytes:
    """
    Return the instrumented ROM bytes for a built release.

    With glyph_log disabled, installs only text-start and page-end hooks for
    the Japanese reader. Accumulation preserves menu glyphs drawn before a
    dialogue page restarts its trace.

    Args:
        release (bytes | bytearray): Uninstrumented release or original ROM; never modified
            in place.
        glyph_log (bool, optional): Enable glyph-read instrumentation used for
            English pixel checks.
        accumulate (bool, optional): Retain glyph-read history across text
            draws for interface-screen checks.

    Returns:
        bytes: Instrumented copy with trace routines and additional cartridge
            SRAM.

    Raises:
        ValueError: The image does not contain the complete reserved hook range,
            hook instructions/free space do not match expectations, or
            instrumentation assembly fails.
        OSError: Scratch-file creation, access, or cleanup fails, or Asar cannot start.
    """
    # Slicing a short image would silently omit missing bytes from the free-space
    # check. Require the entire reserved range before inspecting any hook sites.
    if len(release) < FREE[1]:
        raise ValueError("ROM does not contain the complete instrumentation hook range")

    # Refuse hook sites whose instructions changed, rather than overwriting
    # unknown code with instrumentation built for a different reader.
    for at, expected in SITES.items():
        if release[at : at + len(expected)] != expected:
            raise ValueError(f"Hook site {at:06X} no longer holds the original instruction")

    if glyph_log and release[DRAW] != 0x20:
        raise ValueError("Glyph draw site no longer holds a JSR")

    if any(b != 0xFF for b in release[FREE[0] : FREE[1]]):
        raise ValueError("$84:F000 is no longer free")

    asm = ASM.replace("{reset}", "" if accumulate else RESET)

    # Only the English renderer provides the VWF draw hook. Japanese scans
    # retain text-start and terminator hooks without that glyph hook.
    if glyph_log:
        asm = asm.replace("${draw:04X}", f"${release[DRAW + 1] | release[DRAW + 2] << 8:04X}")
    else:
        asm = asm.replace("org $84A9CD\n    JSR draw_log\n", "").replace("${draw:04X}", "$8000")

    # Assemble instrumentation into a disposable image; only the returned
    # bytes are used by verification sessions.
    with assemble.create_scratch_directory("solid-runner-instrument-") as folder:
        source = Path(folder) / "instrument.asm"
        image = Path(folder) / "instrumented.sfc"
        source.write_text(asm, encoding="utf-8")
        image.write_bytes(release)
        result = assemble.run_asar(source, image, "--fix-checksum=on")
        if result.returncode:
            raise ValueError(f"Asar failed on the instrumentation:\n{result.stdout}{result.stderr}")

        return image.read_bytes()


def read_glyph_log(
    s: bytes | ctypes.Array[ctypes.c_ubyte], start: int = 0, end: int | None = None
) -> list[GlyphRecord]:
    """
    Decode glyph-draw records from the instrumented ROM's scratch SRAM.

    The recorded font offsets and before/after cursor positions allow pixel checks
    to reconstruct expected text independently of the game's renderer. start and
    end select byte offsets within the glyph-log buffer, not record numbers.

    Args:
        s (bytes | ctypes.Array[ctypes.c_ubyte]): Cartridge SRAM containing the
            instrumentation trace and log.
        start (int, optional): First byte offset within the glyph log to
            decode; it must be a nonnegative integer multiple of twelve.
            Booleans are rejected.
        end (int | None, optional): Exclusive glyph-log byte offset; None
            uses the recorded log end, capped at LOG_SIZE. Explicit values must be
            integers (not booleans) between start and LOG_SIZE. An incomplete final
            record is ignored, including when end is not a multiple of twelve.

    Returns:
        list[GlyphRecord]: Glyph-draw records in observation order; no terminator
            records are included.

    Raises:
        ValueError: Bounds are not integers (excluding booleans), are outside
            the buffer, or start is not record-aligned; or the SRAM snapshot
            lacks the counters or records.
    """
    # Check types before arithmetic so invalid bounds have the same documented
    # failure as out-of-range bounds, and booleans cannot act as offsets.
    if (
        type(start) is not int
        or (end is not None and type(end) is not int)
        or start < 0
        or start > LOG_SIZE
        or start % 12
        or (end is not None and not start <= end <= LOG_SIZE)
    ):
        raise ValueError("Glyph-log bounds must select complete records within the buffer")
    if len(s) < TRACE + 0x22:
        raise ValueError("SRAM does not contain the glyph-log counters")
    used = (
        min(s[TRACE + 0x20] | s[TRACE + 0x21] << 8, LOG_SIZE) if end is None else min(end, LOG_SIZE)
    )
    out: list[GlyphRecord] = []
    if GLYPHS + used - used % 12 > len(s):
        raise ValueError("SRAM does not contain the requested glyph-log records")

    # Glyph records are twelve bytes. Ignore an incomplete trailing record
    # when a capture observes the trace between writes.
    for i in range(GLYPHS + start, GLYPHS + used - used % 12, 12):
        out.append(
            {
                "font_offset": s[i] | s[i + 1] << 8,
                "before": s[i + 2] | s[i + 3] << 8,
                "col": s[i + 4],
                "row": s[i + 5],
                "bank": s[i + 6],
                "after": s[i + 7] | s[i + 8] << 8,
                "after_col": s[i + 9],
                "after_row": s[i + 10],
            }
        )

    return out


def read_terminator_log(s: bytes | ctypes.Array[ctypes.c_ubyte]) -> list[TerminatorRecord]:
    """
    Decode the native text terminators recorded since the trace was cleared.

    Records retain text pointers, cursor state and script positions, allowing menu
    captures to be matched to their owning translated entries in draw order.

    Args:
        s (bytes | ctypes.Array[ctypes.c_ubyte]): Cartridge SRAM containing the
            instrumentation trace and log.

    Returns:
        list[TerminatorRecord]: Terminator-read records in observation order.
            The recorded byte count is capped at RING_SIZE and incomplete final
            sixteen-byte records are ignored.

    Raises:
        ValueError: The SRAM snapshot lacks the recorded terminator data.
        IndexError: The SRAM snapshot is too short to read the ring counters.
    """
    used = min(s[TRACE + 0x70] | s[TRACE + 0x71] << 8, RING_SIZE)
    if RING + used - used % 16 > len(s):
        raise ValueError("SRAM does not contain the recorded terminator log")
    out: list[TerminatorRecord] = []

    # Terminator records are sixteen bytes; their draw_end field delimits
    # the glyphs accumulated when that particular terminator was read.
    for i in range(RING, RING + used - used % 16, 16):
        b = bytes(s[i : i + 16])
        out.append(
            {
                "term": int.from_bytes(b[0:3], "little"),
                "start": int.from_bytes(b[3:6], "little"),
                "column": b[6],
                "row": b[7],
                "flags": b[8],
                "ui_mode": b[9],
                "speaker_side": b[10],
                "draw_end": int.from_bytes(b[11:13], "little"),
                "script": int.from_bytes(b[13:16], "little"),
            }
        )

    return out
