"""
This module defines where the game's two font bitmap tables reside in an unheadered ROM.

Text measurement and font installation use these file offsets to locate glyph
records. Keeping the addresses here lets either operation locate the tables
without importing load_font_artwork, which reads the editable artwork on import.
Both tables store each glyph as eleven little-endian 16-bit rows (22 bytes).
This module defines addresses only; it does not read files or modify the ROM.
"""

# Native fixed-width font table, including the original Japanese glyphs.
MAIN_FONT = 0x170000
# Separate 256-glyph Latin table read by the variable-width text renderer.
VWF_FONT = 0x3F0000
