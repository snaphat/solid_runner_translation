"""
This module rejects unsupported buffer types before binary readers or writers use them.

Readers can allow bytes and bytearray; writers can require a mutable bytearray.
The caller chooses the accepted classes, and a mismatch raises TypeError.
"""


def require_byte_buffer(data: object, accepted: tuple[type[bytes] | type[bytearray], ...]) -> None:
    """
    Reject values outside a reader or writer's supported byte-buffer types.

    Args:
        data (object): Resource, tile or destination buffer supplied by the caller.
        accepted (tuple[type[bytes] | type[bytearray], ...]): Allowed buffer classes;
            bytes is immutable and bytearray supports in-place writes.

    Raises:
        TypeError: data is not an instance of an accepted buffer class.
    """
    if not isinstance(data, accepted):
        raise TypeError("Expected " + " or ".join(kind.__name__ for kind in accepted) + " data")
