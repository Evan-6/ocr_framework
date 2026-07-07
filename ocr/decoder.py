"""CTC greedy decoding — works on torch tensors or numpy arrays."""
from __future__ import annotations

from .charset import Charset


def ctc_greedy_decode(logits, charset: Charset) -> list[str]:
    """logits: (B, T, C) torch.Tensor or np.ndarray -> list of B strings."""
    ids = logits.argmax(-1)
    if hasattr(ids, "cpu"):
        ids = ids.cpu().numpy()
    return [charset.decode_ids(row) for row in ids]
