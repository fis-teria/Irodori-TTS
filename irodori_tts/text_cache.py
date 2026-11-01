"""Bounded, process-local storage of detached frozen-backbone features."""
from collections import OrderedDict

import torch

TextFeatureKey = tuple[tuple[int, ...], tuple[bool, ...]]


def make_text_feature_keys(input_ids: torch.Tensor, mask: torch.Tensor) -> tuple[TextFeatureKey, ...]:
    """Build keys on CPU, before H2D; ignore only masked right padding."""
    if input_ids.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("Text feature keys must be built from CPU tensors before transfer.")
    if input_ids.ndim != 2 or input_ids.shape != mask.shape:
        raise ValueError("Text IDs and mask must have the same [batch, tokens] shape.")
    result = []
    for ids, valid in zip(input_ids.tolist(), mask.bool().tolist()):
        length = max((i + 1 for i, flag in enumerate(valid) if flag), default=1)
        result.append((tuple(ids[:length]), tuple(valid[:length])))
    return tuple(result)


class FrozenTextFeatureCache:
    """LRU of device tensors; deliberately not a Module/buffer/checkpoint field."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self.max_bytes = int(max_bytes)
        self._values = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = self.bypasses = 0

    def clear(self):
        self._values.clear()
        self.bytes = 0

    def lookup(self, namespace, keys):
        entries = []
        for key in keys:
            full = (namespace, key)
            value = self._values.get(full)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
                self._values.move_to_end(full)
            entries.append(value)
        return entries

    def put(self, namespace, key, value):
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            return
        full = (namespace, key)
        old = self._values.pop(full, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        while self.bytes + size > self.max_bytes:
            _, removed = self._values.popitem(last=False)
            self.bytes -= removed.numel() * removed.element_size()
            self.evictions += 1
        self._values[full] = value
        self.bytes += size

    def stats(self):
        return dict(entries=len(self._values), payload_bytes=self.bytes,
                    max_bytes=self.max_bytes, hits=self.hits, misses=self.misses,
                    evictions=self.evictions, bypasses=self.bypasses)
