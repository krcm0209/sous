"""The array half of fork persistence: one safetensors file per fork,
written from a live cache and read back into a fresh one. Every mlx import
is function-local; the rules around a file (naming, budget, when to write,
what to delete) live in forkstore, which imports no mlx at all.

Two layer classes are supported, recognised by name so neither mlx-lm nor
mlx-vlm is imported here: KVCache (keys, values, offset — the attention
layers) and ArraysCache (a list of state arrays — the recurrent layers).
They are what make_prompt_cache builds for every model sous serves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sous.engine.forkstore import ForkFileError, Header


class ForkUnsupported(Exception):
    """A cache with a layer class the file format does not cover."""


def layer_kinds(cache: Sequence[Any]) -> list[str]:
    kinds: list[str] = []
    for c in cache:
        name = type(c).__name__
        if name == "KVCache":
            kinds.append("kv")
        elif name == "ArraysCache":
            kinds.append("arrays")
        else:
            raise ForkUnsupported(f"layer class {name} cannot be persisted")
    return kinds


def eval_cache(cache: Sequence[Any]) -> None:
    """Materialise every layer's arrays on the calling thread. A fork copy
    is lazy until something evaluates it, and mx.get_active_memory() — what
    the pressure valve reads — does not count a lazy array."""
    import mlx.core as mx

    arrays = [a for c in cache for a in _arrays_of(c) if a is not None]
    if arrays:
        mx.eval(*arrays)


def _arrays_of(c: Any) -> list[Any]:
    if type(c).__name__ == "KVCache":
        return [c.keys, c.values]
    return list(c.cache)


def persist_cache(
    cache: Sequence[Any], path: Path, ids: Sequence[int], metadata: Mapping[str, str]
) -> None:
    """Write `cache` at `path`. The FULL padded key/value buffers go in, with
    the offset in metadata: saving the offset slice would materialise every
    tensor at once (a transient the size of the whole fork), and restoring
    the padded buffer keeps the capacity a never-serialised cache has. The
    arrays are evaluated first — free when they already are — so the save
    itself allocates nothing."""
    import mlx.core as mx

    kinds = layer_kinds(cache)
    eval_cache(cache)
    arrays: dict[str, Any] = {"ids": mx.array(list(ids), dtype=mx.int32)}
    meta: dict[str, str] = {**metadata, "kinds": ",".join(kinds)}
    for i, (c, kind) in enumerate(zip(cache, kinds, strict=True)):
        if kind == "kv":
            if c.keys is None or c.values is None:
                raise ForkUnsupported(f"layer {i}: an empty KVCache cannot be persisted")
            meta[f"c{i}_kind"] = "kv"
            meta[f"c{i}_offset"] = str(int(c.offset))
            arrays[f"c{i}_k"] = c.keys
            arrays[f"c{i}_v"] = c.values
        else:
            meta[f"c{i}_kind"] = "arrays"
            meta[f"c{i}_size"] = str(len(c.cache))
            for j, state in enumerate(c.cache):
                if state is None:
                    meta[f"c{i}_s{j}_none"] = "1"
                else:
                    arrays[f"c{i}_s{j}"] = state
    mx.save_safetensors(str(path), arrays, metadata=meta)


def restore_cache(path: Path, header: Header, cache: Sequence[Any], ids: Sequence[int]) -> None:
    """Fill `cache` — fresh from new_cache() — from `path`, on the calling
    thread, and evaluate it: the arrays then belong to this thread and are
    visible to mx.get_active_memory() before the pressure valve next reads
    it. A KVCache gets keys, values and then offset from metadata — never
    the state setter, which derives the offset from the padded shape and
    would misposition every token appended afterwards. Raises ForkFileError
    on any mismatch: the caller deletes the file."""
    import mlx.core as mx

    meta = header.metadata
    kinds = layer_kinds(cache)
    if meta.get("kinds", "").split(",") != kinds:
        raise ForkFileError(f"{path.name}: layer layout {meta.get('kinds')!r} != {kinds}")
    n = len(ids)
    if meta.get("n_tokens") != str(n):
        raise ForkFileError(f"{path.name}: n_tokens {meta.get('n_tokens')!r} != {n}")
    loaded = mx.load(str(path))
    if "ids" not in loaded:
        raise ForkFileError(f"{path.name}: no ids tensor")
    if loaded["ids"].tolist() != list(ids):  # ty: ignore[invalid-argument-type]
        raise ForkFileError(f"{path.name}: ids differ")
    targets: list[Any] = []
    for i, (c, kind) in enumerate(zip(cache, kinds, strict=True)):
        if kind == "kv":
            try:
                offset = int(meta[f"c{i}_offset"])
                keys, values = (
                    loaded[f"c{i}_k"],  # ty: ignore[invalid-argument-type]
                    loaded[f"c{i}_v"],  # ty: ignore[invalid-argument-type]
                )
            except (KeyError, ValueError) as e:
                raise ForkFileError(f"{path.name}: layer {i} incomplete") from e
            if offset != n or keys.shape[2] < n or values.shape[2] < n:
                raise ForkFileError(f"{path.name}: layer {i} offset {offset} for {n} ids")
            c.keys, c.values, c.offset = keys, values, offset
            targets += [keys, values]
        else:
            try:
                size = int(meta[f"c{i}_size"])
            except (KeyError, ValueError) as e:
                raise ForkFileError(f"{path.name}: layer {i} incomplete") from e
            if size != len(c.cache):
                raise ForkFileError(
                    f"{path.name}: layer {i} has {size} states, want {len(c.cache)}"
                )
            states: list[Any] = []
            for j in range(size):
                if meta.get(f"c{i}_s{j}_none") == "1":
                    states.append(None)
                elif f"c{i}_s{j}" in loaded:
                    states.append(loaded[f"c{i}_s{j}"])  # ty: ignore[invalid-argument-type]
                    targets.append(states[-1])
                else:
                    raise ForkFileError(f"{path.name}: layer {i} state {j} missing")
            c.state = states
    mx.eval(*targets)
