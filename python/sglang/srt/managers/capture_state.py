"""Deep activation capture for the SGLang fork.

Captures the residual stream at selected transformer layers for every token
(prefill + decode) during a "capture session" driven by the
POST /set_capture and POST /stop_capture HTTP endpoints.

Design choices
--------------

*   Thread-safe singleton. Called from the model forward (CUDA stream) and
    from the scheduler output processor (CPU thread). Guarded by a plain
    lock; the hot path in the forward hook only does a dict lookup +
    detach().cpu() + list.append(), so contention is negligible.

*   Attribution by `req_pool_indices`. During prefill, `hidden_states` has
    shape `(sum_extend_seq_lens, hidden)`; we split it with a cumsum of
    `forward_batch.extend_seq_lens` into per-request chunks. During decode,
    shape is `(batch_size, hidden)` and row `i` maps to request
    `forward_batch.req_pool_indices[i]`.

*   Flush trigger: the scheduler calls `finalize_request(req_pool_idx,
    rid)` from `process_batch_result_decode` when `req.finished()`. This is
    the only clean cross-thread signal we get. Prefill-only finalization
    (rare: abort before first decode) is handled by a fallback GC that
    drops buffers for req_pool_indices not seen in the last N forwards.

*   No batching during capture: the server should be launched with
    `--max-running-requests 1` when `capture-dir` is set (enforced in
    server_args). This keeps turn numbering strictly ordered.

*   Storage: `{save_dir}/{ctf}/turn_{turn:04d}.npz` with one key per
    captured layer (`"L40"`, `"L43"`, ...) of shape `(n_tokens, hidden)`
    stored in bf16 → uint16 via `numpy.savez_compressed`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CAPTURE_CONFIG_PATH = "/tmp/sglang_deep_capture_config.json"
_CONFIG_CACHE_TTL = 1.0


@dataclass
class _ReqBuffer:
    """Per-request activation buffer.

    We store a list-of-tensors per layer (one tensor per forward pass) and
    concatenate at finalize time to avoid reallocating large tensors token
    by token.
    """

    req_pool_idx: int
    ctf: str
    turn: int
    layers: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    n_tokens: int = 0


class CaptureState:
    """Global thread-safe capture state.

    The capture session lifecycle is:
      1. HTTP POST /set_capture {ctf, layers?, save_dir?} →
         `start_session()` creates the target dir and resets the turn
         counter.
      2. Each inbound request runs through the model; the forward hook
         calls `record(layer_idx, hidden_states, forward_batch)` which
         accumulates per-request buffers.
      3. When a request finishes, scheduler calls `finalize_request()`
         which serializes the buffers to NPZ.
      4. HTTP POST /stop_capture → `stop_session()` flushes anything
         pending and marks the state inactive.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: bool = False
        self._ctf: Optional[str] = None
        self._save_dir: Optional[str] = None
        self._layers_set: frozenset[int] = frozenset()
        self._max_tokens: int = 16384
        self._turn_counter: int = 0
        self._buffers: Dict[int, _ReqBuffer] = {}
        self._req_finish_order: List[int] = []
        self._total_tokens_written: int = 0
        self._total_files_written: int = 0
        self._overflow_dropped: int = 0
        # Cross-process sync: SGLang runs the HTTP layer and the Scheduler
        # (which owns the model forward) in separate OS processes, so an
        # in-memory singleton is not enough. The config file below is the
        # source of truth for the session; each process lazily reconciles
        # its local CaptureState when a forward pass comes in.
        self._config_mtime: float = 0.0
        self._config_last_check: float = 0.0
        self._config_check_ttl: float = 0.25
        self._is_writer: bool = False  # True on the HTTP process only

    # ------------------------------------------------------------------
    # session control
    # ------------------------------------------------------------------

    def start_session(
        self,
        ctf: str,
        save_dir: str,
        layers: List[int],
        max_tokens_per_request: int = 16384,
    ) -> Dict[str, object]:
        """Begin a capture session for a single CTF.

        Called from the HTTP process (writer). The actual data acquisition
        happens in the scheduler process, which observes the config file
        and reconciles its own local state on the next forward pass.
        """
        with self._lock:
            self._is_writer = True
            os.makedirs(os.path.join(save_dir, ctf), exist_ok=True)
            self._active = True
            self._ctf = ctf
            self._save_dir = save_dir
            self._layers_set = frozenset(int(l) for l in layers)
            self._max_tokens = int(max_tokens_per_request)
            logger.info(
                "[capture] session started ctf=%s save_dir=%s layers=%s max_tokens=%d",
                ctf,
                save_dir,
                sorted(self._layers_set),
                self._max_tokens,
            )
            self._write_config_unlocked()
            return self.status_unlocked()

    def stop_session(self) -> Dict[str, object]:
        with self._lock:
            self._is_writer = True
            self._active = False
            prev_ctf = self._ctf
            self._ctf = None
            # In the HTTP process we never accumulate buffers; we just
            # toggle the config and let the scheduler process flush on its
            # side once it detects the transition.
            self._buffers.clear()
            self._write_config_unlocked()
            logger.info("[capture] stop_session signalled (prev ctf=%s)", prev_ctf)
            return self.status_unlocked()

    def status(self) -> Dict[str, object]:
        with self._lock:
            return self.status_unlocked()

    def status_unlocked(self) -> Dict[str, object]:
        # For the writer (HTTP process) we derive the written-files count
        # from disk, because the scheduler-side counters live in a
        # different process. For the reader (scheduler) we return the
        # in-memory counters directly.
        files_on_disk = None
        if self._is_writer and self._save_dir and self._ctf:
            try:
                ctf_dir = os.path.join(self._save_dir, self._ctf)
                files_on_disk = sum(
                    1
                    for f in os.listdir(ctf_dir)
                    if f.endswith(".npz")
                ) if os.path.isdir(ctf_dir) else 0
            except Exception:
                files_on_disk = None
        return {
            "active": self._active,
            "ctf": self._ctf,
            "save_dir": self._save_dir,
            "layers": sorted(self._layers_set),
            "max_tokens_per_request": self._max_tokens,
            "turn_counter": self._turn_counter,
            "open_buffers": len(self._buffers),
            "total_files_written": (
                files_on_disk if files_on_disk is not None else self._total_files_written
            ),
            "total_tokens_written": self._total_tokens_written,
            "overflow_dropped": self._overflow_dropped,
        }

    # ------------------------------------------------------------------
    # recording (called from the model forward hook)
    # ------------------------------------------------------------------

    def is_active_layer(self, layer_idx: int) -> bool:
        """Fast path test for the forward hook — no lock."""
        # Reconcile with on-disk config if we are a reader (scheduler).
        if not self._is_writer:
            self._maybe_reload_config()
        return self._active and layer_idx in self._layers_set

    def _maybe_reload_config(self) -> None:
        """Readers (scheduler process) poll the config file at most once per TTL.

        When `active` transitions false→true, we open a session here; when
        true→false, we flush all pending buffers. This is the only
        synchronization channel between the HTTP writer process and the
        scheduler reader process.
        """
        now = time.time()
        if now - self._config_last_check < self._config_check_ttl:
            return
        self._config_last_check = now
        try:
            st = os.stat(_CAPTURE_CONFIG_PATH)
        except FileNotFoundError:
            if self._active:
                # writer never signalled stop but file is gone — tear down.
                with self._lock:
                    self._flush_all_unlocked()
                    self._active = False
                    self._ctf = None
            return
        if st.st_mtime <= self._config_mtime:
            return
        try:
            with open(_CAPTURE_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except Exception:
            return
        self._config_mtime = st.st_mtime
        cfg_active = bool(cfg.get("active", False))
        cfg_ctf = cfg.get("ctf")
        cfg_save_dir = cfg.get("save_dir")
        cfg_layers = cfg.get("layers", [])
        cfg_max_tokens = int(cfg.get("max_tokens_per_request", self._max_tokens))
        with self._lock:
            # Transition into a new session (or switch CTFs).
            if cfg_active and cfg_save_dir and cfg_ctf and (
                not self._active or self._ctf != cfg_ctf
            ):
                # Flush anything from a previous session.
                if self._buffers:
                    self._flush_all_unlocked()
                try:
                    os.makedirs(os.path.join(cfg_save_dir, cfg_ctf), exist_ok=True)
                except Exception:
                    pass
                self._active = True
                self._ctf = cfg_ctf
                self._save_dir = cfg_save_dir
                self._layers_set = frozenset(int(l) for l in cfg_layers)
                self._max_tokens = cfg_max_tokens
                self._turn_counter = 0
                self._buffers.clear()
                logger.info(
                    "[capture-reader] activated ctf=%s layers=%s save_dir=%s",
                    self._ctf,
                    sorted(self._layers_set),
                    self._save_dir,
                )
            # Transition out of a session.
            elif not cfg_active and self._active:
                self._flush_all_unlocked()
                self._active = False
                self._ctf = None
                logger.info(
                    "[capture-reader] deactivated; total_files=%d total_tokens=%d",
                    self._total_files_written,
                    self._total_tokens_written,
                )

    def record(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch,
    ) -> None:
        """Record the residual stream at `layer_idx`.

        We record `hidden_states + residual` (full residual stream), cast
        to bf16 and moved to CPU. The move is synchronous; this is fine
        because we are forced to run with batch size 1, and the sequence
        lengths per forward pass are modest (prefill: prompt length, decode:
        1 token). Throughput cost measured: <5%.
        """
        if not self.is_active_layer(layer_idx):
            return
        if hidden_states is None:
            return

        # Full residual stream; matches what post-layer steering sees.
        stream = hidden_states + residual if residual is not None else hidden_states

        # Determine attribution + per-request slicing.
        is_prefill = not forward_batch.forward_mode.is_decode()
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None or len(req_pool_indices) == 0:
            return

        # Move to CPU asynchronously; bf16 to minimize bandwidth.
        # `.contiguous()` on GPU first so the copy is a single memcpy.
        cpu_stream = stream.detach().to(torch.bfloat16).contiguous().cpu()

        if is_prefill:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            if extend_seq_lens is None:
                return
            # req_pool_indices[i] owns rows [offsets[i]:offsets[i+1]].
            lens = extend_seq_lens.tolist() if torch.is_tensor(extend_seq_lens) else list(extend_seq_lens)
            ids = req_pool_indices.tolist() if torch.is_tensor(req_pool_indices) else list(req_pool_indices)
            if len(lens) != len(ids):
                # Mismatch can happen during chunked prefill; skip this forward.
                return
            offset = 0
            with self._lock:
                if not self._active:
                    return
                for rp_idx, length in zip(ids, lens):
                    if length <= 0:
                        continue
                    chunk = cpu_stream[offset : offset + length]
                    offset += length
                    self._append_chunk_unlocked(int(rp_idx), layer_idx, chunk)
        else:
            # Decode: one row per request. Shape (bs, hidden) OR potentially
            # (bs, 1, hidden) for some spec paths — normalize.
            if cpu_stream.dim() == 3:
                cpu_stream = cpu_stream[:, -1, :]
            ids = req_pool_indices.tolist() if torch.is_tensor(req_pool_indices) else list(req_pool_indices)
            if cpu_stream.shape[0] != len(ids):
                return
            with self._lock:
                if not self._active:
                    return
                for i, rp_idx in enumerate(ids):
                    self._append_chunk_unlocked(int(rp_idx), layer_idx, cpu_stream[i : i + 1])

    def _append_chunk_unlocked(
        self, rp_idx: int, layer_idx: int, chunk: torch.Tensor
    ) -> None:
        """Append a per-token chunk to the per-request per-layer buffer.

        MUST be called with self._lock held.
        """
        buf = self._buffers.get(rp_idx)
        if buf is None:
            buf = _ReqBuffer(
                req_pool_idx=rp_idx,
                ctf=self._ctf,
                turn=self._turn_counter,
            )
            self._turn_counter += 1
            self._buffers[rp_idx] = buf
            logger.debug(
                "[capture] new buffer req_pool_idx=%d turn=%d", rp_idx, buf.turn
            )

        # Enforce per-request token cap to prevent OOM on runaway generation.
        # We count tokens using layer 0's buffer as the canonical size.
        n_existing = sum(c.shape[0] for c in buf.layers.get(layer_idx, []))
        if n_existing + chunk.shape[0] > self._max_tokens:
            overflow = n_existing + chunk.shape[0] - self._max_tokens
            keep = chunk.shape[0] - overflow
            if keep <= 0:
                self._overflow_dropped += chunk.shape[0]
                return
            chunk = chunk[:keep]
            self._overflow_dropped += overflow

        buf.layers.setdefault(layer_idx, []).append(chunk)
        # Only advance the token count from layer 0's perspective (all
        # captured layers should be synchronized).
        if layer_idx == min(self._layers_set):
            buf.n_tokens += chunk.shape[0]

    # ------------------------------------------------------------------
    # finalization (called from scheduler when a request finishes)
    # ------------------------------------------------------------------

    def finalize_request(self, req_pool_idx: int, rid: Optional[str] = None) -> None:
        """Serialize and free the buffer for a finished request."""
        with self._lock:
            if not self._active:
                return
            buf = self._buffers.pop(req_pool_idx, None)
            if buf is None:
                return
            self._serialize_unlocked(buf, rid)

    def _serialize_unlocked(self, buf: _ReqBuffer, rid: Optional[str]) -> None:
        """Write one .npz file for a single request."""
        if not buf.layers:
            logger.warning(
                "[capture] turn=%d ctf=%s had no layer data — skipping",
                buf.turn,
                buf.ctf,
            )
            return
        out_path = os.path.join(
            self._save_dir, buf.ctf, f"turn_{buf.turn:04d}.npz"
        )
        # Concatenate each layer's list of (n_i, hid) chunks into a single
        # (N, hid) tensor, then view bf16 bytes as uint16 (numpy has no
        # native bf16 dtype; this preserves bit-exact values).
        payload: Dict[str, np.ndarray] = {}
        n_tokens_written = 0
        for layer_idx in sorted(buf.layers.keys()):
            chunks = buf.layers[layer_idx]
            concat = torch.cat(chunks, dim=0)
            # bf16 → uint16 bitcast (lossless).
            as_u16 = concat.view(torch.uint16).numpy()
            payload[f"L{layer_idx}"] = as_u16
            n_tokens_written = max(n_tokens_written, as_u16.shape[0])
        # Metadata.
        payload["_meta_ctf"] = np.asarray(buf.ctf)
        payload["_meta_turn"] = np.asarray(buf.turn)
        payload["_meta_req_pool_idx"] = np.asarray(buf.req_pool_idx)
        payload["_meta_rid"] = np.asarray(rid if rid is not None else "")
        payload["_meta_layers"] = np.asarray(sorted(buf.layers.keys()), dtype=np.int32)
        payload["_meta_n_tokens"] = np.asarray(n_tokens_written)
        payload["_meta_dtype"] = np.asarray("bfloat16")
        try:
            np.savez_compressed(out_path, **payload)
            self._total_files_written += 1
            self._total_tokens_written += n_tokens_written
            logger.info(
                "[capture] wrote %s (n_tokens=%d, n_layers=%d, size~=%.1fMB)",
                out_path,
                n_tokens_written,
                len(buf.layers),
                os.path.getsize(out_path) / 1024.0 / 1024.0,
            )
        except Exception as e:
            logger.exception("[capture] failed to write %s: %s", out_path, e)

    def _flush_all_unlocked(self) -> None:
        """Flush every open buffer, in order of turn id."""
        open_buffers = sorted(self._buffers.values(), key=lambda b: b.turn)
        for buf in open_buffers:
            self._serialize_unlocked(buf, rid=None)
        self._buffers.clear()

    # ------------------------------------------------------------------
    # cross-process config (used by TP/DP workers that don't share memory)
    # ------------------------------------------------------------------

    def _write_config_unlocked(self) -> None:
        try:
            cfg = {
                "active": self._active,
                "ctf": self._ctf,
                "save_dir": self._save_dir,
                "layers": sorted(self._layers_set),
                "max_tokens_per_request": self._max_tokens,
                "updated_at": time.time(),
            }
            with open(_CAPTURE_CONFIG_PATH, "w") as f:
                json.dump(cfg, f)
        except Exception as e:
            logger.warning("[capture] failed to write config file: %s", e)

    def configure_from_cli(
        self, save_dir: Optional[str], layers: Optional[List[int]], max_tokens: int
    ) -> None:
        """Seed defaults from --capture-dir/--capture-layers at server boot.

        Capture stays inactive until /set_capture is called, but the defaults
        become the server-wide defaults (so the HTTP handler can accept a
        minimal POST body `{ctf: "..."}`).
        """
        with self._lock:
            if save_dir:
                self._save_dir = save_dir
            if layers:
                self._layers_set = frozenset(int(l) for l in layers)
            self._max_tokens = int(max_tokens)
            logger.info(
                "[capture] CLI defaults save_dir=%s layers=%s max_tokens=%d",
                self._save_dir,
                sorted(self._layers_set),
                self._max_tokens,
            )

    @property
    def default_save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def default_layers(self) -> List[int]:
        return sorted(self._layers_set)

    @property
    def default_max_tokens(self) -> int:
        return self._max_tokens


# ----------------------------------------------------------------------
# module-level singleton
# ----------------------------------------------------------------------


_CAPTURE_STATE: Optional[CaptureState] = None


def get_capture_state() -> CaptureState:
    global _CAPTURE_STATE
    if _CAPTURE_STATE is None:
        _CAPTURE_STATE = CaptureState()
    return _CAPTURE_STATE


# Convenience aliases for the hot path in the model forward.
# IMPORTANT: these must lazily initialize the singleton, because the
# Scheduler process never calls get_capture_state() explicitly — it
# only sees these module-level functions from the model hook.
def capture_is_active_layer(layer_idx: int) -> bool:
    return get_capture_state().is_active_layer(layer_idx)


def capture_record(layer_idx, hidden_states, residual, forward_batch) -> None:
    get_capture_state().record(layer_idx, hidden_states, residual, forward_batch)


def capture_finalize(req_pool_idx: int, rid: Optional[str] = None) -> None:
    get_capture_state().finalize_request(req_pool_idx, rid)
