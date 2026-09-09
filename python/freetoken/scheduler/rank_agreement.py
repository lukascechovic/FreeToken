"""Agree an outcome across TP ranks before it changes control flow (llm-server #871, patch 0022).

Every rank runs its own scheduler process (``server/launch.py:_run_scheduler``, one ``mp.Process``
per rank) over the same broadcast message stream, so control flow is identical on every rank as
long as every branch is decided by the REQUEST. ``_attach_mm_embeds`` has one branch that is not:
the ``except`` around the vision encode, whose outcome depends on that rank's card at that instant.
The vision tower is replicated rather than sharded, so nothing inside the encode keeps the ranks in
step, and the two cards are genuinely in different states -- #871 recorded them releasing memory
independently, interleaved in time and different in size, with rank 0 reliably the one that runs out.

When they disagree the refusing rank returns to idle and its peer enters the forward and blocks in
the embedding all-reduce. Nothing errors: 60 s later NCCL's watchdog times out
(``distributed_timeout: float = 60.0``, ``engine/config.py``, which has no CLI flag) and takes the
process down, and by then the OOM has scrolled off the tail of the log.

⭐ The agreement is a CPU-side all-reduce of one int64, taking the MAX -- the same primitive
``engine.py``'s ``_sync_get_memory`` already uses over ``tp_cpu_group`` (with ``op=MIN``), which is
this module's precedent. ``scheduler.py`` names the gap it fills as *"the all-rank failure-agreement
machinery (deferred)"*.

⛔ ``sync_all_ranks()`` (``scheduler/io.py``) is NOT this: it is ``barrier().wait()``, which agrees
on ARRIVAL and carries no value. The ranks here arrive at the same place holding different answers,
so a barrier would let both through still disagreeing.

⭐ ``tp_cpu_group`` is a **gloo** group in BOTH branches of ``engine._init_communication`` -- under
``--disable-pynccl`` it is an explicit ``new_group(backend="gloo")``, and otherwise it is a WORLD
group that was itself initialised with ``backend="gloo"`` (pynccl carries the device traffic). So
this collective can never touch the NCCL path on any configuration, which is the property that
makes it safe to add to a served row.

⚠ Cost: one small collective per multimodal request, on a path that already runs a whole vision
tower. ``scheduler/io.py``'s receive loop already broadcasts over this same group on EVERY
iteration, so the agreement is strictly rarer than gloo traffic the row ships today.
"""
from __future__ import annotations

from typing import Callable, TypeAlias

import torch

__all__ = ["any_rank_failed"]

ReduceMax: TypeAlias = Callable[[torch.Tensor], None]


def any_rank_failed(local_failed: bool, tp_size: int, reduce_max: ReduceMax) -> bool:
    """Did ANY rank fail? Returns the same answer on every rank.

    ``reduce_max`` is the in-place max all-reduce over the TP CPU group -- injected rather than
    reached for, so the agreement can be tested without standing up a process group, and so the
    caller owns the group and its timeout.

    ⛔ At ``tp_size <= 1`` the collective is not reached at all. A single-card row cannot disagree
    with itself, and must not pay -- or risk -- one instruction of distributed machinery for it.
    Both deployed single-card vision rows take this line and nothing else new.
    """
    if tp_size <= 1:
        return local_failed
    # int64 on the HOST: gloo takes host tensors, and a device tensor here would put the agreement
    # back on the card -- reintroducing exactly the coupling this exists to remove.
    flag = torch.tensor([1 if local_failed else 0], dtype=torch.int64, device="cpu")
    reduce_max(flag)
    return bool(int(flag.item()) != 0)
