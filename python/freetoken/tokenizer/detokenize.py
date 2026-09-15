from dataclasses import dataclass
from typing import Dict, FrozenSet, List

from freetoken.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# ── #801 overlay marker ──────────────────────────────────────────────────────────────────────
# This file is `tokenizer/detokenize.py` from image
# `llm-server/freetoken-gfx1201:2026-09-09-agree-0022` (md5 3d20c1e8e4b3244040f433f5588addad,
# 147 lines) plus ONE edit: `DetokenizeManager.detokenize` is now a dispatcher that runs the
# image's own body once per ROUND, and that body is the image's `detokenize` renamed to
# `_ft801_detokenize_round`, byte for byte. ⛔ In no image, in no Dockerfile ladder:
# `arm_mtp_801.sh`'s OVERLAY and ORIGS lists or the tokenizer worker runs the image's copy
# silently (#866). The differential is gated by `test_detokenize_801.py`, on the HOST.
#
# ⛔⛆ WHY THE TOKENIZER WORKER IS TOUCHED AT ALL, AND WHY NINETEEN LOADS WALKED PAST IT.
#   `detokenize` takes a LIST. Its first loop appends every message's token and slices the ids
#   with `surr_offset`/`read_offset`; the SECOND loop is what advances those two offsets. At one
#   message per uid per batch that is correct -- and one message per uid per batch is exactly what
#   a one-token-per-step engine produces. It is the image's own invariant, and nothing in the
#   image states it. #801's verify step commits 1 OR 2 tokens per request per forward and
#   `scheduler.py`'s drain ships them in ONE reply list, so the second message is sliced against
#   the FIRST message's pre-append offsets and its `new_text` re-includes the first token:
#       control  'We need to respond to user:'
#       verify   'We need to respond respond to user user:'
#   ⭐⭐⭐ That is load 19's warmup to the byte, and THE COMMIT PATH WAS RIGHT THE WHOLE TIME. One
#   reply goes out per message either way (`tokenizer/server.py` stamps `completion_tokens_delta=1`
#   per message and zips `strict=True`), which is why the load reported 61 chunks and 61 completion
#   tokens while the text inside them carried ~86 tokens' worth -- the duplication rides INSIDE a
#   reply, where no count can see it.
#
# ⭐ IDENTICAL FOR AN ENGINE THAT COMMITS ONE TOKEN A STEP: such a batch holds at most one message
#   per uid, which is ONE round, which is the image's body called once on the whole list. Gated
#   reply-for-reply against the `.orig` over seven batch shapes rather than argued.
import os as _ft801_os
import sys as _ft801_sys

print(
    "[#801] overlay ACTIVE: tokenizer/detokenize.py bind-mounted from the repo "
    f"(pid {_ft801_os.getpid()}, base md5 3d20c1e8e4b3244040f433f5588addad, "
    f"FREETOKEN_MTP801_VERIFY={_ft801_os.getenv('FREETOKEN_MTP801_VERIFY', '<unset>')})",
    file=_ft801_sys.stderr,
    flush=True,
)

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def _stop_prefix_holdback(text: str, stop_strs: list[str]) -> int:
    """Length of the longest trailing suffix of ``text`` that is a proper prefix of some
    stop string. Those chars are withheld until a later token resolves whether the stop
    completes, so a partial stop is never streamed and then needs retracting."""
    hold = 0
    for stop in stop_strs:
        for i in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:i]):
                hold = max(hold, i)
                break
    return hold


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string


def _ft801_rounds_by_uid(msgs: List[DetokenizeMsg]) -> List[List[int]]:
    """``msgs`` split into rounds of INDICES, each round holding at most one message per uid.

    ⭐ A uid's k-th message lands in round k: every earlier round already holds that uid, and the
    first round that does not is the one it joins. So a request's tokens keep their order, and two
    requests never wait on each other -- a round fills up with DISTINCT uids, so a batch of plain
    decode rows is still one round however wide it is.

    ⚠ Indices, not messages. The caller owes one reply per message IN THE ORDER IT WAS GIVEN --
    `tokenizer/server.py` zips the replies with the messages ``strict=True`` and stamps a
    ``completion_tokens_delta`` per pair -- and an index is what puts a round's replies back where
    they came from.
    """
    rounds: List[List[int]] = []
    seen: List[set] = []
    for index, msg in enumerate(msgs):
        for members, uids in zip(rounds, seen):
            if msg.uid not in uids:
                members.append(index)
                uids.add(msg.uid)
                break
        else:
            rounds.append([index])
            seen.append({msg.uid})
    return rounds


class DetokenizeManager:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, eos_token_ids: FrozenSet[int] | None = None
    ) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else frozenset({tokenizer.eos_token_id})
        )

    def discard(self, uid: int) -> None:
        """Drop a uid's decode state without a finished DetokenizeMsg. An aborted or errored
        request never sends one (its terminal reply is an ErrorReplyMsg), so without this its
        accumulated ids and text stay in ``decode_map`` for the life of the worker."""
        self.decode_map.pop(uid, None)

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """One reply per message, in the order given -- the image's contract, unchanged.

        ⛔⛆ THE ROUNDS ARE THE WHOLE EDIT, and the marker block at the top of this file says what
        they are for: `_ft801_detokenize_round` below is the image's own body, and it is correct
        only for a batch holding at most one message per uid. #801's verify step commits two
        tokens for one request on one forward, and the drain ships both in one list.

        ⚠ One `batch_decode` pair per round, not per message: a batch that never speculates is one
        round and costs exactly what it costs today. A serving batch where every row accepted its
        draft is two.
        """
        replies: List[str] = [""] * len(msgs)
        for members in _ft801_rounds_by_uid(msgs):
            for index, reply in zip(
                members, self._ft801_detokenize_round([msgs[i] for i in members]), strict=True
            ):
                replies[index] = reply
        return replies

    def _ft801_detokenize_round(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """⛔ The image's `detokenize`, byte for byte -- NEVER call it directly.

        It is correct only for ``msgs`` holding at most one message per uid; `detokenize` above is
        what guarantees that. `test_detokenize_801.py` asserts this body against the `.orig`'s, so
        an image bump that moves it is a failing gate rather than a silent fork.
        """
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            prev_sent = s.sent_offset
            if msg.finished:
                # Generation is over: flush everything, trimming at the matched stop string.
                if msg.matched_stop:
                    cut = output_str.find(msg.matched_stop)
                    emit_end = cut if cut >= 0 else len(output_str)
                else:
                    emit_end = len(output_str)
            elif msg.stop_strs:
                # Hold back a trailing suffix that could still grow into a stop string.
                emit_end = len(output_str) - _stop_prefix_holdback(output_str, msg.stop_strs)
            else:
                emit_end = len(output_str)
            incremental_output = output_str[prev_sent:emit_end] if emit_end > prev_sent else ""
            s.sent_offset = max(prev_sent, emit_end)
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
