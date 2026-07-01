"""
Build SFT WDYL dataset for EAR conversational-repair training.
"""

import argparse
import io
import os
import re
import time

import librosa
import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from openai import OpenAI

AUDIO_SAMPLING_RATE = 16000


def decode_audio_field(field):
    raw = field.get("bytes")
    assert raw

    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)

    assert getattr(arr, "ndim", 1) == 1
    if sr != AUDIO_SAMPLING_RATE:
        # origin sr of the ear_dataset is 44100
        arr = librosa.resample(arr, orig_sr=sr, target_sr=AUDIO_SAMPLING_RATE)

    return arr


def build_masked_transcript(a_content, q_content, ground_truth):
    """Returns (masked_transcript, )"""
    gt = ground_truth.strip()

    def _mask(text):
        pat = re.compile(r"\b" + re.escape(gt) + r"\b", re.IGNORECASE)
        m = pat.search(text)
        if m:
            return text[: m.start()] + f"[UNCERTAIN: {m.group(0)}]" + text[m.end() :]
        idx = text.lower().find(gt.lower())
        if idx != -1:
            return text[:idx] + f"[UNCERTAIN: {gt}]" + text[idx + len(gt) :]
        return None

    for text in (a_content, q_content):
        masked = _mask(text)
        if masked is not None:
            return masked

    return None


TEACHER_SYSTEM = (
    "You label training data for an audio question-answering assistant. Write one "
    "short, natural clarification the assistant would say when it could not hear "
    "the answer. Under 30 words."
)


def teacher_user(question, masked, gt):
    """Build a user prompt for teacher query"""
    lines = [f'A user asked: "{question}"']

    lines.append(
        f'The other speaker answered, but the key word was masked by noise:\n"{masked}"'
    )
    lines.append(
        f'For labeling only, the masked word was "{gt}" -- never reveal or guess it.\n'
        "Write ONE clarification that (1) signals the detail could not be heard, "
        '(2) asks for the missing category the question implies (e.g. "which city", '
        f'"what food", "who"), (3) never states or guesses "{gt}", (4) is under 20 words.'
    )
    return "\n".join(lines)


def get_repair(client, model, question, masked, gt):
    assert masked
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.7,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": teacher_user(question, masked, gt)},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text and gt.lower() not in text.lower():
                return text
        except Exception as e:
            print("teacher error:", e)
            time.sleep(2)
    raise Exception("couldn't reach gpt-4o for repair")


def test():
    ds = load_dataset("keylazy/ear-datasets", split="train")
    ds = ds.cast_column("answerable_audio", Audio(decode=False))
    ds = ds.cast_column("unanswerable_audio", Audio(decode=False))

    for row in ds:
        arr = decode_audio_field(row["answerable_audio"])
        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keylazy/ear-datasets")
    ap.add_argument("--split", default="train")
    ap.add_argument("--row-start", type=int, default="50")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out-repo", default="keylazy/ear-omni-sft")
    ap.add_argument("--teacher-model", default="gpt-4o")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    client = OpenAI()

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("answerable_audio", Audio(decode=False))
    ds = ds.cast_column("unanswerable_audio", Audio(decode=False))
    end = (
        len(ds)
        if args.max_rows is None
        else min(args.row_start + args.max_rows, len(ds))
    )

    rows = []
    for ridx in range(args.row_start, end):
        row = ds[ridx]
        gt = str(row["ground_truth"])
        masked = build_masked_transcript(row["a_content"], row["q_content"], gt)
        ans_audio = {
            "array": decode_audio_field(row["answerable_audio"]),
            "sampling_rate": AUDIO_SAMPLING_RATE,
        }
        una_audio = {
            "array": decode_audio_field(row["unanswerable_audio"]),
            "sampling_rate": AUDIO_SAMPLING_RATE,
        }

        repair = None
        for qcol in ("question_0", "question_1"):
            q = row[qcol]
            rows.append({
                "id": row["id"],
                "question": q,
                "target": gt,
                "kind": "answer",
                "audio": ans_audio
            })
            repair = get_repair(client, args.teacher_model, q, masked, gt)
            rows.append({
                "id": row["id"],
                "question": q,
                "target": repair,
                "kind": "repair",
                "audio": una_audio
            })
    
    out = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)).shuffle(seed=0)
    print(f"built {len(out)} examples")

    if args.push:
        out.push_to_hub(args.out_repo)
        print("pushed ->", args.out_repo)
    else:
        print("not pushed (add --push)")


if __name__ == "__main__":
    main()
