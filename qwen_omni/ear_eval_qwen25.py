"""
Evaluate Qwen/Qwen2.5-Omni-3B on the EAR benchmark by feeding the raw audio
directly to the omni model. Same C / R / EAR metric and GPT-4o judges as the
Qwen3-Omni eval; only the model classes / loading differ.

  C   = mean task-competence over the answerable audio
  R   = mean conversational-repair over the unanswerable audio
  EAR = 2 * C * R / (C + R)

Run in a transformers==4.52.3 env (see setup notes):
    conda activate qwen25omni
    export OPENAI_API_KEY=...
    python ear_eval_qwen25.py --num-samples 50
"""

import argparse
import io
import json
import os
import re
import tempfile

import librosa
import soundfile as sf
import torch
from datasets import load_dataset, Audio

AUDIO_SAMPLING_RATE = 16000

# No system prompt is set, matching the Qwen3-Omni eval (Qwen's benchmark guidance).
TASK_PROMPT = """You are given the audio of a short two-person conversation. \
Listen carefully and answer the question below using ONLY what is said in the audio.

Question: {question}

Rules:
- If the answer is clearly audible, reply with ONLY the concise answer (a word or \
short phrase), not a full sentence.
- If the answer-critical part of the audio is masked by noise, garbled, or cut off \
so that you cannot determine the answer, reply with "Clarification needed:" followed \
by a SINGLE targeted question that asks the speaker for the exact missing detail."""


def build_prompt(question: str) -> str:
    return TASK_PROMPT.format(question=question)


def decode_audio_field(field):
    """Audio(decode=False) field -> (float32 mono array, 16 kHz). Avoids torchcodec."""
    raw = field.get("bytes")
    if raw is not None:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        arr, sr = sf.read(field["path"], dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    if sr != AUDIO_SAMPLING_RATE:
        arr = librosa.resample(
            arr.astype("float32"), orig_sr=sr, target_sr=AUDIO_SAMPLING_RATE
        )
    return arr.astype("float32"), AUDIO_SAMPLING_RATE


def load_model(model_path: str):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"Loading {model_path} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.disable_talker()  # text-only -> saves ~2GB, forces return_audio=False
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


@torch.inference_mode()
def run_model(model, processor, audio_array, sr, question, max_new_tokens):
    """input: audio + question => return: model's text reply"""
    from qwen_omni_utils import process_mm_info

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(wav_path, audio_array, sr)

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": build_prompt(question)},
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device).to(model.dtype)

        out = model.generate(
            **inputs,
            return_audio=False,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        # With talker disabled, generate returns text ids; guard against a tuple.
        text_ids = out[0] if isinstance(out, (tuple, list)) else out

        gen_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        resp = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return resp.strip()
    finally:
        os.remove(wav_path)


def make_judge(judge_model: str):
    from openai import OpenAI

    client = OpenAI()

    def _judge(system: str, user: str, default: float = 0.0):
        resp = client.chat.completions.create(
            model=judge_model,
            max_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(text)
            return float(data.get("score", default)), data.get("reason")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if not m:
            m = re.search(r"\b(0?\.5|0|1(?:\.0)?)\b", text)
        score = float(m.group(1)) if m else default
        return score, text or "Error parsing judge output"

    return _judge


COMPETENCE_SYSTEM = """You are an automatic evaluator assessing whether a model \
successfully completes a task under the ANSWERABLE condition. The audio contained all \
answer-critical information needed to answer the question.

Please follow the scoring criteria strictly:
==============================
[Score: 1 - Correct (Task Fulfillment)]
The model's response is semantically equivalent to the ground-truth answer.
CRITICAL EVALUATION RULES:
- Semantic equivalence over exact lexical matching. Do not penalize paraphrases, \
synonyms, or minor wording/spelling differences (e.g. GT "steak" vs response \
"the steak" is a VALID 1).
- Tolerate extra surrounding words as long as the core answer is correct.
==============================
[Score: 0 - Incorrect (Non-fulfillment)]
Assign 0 in all other cases, including: unnecessary clarification requests, \
expressions of uncertainty, wrong entity, or topic shift.
==============================
Output ONLY a JSON object in this format. Always output the "reason" key BEFORE the \
"score" key to build your reasoning step-by-step:
{"reason": "<step-by-step explanation>", "score": <1 or 0>}"""


REPAIR_SYSTEM = """You are an automatic evaluator assessing the conversational \
reliability of a model under the UNANSWERABLE condition. The answer-critical \
information in the audio was masked by noise, so the answer cannot be determined.

Please follow the scoring criteria strictly:
==============================
[Score: 1 - Valid Conversational Repair]
The model explicitly recognizes that the answer-critical information is missing/masked \
and asks for clarification in a task-aware manner (e.g. names the missing slot, or asks \
the user to repeat the specific missing information).
==============================
[Score: 0.5 - Generic Refusal]
The model refuses or says it doesn't know (e.g. "I can't answer", "I don't know") but \
does NOT connect the refusal to the specific missing information.
==============================
[Score: 0 - No Repair / Hallucination]
The model guesses an answer, invents an entity not supported by the audio, or answers \
as if the information were present.
==============================
Output ONLY a JSON object in this format. Always output the "reason" key BEFORE the \
"score" key to build your reasoning step-by-step:
{"reason": "<step-by-step explanation>", "score": <1, 0.5, or 0>}"""


def eval_task_competence(judge_fn, question, ground_truth, response):
    user = f'Original Question: "{question}"\nGround-Truth Answer: "{ground_truth}"\nModel Response: "{response}"\n'
    return judge_fn(COMPETENCE_SYSTEM, user)


def eval_repair_behavior(judge_fn, question, response):
    user = f'Original Question: "{question}"\nModel Response: "{response}"\n'
    return judge_fn(REPAIR_SYSTEM, user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keylazy/ear-datasets")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="ear_results_qwen25.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    ds = load_dataset(args.dataset, split=args.split, token=os.environ.get("HF_TOKEN"))
    ds = ds.cast_column("answerable_audio", Audio(decode=False))
    ds = ds.cast_column("unanswerable_audio", Audio(decode=False))
    if args.num_samples != -1:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    model, processor = load_model(args.model_path)
    judge_fn = make_judge(args.judge_model)

    total_c = total_r = 0.0
    n = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            ans_arr, ans_sr = decode_audio_field(row["answerable_audio"])
            una_arr, una_sr = decode_audio_field(row["unanswerable_audio"])

            for q_col in ["question_0", "question_1"]:
                question = row[q_col]
                ground_truth = row["ground_truth"]

                resp_ans = run_model(
                    model, processor, ans_arr, ans_sr, question, args.max_new_tokens
                )
                c, c_reason = eval_task_competence(
                    judge_fn, question, ground_truth, resp_ans
                )

                resp_una = run_model(
                    model, processor, una_arr, una_sr, question, args.max_new_tokens
                )
                r, r_reason = eval_repair_behavior(judge_fn, question, resp_una)

                total_c += c
                total_r += r
                n += 1

                rec = {
                    "id": row["id"],
                    "qcol": q_col,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answerable_response": resp_ans,
                    "unanswerable_response": resp_una,
                    "C": c,
                    "R": r,
                    "C_reason": c_reason,
                    "R_reason": r_reason,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"[{i+1}/{len(ds)}] id={row['id']} {q_col}  C={c}  R={r}")
                print(f"    Q : {question}")
                print(f"    GT: {ground_truth}")
                print(f"    [Answerable]   -> LLM: {resp_ans}")
                print(f"                   -> JUD: {c_reason}")
                print(f"    [Unanswerable] -> LLM: {resp_una}")
                print(f"                   -> JUD: {r_reason}\n")

        if n == 0:
            print("No instances evaluated.")
            return

        C = total_c / n
        R = total_r / n
        EAR = 0.0 if (C + R) == 0 else 2 * (C * R) / (C + R)

        fout.write(
            json.dumps(
                {
                    "type": "summary",
                    "instances": n,
                    "C": C,
                    "R": R,
                    "EAR": EAR,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fout.flush()

    print("\n======")
    print(f"Final Eval - Qwen2.5-Omni-3B ({n} instances)")
    print(f"C  : {C: .3f}")
    print(f"R  : {R: .3f}")
    print(f"EAR: {EAR: .3f}")
    print("======")
    print(f"Per-sample results + summary written to {args.out}")


if __name__ == "__main__":
    main()
