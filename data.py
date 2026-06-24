from pprint import pprint as pprint
import random
from datasets import Audio, Dataset, DatasetDict, load_dataset
import numpy as np
import torch

HF_DATASET_REPO = "keylazy/slurp-noisy-asr-calibration"
AUDIO_SAMPLING_RATE = 16000

random.seed(42)
print("Starting data.py")

#
# Dataset Generation
#
slurp_dataset = load_dataset("qmeeus/slurp", split="train", streaming=True)
dataset_iterator = iter(slurp_dataset)

dev_commands = []
test_commands = []
babble_pool = []
seen_slurp_ids = set()

total_target = 300
test_target = 200
while len(dev_commands) + len(test_commands) < total_target:
    try:
        row = next(dataset_iterator)
        audio_array = row["audio"]["array"]
        babble_pool.append(audio_array)

        slurp_id = row["slurp_id"]
        if slurp_id in seen_slurp_ids:
            continue

        sentence = row["sentence"]
        if len(sentence.split()) < 4:
            continue

        if len(test_commands) < test_target:
            test_commands.append(row)
        else:
            dev_commands.append(row)

        seen_slurp_ids.add(slurp_id)
    except StopIteration:
        break


print(
    f"Collected {len(dev_commands)} Dev examples, {len(test_commands)} Test examples."
)


# return: ndarray[float32]
def synthesize_noisy_audio(clean_audio, babble_pool, snr_db):
    if snr_db == "clean":
        return clean_audio

    assert len(babble_pool) >= 3
    background_audios = random.sample(babble_pool, 3)
    mixed_babble = torch.zeros(len(clean_audio), dtype=torch.float32)

    for b in background_audios:
        if len(b) < len(clean_audio):
            babble = np.pad(b, (0, len(clean_audio) - len(b)), "wrap")
        else:
            babble = b[: len(clean_audio)]

        mixed_babble += torch.tensor(babble, dtype=torch.float32)

    mixed_babble /= len(background_audios)
    clean_tensor = torch.tensor(clean_audio, dtype=torch.float32)
    clean_power = torch.mean(clean_tensor**2)
    babble_power = torch.mean(mixed_babble**2)
    assert babble_power > 0

    # snr_db = 10 * log10 (clean_power / target_babble_power)
    # ->
    target_babble_power = clean_power / (10 ** (snr_db / 10))
    scaling_factor = torch.sqrt(target_babble_power / babble_power)

    result_audio = clean_tensor + scaling_factor * mixed_babble
    return result_audio.numpy()


def build_ds_rows(commands):
    rows = []
    for c in commands:
        clean_audio = c["audio"]["array"]
        rows.append(
            {
                "sentence": c["sentence"],
                "audio_clean": {
                    "array": synthesize_noisy_audio(clean_audio, babble_pool, "clean"),
                    "sampling_rate": AUDIO_SAMPLING_RATE,
                },
                "audio_10dB": {
                    "array": synthesize_noisy_audio(clean_audio, babble_pool, 10),
                    "sampling_rate": AUDIO_SAMPLING_RATE,
                },
                "audio_5dB": {
                    "array": synthesize_noisy_audio(clean_audio, babble_pool, 5),
                    "sampling_rate": AUDIO_SAMPLING_RATE,
                },
                "audio_0dB": {
                    "array": synthesize_noisy_audio(clean_audio, babble_pool, 0),
                    "sampling_rate": AUDIO_SAMPLING_RATE,
                },
            }
        )
    return rows


dev_dataset = build_ds_rows(dev_commands)
test_dataset = build_ds_rows(test_commands)

#
# Uploading
#
dev_ds = Dataset.from_list(dev_dataset)
test_ds = Dataset.from_list(test_dataset)

audio_feature = Audio(sampling_rate=AUDIO_SAMPLING_RATE)
audio_cols = ["audio_clean", "audio_10dB", "audio_5dB", "audio_0dB"]
for col in audio_cols:
    dev_ds = dev_ds.cast_column(col, audio_feature)
    test_ds = test_ds.cast_column(col, audio_feature)

hf_dataset = DatasetDict({"dev": dev_ds, "test": test_ds})

print(f"Pushing Dataset to {HF_DATASET_REPO}...")
hf_dataset.push_to_hub(HF_DATASET_REPO)

print("\nUpload Complete!")
