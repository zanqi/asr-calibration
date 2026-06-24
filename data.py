from pprint import pprint
import random
from datasets import load_dataset
import jiwer
import numpy as np
import torch
from whisper.normalizers import EnglishTextNormalizer


HF_DATASET_REPO = "keylazy/slurp-noisy-asr-calibration"
AUDIO_SAMPLING_RATE = 16000

random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
normalizer = EnglishTextNormalizer()

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

        slurp_id = row['slurp_id']
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


print(f"Collected {len(dev_commands)} Dev examples, {len(test_commands)} Test examples.")

# return: ndarray[float32]
def synthesize_noisy_audio(clean_audio, babble_pool, snr_db):
    if snr_db == "clean":
        return clean_audio
    
    assert len(babble_pool) >= 3
    background_audios = random.sample(babble_pool, 3)
    mixed_babble = torch.zeros(len(clean_audio), dtype=torch.float32)

    for b in background_audios:
        if len(b) < len(clean_audio):
            babble = np.pad(b, (0, len(clean_audio) - len(b)), 'wrap')
        else:
            babble = b[:len(clean_audio)]
        
        mixed_babble += torch.tensor(babble, dtype=torch.float32)
    
    mixed_babble /= len(background_audios)
    clean_tensor = torch.tensor(clean_audio, dtype=torch.float32)
    clean_power = torch.mean(clean_tensor ** 2)
    babble_power = torch.mean(mixed_babble ** 2)
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
        clean_audio = c['audio']['array']
        rows.append({
            'sentence': row['sentence'],
            'audio_clean': synthesize_noisy_audio(clean_audio, babble_pool, "clean").tolist(),
            'audio_10dB': synthesize_noisy_audio(clean_audio, babble_pool, 10).tolist(),
            'audio_5dB': synthesize_noisy_audio(clean_audio, babble_pool, 5).tolist(),
            'audio_0dB': synthesize_noisy_audio(clean_audio, babble_pool, 0).tolist(),
        })
    return rows

dev_dataset = build_ds_rows(dev_commands)
test_dataset = build_ds_rows(test_commands)

# return: [(logit, is_correct)], e.g. [(32, 1), (6, 0)]
def align_and_label(words_info, ground_truth):
    #
    # 1. run Jiwer's alignment on hypothesis and refernece sentences
    # 2. assign label of the words within chunks based on chunk.type
    #  ('equal', 'insert', 'substitute') == 'equal'
    #
    ref = normalizer(ground_truth).strip()

    # normalize hypothesis word by word
    normalized_words_info = []
    for w in words_info:
        clean_word = normalizer(w['word']).strip()

        if clean_word:
            for sub_word in clean_word.split():
                normalized_words_info.append({
                    "word": sub_word,
                    "logits": w["logits"]
                })

    hypothesis = " ".join([w['word'] for w in normalized_words_info])

    if not hypothesis:
        print(f"  [!] empty normalized words: Total failure. Ground truth: '{ground_truth}'")
        return []

    # TODO: how does jiwer align two sentences?
    out = jiwer.process_words(ref, hypothesis)
    labeled_words = []

    for chunk in out.alignments[0]:
        # if the chunk is a match, all words in this chunk are correct
        if chunk.type == 'equal':
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                labeled_words.append((normalized_words_info[i]['logits'], 1))

        # if the chunk is a insert or substitute, those words are wrong
        elif chunk.type in ['insert', 'substitute']:
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                labeled_words.append((normalized_words_info[i]['logits'], 0))

        # ignore 'delete' chunk because the model didn't output a word to assign a confidence to
    return labeled_words

def extract_logits(model, processor, audio_array, ground_truth):
    tokenizer = processor.tokenizer
    if not isinstance(audio_array, torch.Tensor):
        audio_array = torch.tensor(audio_array).float()
    
    inputs = processor(
        audio=audio_array,
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)

    # TODO
    pprint(inputs)

    inputs = {k: v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            num_beams=1,
            language="en",
            task="transcribe",
        )
    
    num_generated_tokens = len(outputs.scores)
    generated_tokens = outputs.sequences[0, -num_generated_tokens:]

    words_info = []
    current_word_str = ""
    current_word_logits = []

    for token, logit_tensor in zip(generated_tokens, outputs.scores):
        if token.item() in tokenizer.all_special_ids:
            continue

        token_str = processor.decode(token, clean_up_tokenization_spaces=False)
        token_str = token_str.replace('Ġ', ' ')
        # TODO: logit_tensor size = ?
        vocab_logits = logit_tensor[0].cpu()

        if token_str.startswith(" ") and current_word_str:
            words_info.append({
                "word": current_word_str.strip(),
                "logits": current_word_logits
            })
            current_word_str = token_str
            current_word_logits = [vocab_logits]
        else:
            current_word_str += token_str
            current_word_logits.append(vocab_logits)
    
    if current_word_str.strip():
        words_info.append({
            "word": current_word_str.strip(),
            "logits": current_word_logits
        })
    
    if not words_info:
        print(f"  [!] empty model outputs: Total failure. Ground truth: '{ground_truth}'")
        return []
    
    return align_and_label(words_info, ground_truth)