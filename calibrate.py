import random

from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import jiwer
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from whisper.normalizers import EnglishTextNormalizer

HF_DATASET_REPO = "keylazy/slurp-noisy-asr-calibration"
HF_MODEL_REPO = "keylazy/whisper-temperature-calibrator"
AUDIO_SAMPLING_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

normalizer = EnglishTextNormalizer()
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


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
        normalized_word = normalizer(w["word"]).strip()

        if normalized_word:
            for sub_word in normalized_word.split():
                normalized_words_info.append({"word": sub_word, "logits": w["logits"]})

    hypothesis = " ".join([w["word"] for w in normalized_words_info])

    if not hypothesis:
        print(
            f"  [!] empty normalized words: Total failure. Ground truth: '{ground_truth}'"
        )
        return []

    # TODO: how does jiwer align two sentences?
    out = jiwer.process_words(ref, hypothesis)
    labeled_words = []

    for chunk in out.alignments[0]:
        # if the chunk is a match, all words in this chunk are correct
        if chunk.type == "equal":
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                labeled_words.append((normalized_words_info[i]["logits"], 1))

        # if the chunk is a insert or substitute, those words are wrong
        elif chunk.type in ["insert", "substitute"]:
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                labeled_words.append((normalized_words_info[i]["logits"], 0))

        # ignore 'delete' chunk because the model didn't output a word to assign a confidence to
    return labeled_words


# return: [ (token_logits_list, is_correct), (token_logits_list, is_correct), ... ]
# token_logits_list is the subword logits of a word: list[(vocab,)]
def extract_logits(model, processor, audio_array, ground_truth):
    tokenizer = processor.tokenizer
    if not isinstance(audio_array, torch.Tensor):
        audio_array = torch.tensor(audio_array).float()

    inputs = processor(
        audio=audio_array,
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    inputs = {
        k: (
            v.to(model.device, dtype=model.dtype)
            if v.is_floating_point()
            else v.to(model.device)
        )
        for k, v in inputs.items()
    }

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
        token_str = token_str.replace("Ġ", " ")
        # TODO: logit_tensor size = ?
        vocab_logits = logit_tensor[0].cpu()

        if token_str.startswith(" ") and current_word_str:
            words_info.append(
                {"word": current_word_str.strip(), "logits": current_word_logits}
            )
            current_word_str = token_str
            current_word_logits = [vocab_logits]
        else:
            current_word_str += token_str
            current_word_logits.append(vocab_logits)

    if current_word_str.strip():
        words_info.append(
            {"word": current_word_str.strip(), "logits": current_word_logits}
        )

    if not words_info:
        print(
            f"  [!] empty model outputs: Total failure. Ground truth: '{ground_truth}'"
        )
        return []

    # words_info: list[(word: str, logits: list[list[float]])]
    return align_and_label(words_info, ground_truth)


class TemperatureCalibrator(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    # word_logits_list: list of subwords's logits [(1, V), (1, V), ...]
    def forward(self, word_logits_list):
        device = self.tau.device
        preds = []

        safe_tau = 1.0 + F.softplus(self.tau)
        for token_logits in word_logits_list:
            token_logits = token_logits.to(device)
            token_logits = torch.nan_to_num(
                token_logits, nan=0.0, neginf=-1e5, posinf=1e5
            )

            scaled_logits = token_logits / safe_tau
            # It can be proved:
            # softmax(log(p_k) / tau) == softmax(z_k / tau)
            # by plug in p_k = exp(z_i) / sum(exp(z_j))
            probs = torch.softmax(scaled_logits, dim=-1)
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            neg_ent = torch.sum(probs * log_probs, dim=-1)
            word_score = torch.sum(neg_ent)  # TODO: experiment with aggregation
            pred = self.alpha * word_score + self.beta
            preds.append(pred)

        return torch.stack(preds)


if __name__ == "__main__":
    asr_id = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(asr_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_id, dtype=torch_dtype, attn_implementation="flash_attention_2"
    ).to(device)

    print("\n--- 1: Extracting Logits on Dev set ---")
    snr_levels = ["clean", 10, 5, 0]
    snr2col = {"clean": "audio_clean", 10: "audio_10dB", 5: "audio_5dB", 0: "audio_0dB"}

    # list[ (token_logits_list, float), ]
    dev_features = []

    dev_dataset = load_dataset(HF_DATASET_REPO, split="dev")
    for item in tqdm(dev_dataset.select(range(1000))):
        ground_truth = item["sentence"]

        for snr in snr_levels:
            audio_array = np.array(item[snr2col[snr]]["array"], dtype=np.float32)
            labeled_words = extract_logits(model, processor, audio_array, ground_truth)

            for token_logits_list, label in labeled_words:
                if len(token_logits_list) > 0:
                    dev_features.append((torch.stack(token_logits_list), float(label)))

    calibrator = TemperatureCalibrator().to(device)
    optimizer = optim.Adam(calibrator.parameters(), lr=0.01)
    # BCEWithLogitsLoss vs BCELoss
    # BCELoss takes probability, BCEWithLogitsLoss takes logits
    criterion = nn.BCEWithLogitsLoss()

    epochs = 50
    batch_size = 512

    for epoch in range(epochs):
        random.shuffle(dev_features)
        epoch_loss = 0.0

        for i in range(0, len(dev_features), batch_size):
            batch = dev_features[i : i + batch_size]
            batch_logits = [item[0] for item in batch]
            batch_labels = torch.tensor(
                [item[1] for item in batch], dtype=torch.float32
            ).to(device)

            optimizer.zero_grad()
            preds = calibrator(batch_logits)
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dev_features):.4f} | "
                f"Tau: {calibrator.tau.item():.4f}, Alpha: {calibrator.alpha.item():.4f}, Beta: {calibrator.beta.item():.4f}"
            )

    calibrator.push_to_hub(HF_MODEL_REPO, private=False)
    print("Done!")
