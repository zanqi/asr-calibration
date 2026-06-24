from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import torch.optim as optim
import jiwer
from whisper.normalizers import EnglishTextNormalizer
from pprint import pprint

HF_MODEL_REPO = "your-username/whisper-temperature-calibrator"
AUDIO_SAMPLING_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

normalizer = EnglishTextNormalizer()

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

        for token_logits in word_logits_list:
            token_logits = token_logits.to(device)
            scaled_logits = token_logits / self.tau
            probs = torch.softmax(scaled_logits, dim=-1)
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            neg_ent = torch.sum(probs * log_probs, dim=-1)
            word_score = torch.sum(neg_ent)
            pred = self.alpha * word_score + self.beta
            preds.append(pred)

        return torch.stack(preds)
