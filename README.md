# NPPE-2: Multilingual Speech Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Fine-tuned OpenAI Whisper for multilingual ASR across Indian languages using QLoRA**

*Achieved WER of 0.477 on held-out test set*

</div>

---

##  Overview

This project tackles multilingual automatic speech recognition (ASR) across diverse Indian languages including Tamil, Hindi, English, and others. The solution fine-tunes **OpenAI Whisper-medium** using **QLoRA (Quantized Low-Rank Adaptation)** to work within strict T4 GPU compute constraints on Kaggle.

The competition evaluates on **Word Error Rate (WER)** — lower is better.

```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

---

##  Architecture

```
Raw Audio (.wav)
      │
      ▼
┌─────────────────────────────┐
│   Audio Preprocessing       │
│   • Resample → 16kHz mono   │
│   • Trim silence (top_db=20)│
│   • Amplitude normalize     │
│   • Truncate to 30s         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Whisper Feature Extractor │
│   • Log-mel spectrogram     │
│   • Shape: (80, 3000)       │
│   • 80 mel bins × 3000 frames│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Whisper-medium Encoder    │
│   • 24 transformer layers   │
│   • 1024 hidden dim         │
│   + LoRA adapters on        │
│     q_proj, v_proj          │
│     (r=8, α=16)             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Whisper Decoder           │
│   • Cross-attention         │
│   • Autoregressive text gen │
│   • Beam search (k=5)       │
└─────────────┬───────────────┘
              │
              ▼
       Transcript Text
```

---

##  Dataset

| Split | Samples | Languages |
|-------|---------|-----------|
| Train | 2,000   | Tamil, Hindi, English, + others |
| Test  | 100     | Same distribution |

**Directory structure:**
```
competition_data/
├── train/
│   ├── audio_00000.wav
│   ├── audio_00001.wav
│   └── ...
└── test/
    ├── audio_00000.wav
    └── ...
train.csv       → audio, text columns
test.csv        → audio column
sample_submission.csv
```

---

## 🔧 Technical Approach

### 1. QLoRA Fine-tuning

Instead of full fine-tuning (too expensive on T4), we use **QLoRA** — training only ~0.25% of parameters:

```python
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

# Load base model in fp16
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium",
    torch_dtype = torch.float16,
    device_map  = "auto",
)
base_model.config.use_cache = False

# LoRA config — target attention projections only
lora_config = LoraConfig(
    r              = 8,
    lora_alpha     = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout   = 0.1,
    bias           = "none",
    task_type      = TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(base_model, lora_config)
model.enable_input_require_grads()

# Cast float32 params to fp16
for name, param in model.named_parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)

model.print_trainable_parameters()
# trainable params: 786,432 || all params: 316,337,902 || trainable%: 0.25
```

### 2. Audio Preprocessing

```python
import librosa
import numpy as np

def load_audio(filepath, target_sr=16000):
    """Load audio → 16kHz mono → trim silence → normalize amplitude."""
    y, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)
    peak  = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)
```

### 3. Dataset Preprocessing

Critical fix — **remove special tokens from labels** before training.
Whisper's tokenizer prepends `<|startoftranscript|><|lang|><|transcribe|>` tokens. 
If these appear in labels, the model tries to predict its own prompt tokens → loss explosion.

```python
def preprocess_sample(sample):
    y = load_audio(sample["audio"])
    y = y[:16000 * 30]  # hard truncate to 30s

    # Extract log-mel spectrogram
    input_features = feature_extractor(
        y, sampling_rate=16000, return_tensors="np"
    ).input_features[0]

    # Tokenize — add_special_tokens=False is critical
    labels = tokenizer(
        sample["text"],
        max_length        = 448,
        truncation        = True,
        add_special_tokens= False,  # ← prevents loss explosion
    ).input_ids

    return {"input_features": input_features, "labels": labels}
```

### 4. Data Collator

```python
@dataclasses.dataclass
class WhisperDataCollator:
    processor              : WhisperProcessor
    decoder_start_token_id : int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # Cast to fp16 to match model weights
        batch["input_features"] = batch["input_features"].to(torch.float16)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # Mask padding with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
```

### 5. Training Configuration

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir                  = "./whisper-qlora-ft",
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 8,       # effective batch = 16
    max_steps                   = 1000,
    warmup_steps                = 200,
    learning_rate               = 1e-4,
    fp16                        = False,   # disabled due to QLoRA gradient conflict
    bf16                        = False,
    gradient_checkpointing      = True,
    optim                       = "adamw_torch",
    max_grad_norm               = 1.0,     # gradient clipping — critical for stability
    eval_strategy               = "steps",
    eval_steps                  = 100,
    predict_with_generate       = True,    # real WER during eval, not proxy loss
    generation_max_length       = 448,
    save_strategy               = "steps",
    save_steps                  = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    report_to                   = "none",
    remove_unused_columns       = False,
)
```

### 6. WER Metric

```python
from jiwer import wer as compute_wer

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    predictions = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    references  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": round(compute_wer(references, predictions), 4)}
```

### 7. Inference

```python
model.eval()

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    y     = load_audio(TEST_AUDIO_DIR / row["audio"])
    y     = y[:16000 * 30]
    feats = processor(y, sampling_rate=16000, return_tensors="pt")
    feats = feats.input_features.to(torch.float16).to("cuda")

    with torch.no_grad():
        ids = model.generate(
            input_features = feats,
            num_beams      = 5,
            max_new_tokens = 200,
        )

    # Raw text output — no normalization (competition scores on raw text)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    all_predictions.append(text)
```

---

##  Results

| Experiment | WER | Notes |
|---|---|---|
| Zero-shot whisper-medium | ~0.85 | No fine-tuning |
| Fine-tuned + normalize_text | 0.578 | Lowercase + remove punctuation |
| Fine-tuned + raw text output | **0.477** | Best — competition scores raw text |
| Fine-tuned + beam=10 | 0.490 | Larger beam hurt slightly |

**Key finding:** Removing text normalization from inference dropped WER from 0.578 → 0.477. The competition evaluates on raw casing and punctuation — normalizing predictions was penalizing correct outputs.

---

##  Challenges & Solutions

| Problem | Cause | Fix |
|---|---|---|
| Loss explodes to 1e8 then collapses to 0 | Special tokens in labels | `add_special_tokens=False` in tokenizer |
| `ValueError: Attempting to unscale FP16 gradients` | QLoRA LoRA params stay fp16, scaler conflicts | `fp16=False`, `optim=adamw_torch` |
| `RuntimeError: Input type (float) and bias type (c10::Half)` | Conv layers fp16, input float32 | Don't set `torch_dtype` on base model load |
| WER improves then degrades after step 200 | Overfitting on 2000 samples | Reduced LoRA rank, increased dropout |
| `Seq2SeqTrainer tokenizer argument` | API change in newer transformers | Use `processing_class` instead |

---

##  Repository Structure

```
nppe2-multilingual-asr/
├── 21f3000371.ipynb      # Main fine-tuning notebook
└── README.md
```

---

##  Setup

```bash
pip install transformers==4.40.0
pip install accelerate==0.29.3
pip install peft==0.10.0
pip install datasets==2.19.0
pip install jiwer==3.0.3
pip install librosa soundfile
```

---

##  Key Learnings

1. **Special tokens in seq2seq labels cause catastrophic loss explosion** — always use `add_special_tokens=False` when tokenizing targets for Whisper fine-tuning
2. **QLoRA + fp16 gradient scaler conflict** — bitsandbytes 4-bit quantization and PyTorch fp16 AMP don't work together without careful dtype management
3. **Text normalization can hurt** — if the competition scores on raw text, normalizing predictions (lowercasing, removing punctuation) introduces errors on correctly capitalized outputs
4. **Beam search sweet spot** — beam=5 outperformed beam=10; larger beams don't always help and increase inference time
5. **2000 samples is very few for multilingual ASR** — LoRA rank 8 with dropout 0.1 and early stopping at patience=3 was essential to prevent overfitting

---

##  Author

**Kapishankar Ashtankar**  
3rd Year Data Science Student — IIT Madras  
[GitHub](https://github.com/kapshere)

---

<div align="center">
<i>Built as part of IIT Madras NPPE-2 Kaggle Competition</i>
</div>
