"""
Local Fine-Tuning Script for Digital Twin Project
Optimized for RTX 4070 (8GB VRAM) + 16GB RAM
"""

import os
import sys
# Disable torch.compile and optimizations that hang on Windows
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from pathlib import Path

HF_TOKEN = "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"
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

PERSON_NAME = "Aditya_Ganti"
TRAINING_FILE = f"Training_Data/{PERSON_NAME}.jsonl"
OUTPUT_DIR = f"Adapters/{PERSON_NAME}"
MODEL_NAME = "./models/gemma-3-4b-it"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_RANK = 16
LORA_ALPHA = 16

print("=" * 60)
print("DIGITAL TWIN - LOCAL FINE-TUNING")
print("=" * 60)
print(f"Person: {PERSON_NAME}")
print(f"Model: {MODEL_NAME}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
else:
    print("⚠️  WARNING: CUDA NOT AVAILABLE - Training will be VERY slow!")
    print("   Training on CPU. This is likely a PyTorch installation issue.")
print("=" * 60)

# Check if training file exists
if not Path(TRAINING_FILE).exists():
    print(f"❌ Error: Training file not found: {TRAINING_FILE}")
    sys.exit(1)

print("\n[1/6] Loading dataset...")
dataset = load_dataset("json", data_files=TRAINING_FILE, split="train")
print(f"✓ Loaded {len(dataset)} training examples")

print("\n[2/6] Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"\n[3/6] Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"":0},
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"✓ Model loaded in 4-bit")
print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n[4/6] Configuring LoRA...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✓ LoRA configured")

print("\n[5/6] Configuring training parameters...")

if not torch.cuda.is_available():
    print("❌ ERROR: No CUDA GPU detected! Training will be extremely slow.")
    print("   Please reinstall PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,
)

print("\n[6/6] Starting training...")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Total steps: {(len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS}")
print("\n" + "=" * 60)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)

try:
    checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"))
    resume_from = str(checkpoints[-1]) if checkpoints else None
    
    if resume_from:
        print(f"📂 Resuming from checkpoint: {resume_from}")
    
    trainer.train(resume_from_checkpoint=resume_from)
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    
    # Save the adapter
    print(f"\nSaving LoRA adapter to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✓ Adapter saved!")
    print(f"\nAdapter size: ~150MB")
    print(f"Location: {Path(OUTPUT_DIR).absolute()}")
    
except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}_checkpoint")
    
except Exception as e:
    print(f"\n\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    del model
    del trainer
    torch.cuda.empty_cache()
    print("\n✓ Cleanup complete")
