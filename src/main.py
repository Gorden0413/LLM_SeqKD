from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files="/home/andyee1997/ming/math_hard_inference_results/final_processed_text.json")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="test_1B_final",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="wandb",
)

print(training_args)
trainer = SFTTrainer(
    args=training_args,
    model="meta-llama/Llama-3.2-1B-Instruct",
    train_dataset=dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
)
trainer.train()
model = trainer.model.merge_and_unload()
model.save_pretrained("test_merge_1B_final")
