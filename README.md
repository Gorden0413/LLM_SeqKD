# EEP 596 Final Project: LLM Knowledge Distillation

This project is the final submission for the **EEP 596** course. Our team members are **Kuang-Ming Chen** and **Mike Huang**, and our project focuses on **LLM Knowledge Distillation**.

## Installation

To run this project, install the required Python packages using the following commands:

```bash
!pip install bitsandbytes==0.43.0
!pip install datasets==2.10.1
!pip install transformers==4.38.2
!pip install peft==0.9.0
!pip install sentencepiece==0.1.99
!pip install -U accelerate==0.28.0
!pip install colorama==0.4.6
!pip install fsspec==2023.9.2
!pip install trl
```

## Workflow

### Data Preprocessing
Run the following scripts to preprocess the data:

```bash
!python preprocess_math.py
!python preprocess_text.py
```

### Teacher Model Inference
Perform inference with the teacher model:

```bash
!python main_inference.py
```

### LLM Supervised Fine-Tuning
Fine-tune the student model:

```bash
!python main.py
```

### Evaluation
Use the LM-Eval(https://github.com/EleutherAI/lm-evaluation-harness) repository for evaluation:

```bash
lm_eval --model hf \
    --model_args pretrained=model_path,tokenizer="meta-llama/Llama-3.2-1B-Instruct" \ 
    --tasks mathqa \ 
    --device cuda:0 \
    --batch_size 8
```

## Experiment Results

The table below summarizes the accuracy results on the MathQA dataset for different models and fine-tuning strategies:

| Model                            | Fine-tune Type              | Description           | Accuracy |
|----------------------------------|-----------------------------|-----------------------|----------|
| Llama-3.2-3B-Instruct (LoRA)     | Math_Hard         | Baseline4              | 0.3534   |
| Llama-3.2-3B-Instruct   | Math_Hard         | Baseline3              | 0.3451   |
| Llama-3.2-1B-Instruct (LoRA)     | Math_Hard_inf Our Method    | Proposed Method       | **0.3333**   |
| Llama-3.2-1B-Instruct (LoRA)     | Math_Hard        | Baseline2              | 0.3303   |
| Llama-3.2-1B-Instruct   | Math_Hard        | Baseline1              | 0.3283   |

