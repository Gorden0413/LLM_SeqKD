# EEP 596 Final Project: LLM Knowledge Distillation

This project is the final submission for the **EEP 596** course. Our team members are **Kuang-Ming Chen** and **Mike Huang**, and our project focuses on **LLM Knowledge Distillation**.

## Installation

To run this project, install the required Python packages using the following commands:

```bash
!pip install requirement.txt
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
    --model_args pretrained=./test_merge_1B,tokenizer="meta-llama/Llama-3.2-1B-Instruct" \ 
    --tasks mathqa \ 
    --device cuda:0 \
    --batch_size 8
```

## Experiment Results & Expected Output(The result of our method will show on terminal)

The table below summarizes the accuracy results on the MathQA dataset for different models and fine-tuning strategies:

| Model                            | Fine-tune Type              | Description           | Accuracy |
|----------------------------------|-----------------------------|-----------------------|----------|
| Llama-3.2-3B-Instruct (LoRA)     | Math_Hard         | Baseline4              | 0.3534   |
| Llama-3.2-3B-Instruct   | Math_Hard         | Baseline3              | 0.3451   |
| Llama-3.2-1B-Instruct (LoRA)     | Math_Hard_inf Our Method    | Proposed Method       | **0.3333**   |
| Llama-3.2-1B-Instruct (LoRA)     | Math_Hard        | Baseline2              | 0.3303   |
| Llama-3.2-1B-Instruct   | Math_Hard        | Baseline1              | 0.3283   |

## Pre-trained Model Link

Download here https://drive.google.com/file/d/13kvAtA6-oDNp4BtxErLDImahif94PSHx/view?usp=sharing


## Acknowledgments
 https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf
 https://huggingface.co/datasets/lighteval/MATH-Hard
 https://math-qa.github.io/
 https://arxiv.org/pdf/2106.09685
 https://github.com/EleutherAI/lm-evaluation-harness
