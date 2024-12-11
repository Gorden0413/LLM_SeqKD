import torch
from transformers import pipeline
from datasets import load_dataset
import json
import os
from tqdm import tqdm

model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

dataset = load_dataset("lighteval/MATH-Hard", trust_remote_code=True)
os.makedirs("math_hard_inference_results", exist_ok=True)

def process_problem(problem):
    messages = [
        {"role": "system", "content": "You are a helpful mathematical assistant. Solve the given problem."},
        {"role": "user", "content": problem['problem']}
    ]
    
    try:
        outputs = pipe(
            messages,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7
        )
        
        generated_response = outputs[0]["generated_text"][-1]
        
        return {
            "original_problem": problem['problem'],
            "generated_solution": generated_response,
            "ground_truth": problem['solution']
        }
    
    except Exception as e:
        return {
            "problem_id": problem['problem'],
            "error": str(e)
        }

results = []
for problem in tqdm(dataset['train'], desc="Processing problems", unit="problem"):
    result = process_problem(problem)
    results.append(result)

    if len(results) % 50 == 0:
        with open(f"math_hard_inference_results/inference_results_{len(results)}.json", "w") as f:
            json.dump(results, f, indent=2)

with open("math_hard_inference_results/final_inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Inference completed. Total problems processed: {len(results)}")