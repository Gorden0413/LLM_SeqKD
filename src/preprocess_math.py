import json

def process_json(input_file, output_file):
    """
    Processes a JSON file to extract 'original_problem' and 'content' fields,
    and save them with keys renamed to 'prompt' and 'response'.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the processed JSON file.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        processed_entry = {
            "prompt": item["original_problem"],
            "response": item["generated_solution"]["content"]
        }
        processed_data.append(processed_entry)

    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)

input_file = "/home/andyee1997/ming/math_hard_inference_results/final_inference_results.json"  
output_file = "/home/andyee1997/ming/math_hard_inference_results/final_processed.json"  
process_json(input_file, output_file)
