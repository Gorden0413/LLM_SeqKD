import json

def load_json(file_path):
    """
    Load JSON data from a file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: Parsed JSON data as a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, output_path):
    """
    Save data to a JSON file.
    
    Args:
        data (list): Data to save as a list of dictionaries.
        output_path (str): Path to save the JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def combine_prompt_response(data):
    """
    Combines 'prompt' and 'response' fields into a single text using the given template.

    Args:
        data (list): List of dictionaries containing 'prompt' and 'response'.

    Returns:
        list: A list of dictionaries with the combined 'text' field.
    """
    template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{response}<|eot_id|>"
    )
    
    combined_data = []
    for item in data:
        combined_text = template.format(prompt=item["prompt"], response=item["response"])
        combined_data.append({
            "text": combined_text,
            "prompt": item["prompt"],
            "response": item["response"]
        })
    
    return combined_data

def main(input_path, output_path):
    """
    Main function to load JSON, process data, and save the output.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file.
    """
    data = load_json(input_path)

    processed_data = combine_prompt_response(data)

    save_json(processed_data, output_path)
    print(f"Processed data has been saved to {output_path}")

if __name__ == "__main__":
    input_file = "/home/andyee1997/ming/math_hard_inference_results/final_processed.json"  
    output_file = "/home/andyee1997/ming/math_hard_inference_results/final_processed_text.json"  
    main(input_file, output_file)
