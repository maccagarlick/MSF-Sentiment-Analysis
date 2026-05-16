"""
All functions are collated here and run via main function.

This version uses individual response processing for improved accuracy with smaller models.

This version also includes attention weight analysis functionality for research purposes.
"""

# Import necessary libraries first
import os
import subprocess
import sys
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Import functions
from data_sorting import load_data, create_output_file, output_individual_files, output_file_format
from model_script import run_query

# Function to generate the prompt
def prompt_generation(working_directory):
    """
    Function to generate the prompt and load the model for sentiment analysis.
    
    Args:
        working_directory (str): The working directory path.
        
    Returns:
        model (AutoModelForCausalLM): The loaded Hugging Face model for text generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        system_prompt (str): The system prompt containing instructions for sentiment analysis.
    """

    # Improved prompt for better attention to feedback content
    system_prompt = """Analyze student feedback comments and assign sentiment scores 1-7 for each domain.

SCORING SCALE (1=Very Negative, 7=Very Positive):
1-2: Negative feedback, concerns, problems
3-4: Neutral to slightly negative 
5-6: Positive feedback, good performance
7: Excellent, outstanding feedback

DOMAINS TO SCORE:
- MT: Trust/relationships, professional behaviour
- VC: Communication skills, clarity
- TW: Teamwork, collaboration 
- A: Accessibility, availability, responsibility

STRICT OUTPUT FORMAT: [Response_Code, MT_Score, VC_Score, TW_Score, A_Score]

EXAMPLES:
[123, 2, 3, 2, 2] - for mostly negative feedback
[456, 6, 6, 7, 5] - for mostly positive feedback

OUTPUT ONLY the bracketed scores, nothing else."""

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✓ Tokenizer loaded!")
    
    # set quantise configurations for meta-llama/Llama-3.3-70B-Instruct
    """ quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # Use normalized float 4-bit
        bnb_4bit_use_double_quant=True,  # Double quantization for better compression
    ) """

    # Load the model
    print("Loading model (this may take 1-3 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager"
    )
    print("✓ Model loaded successfully!")
    
    return model, tokenizer, system_prompt

if __name__ == "__main__":
    
    # Define the start time
    start_time = time.time()
    
    # Define working directory path
    working_directory = os.getcwd()
    
    # Load the data
    input_data = load_data(working_directory)
    
    if input_data is not None:
        
        # Add a unique code to each response based on its row position
        input_data.insert(0, 'Response Code', range(1, len(input_data) + 1))
        
        # Add unique codes and export the output file
        output_file_path = create_output_file(input_data, working_directory)
        
        # Sort data and output individual files for each student
        output_individual_files(input_data, working_directory)
        
    else:
        print("No data to process.")
    
    model, tokenizer, system_prompt = prompt_generation(working_directory)
    
    # Run the query and get the response
    run_query(model, tokenizer, system_prompt, working_directory, output_file_path)
    
    # Output numerical only file
    output_file_format(output_file_path, working_directory)
    
    # Define the end time
    end_time = time.time()
    
    # Print the time elapsed
    time_elapsed = end_time - start_time
    print(f"Code executed in {time_elapsed:.2f} seconds")
