"""
This script is designed to take the data and prompt from main_single.py and run a query through a local large language model using the Hugging Face Transformers library.

This version processes each response individually rather than in batches for improved accuracy.
"""

# Import necessary libraries
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch

# Define the working directory path
working_directory = os.getcwd()

# Function to run the query
def run_query(model, tokenizer, system_prompt, working_directory, output_file_path):
    """
    Processes sentiment analysis for all files in the outputs directory using individual response processing.
    
    This function:
    1. Iterates through all files in the data/outputs directory
    2. Loads each file and processes it with the file_import function
    3. Processes each response individually for improved accuracy
    4. Implements retry logic to handle incomplete or invalid responses
    5. Saves the processed sentiment scores back to the files and main output file
    
    Args:
        model (AutoModelForCausalLM): The loaded Hugging Face model for text generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        system_prompt (str): The system prompt containing instructions for sentiment analysis.
        working_directory (str): The working directory path containing the data folder.
        output_file_path (str): Path to the main output Excel file to be updated.
        
    Returns:
        None: This function processes files in-place and saves results to disk.
    """

    
    # Define file directory
    individual_files_directory = os.path.join(working_directory, "data", "outputs")
    
    for file in os.listdir(individual_files_directory):
        # Check if the file exists in the directory
        if not os.path.exists(os.path.join(individual_files_directory, file)):
            print(f"File {file} does not exist in the directory {individual_files_directory}.")
            return
        else:
            print(f"File {file} exists in the directory {individual_files_directory}.")
    
        # Import returned files from file_import
        current_data, current_data_abbriv = file_import(working_directory, file)
        
        print(f"Running query on model")
        
        attempts = 0
        cleaned_text = ""
        cleaned_output = []
        all_responses = {}
        
        # Initial 3 attempts with all data
        while attempts < 3:
            
            cleaned_output = []
            
            # Loop through each row in current_data_abbriv
            for i in range(current_data_abbriv.shape[0]):
                
                # Extract data from current row
                RC = current_data_abbriv.iloc[i]['Response Code']
                MT = current_data_abbriv.iloc[i]['MT Comments']
                VC = current_data_abbriv.iloc[i]['VC Comments']
                TW = current_data_abbriv.iloc[i]['TW Comments']
                A = current_data_abbriv.iloc[i]['A Comments']
                
                # Create directive prompt for the current row
                data_prompt = f"""STUDENT FEEDBACK FOR ANALYSIS:

Response Code: {RC}

FEEDBACK CONTENT TO ANALYZE:
MT Comments (Trust/Professional Behavior): "{MT}"
VC Comments (Communication Skills): "{VC}"
TW Comments (Teamwork/Collaboration): "{TW}"
A Comments (Accessibility/Availability): "{A}"

TASK: Carefully read each comment section above and analyze the sentiment based on the actual content and tone of what the student wrote."""
                
                prompt = f"{data_prompt}\n\n{system_prompt}\n\nFor Response Code {RC}, analyze the feedback content and output:"
                cleaned_text = generate(model, tokenizer, prompt, current_data)
                
                # Append cleaned_text to cleaned_output list
                if cleaned_text:
                    cleaned_text = cleaned_text.strip()
                    if cleaned_text and not cleaned_text.endswith('\n'):
                        cleaned_text += '\n'
                    cleaned_output.append(cleaned_text)

            print(f"Attempt {attempts + 1}: Generated {len(cleaned_output)} responses")
            
            # Convert list to single string for validation
            cleaned_text = ''.join(cleaned_output)
            
            # Extract and accumulate valid responses
            valid_responses = extract_valid_responses(cleaned_text, current_data)
            all_responses.update(valid_responses)
            
            # Check if we have all responses
            if len(all_responses) == len(current_data):
                print("All response IDs found.")
                cleaned_text = rebuild_response_text(all_responses)
                break
            else:
                missing_codes = get_missing_response_codes_from_dict(all_responses, current_data)
                print(f"Attempt {attempts + 1}: Missing response IDs: {missing_codes}")
                
            attempts += 1
        
        # If we still have missing responses after 3 attempts, retry with missing codes only
        if len(all_responses) < len(current_data):
            missing_codes = get_missing_response_codes_from_dict(all_responses, current_data)
            print(f"\nRetrying with {len(missing_codes)} missing response codes: {missing_codes}")
            
            # Create subset of data for missing codes
            missing_data = current_data_abbriv[current_data_abbriv['Response Code'].isin(missing_codes)]
            
            retry_attempts = 0
            while retry_attempts < 3 and len(all_responses) < len(current_data):
                retry_attempts += 1
                print(f"\nMissing codes retry attempt {retry_attempts}/3")
                
                cleaned_output = []
                
                # Loop through only the missing response codes
                for i in range(missing_data.shape[0]):
                    # Extract data from current row
                    RC = missing_data.iloc[i]['Response Code']
                    MT = missing_data.iloc[i]['MT Comments']
                    VC = missing_data.iloc[i]['VC Comments']
                    TW = missing_data.iloc[i]['TW Comments']
                    A = missing_data.iloc[i]['A Comments']
                    
                    # Create a more focused prompt for missing codes
                    data_prompt = f"Response Code: {RC}\nMT Comments: {MT}\nVC Comments: {VC}\nTW Comments: {TW}\nA Comments: {A}\n"
                    prompt = f"STUDENT FEEDBACK:\n{data_prompt}\n{system_prompt}\n\nFor Response Code {RC}, output EXACTLY: [{RC}, MT_score, VC_score, TW_score, A_score]"
                    
                    print(f"Retrying for response code {RC}")

                    cleaned_text = generate(model, tokenizer, prompt, current_data)
                    
                    # append cleaned_text to a new line in the cleaned_output list
                    if cleaned_text:
                        cleaned_text = cleaned_text.strip()
                        if cleaned_text and not cleaned_text.endswith('\n'):
                            cleaned_text += '\n'
                        cleaned_output.append(cleaned_text)
                
                # Process retry responses
                retry_text = ''.join(cleaned_output)
                valid_retry_responses = extract_valid_responses(retry_text, current_data)
                all_responses.update(valid_retry_responses)
                
                print(f"Retry attempt {retry_attempts}: Found {len(valid_retry_responses)} additional responses")
                print(f"Total responses now: {len(all_responses)}/{len(current_data)}")
                
                # Update missing codes for next iteration
                if len(all_responses) < len(current_data):
                    remaining_missing = get_missing_response_codes_from_dict(all_responses, current_data)
                    missing_data = current_data_abbriv[current_data_abbriv['Response Code'].isin(remaining_missing)]
                    print(f"Still missing: {remaining_missing}")
                else:
                    print("All response codes now found!")
                    break
        
        # Final check and save
        final_missing = get_missing_response_codes_from_dict(all_responses, current_data)
        if final_missing:
            print(f"\nFinal result: Still missing {len(final_missing)} response codes after all attempts: {final_missing}")
            print("Processing available responses...")
        else:
            print(f"\nSuccess! All {len(all_responses)} response codes found.")
        
        # Rebuild final response text from all accumulated responses
        cleaned_text = rebuild_response_text(all_responses)
        
        # save the response to a file (even if incomplete)
        save_response(cleaned_text, working_directory, current_data, file, output_file_path)

# Function to get missing response codes from accumulated responses dictionary
def get_missing_response_codes_from_dict(all_responses, current_data):
    """
    Get the list of missing response codes from the accumulated responses dictionary.
    
    Args:
        all_responses (dict): Dictionary of accumulated valid responses.
        current_data (pd.DataFrame): The current data DataFrame containing response codes.
    
    Returns:
        list: List of missing response codes sorted in ascending order.
    """
    expected_codes = set(current_data['Response Code'].astype(int))
    found_codes = set(int(code) for code in all_responses.keys())
    missing_codes = expected_codes - found_codes
    return sorted(list(missing_codes))

# Function to generate the response
def generate(model, tokenizer, prompt, current_data):
    """
    Generate a response from the model based on the input prompt and current data.
    
    This function:
    1. Tokenises the input text
    2. Generates a response from the model using conservative parameters for consistency
    3. Removes the input prompt from the output text
    4. Calls the clean_response_text function to clean the response text
    5. Returns the cleaned response text
    
    Args:
        model (AutoModelForCausalLM): The loaded Hugging Face model for text generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        prompt (str): The input prompt containing the data and system instructions.
        current_data (pd.DataFrame): The current data DataFrame containing response codes.
        
    Returns:
        str: The cleaned response text generated by the model.
    """
    # Tokenise the input text
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = model.device if hasattr(model, "device") else "cuda"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response with conservative parameters for consistency
    response = model.generate(
        **inputs,
        temperature=0.3,
        do_sample=True,
        top_p=0.8,
        max_new_tokens=100,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Get the response text and remove the input prompt
    input_len = inputs['input_ids'].shape[1]
    response_text = tokenizer.decode(response[0][input_len:], skip_special_tokens=True)
    
    # Clean response text to ensure it only contains the relevant part
    cleaned_text = clean_response_text(response_text, current_data)
    
    return cleaned_text

# Function to extract valid responses from cleaned text
def extract_valid_responses(cleaned_output, current_data):
    """
    Extract valid responses from cleaned text and return as a dictionary.
    
    Args:
        cleaned_output (list or str): The cleaned response text or list of responses.
        current_data (pd.DataFrame): The current data DataFrame containing response codes.
    
    Returns:
        dict: Dictionary with response codes as keys and full response strings as values.
    """
    valid_responses = {}
    valid_codes = set(map(str, current_data['Response Code'].astype(int)))
    
    # Convert list to string if needed
    if isinstance(cleaned_output, list):
        cleaned_text = ''.join(cleaned_output)
    else:
        cleaned_text = cleaned_output
    
    # Extract individual responses using regex
    response_pattern = r'\[\d+,\s*[1-7],\s*[1-7],\s*[1-7],\s*[1-7]\]'
    individual_responses = re.findall(response_pattern, cleaned_text)
    
    for response in individual_responses:
        # Extract response code from the response
        parts = response.strip('[]').split(',')
        parts = [part.strip() for part in parts]
        
        if len(parts) == 5 and parts[0] in valid_codes:
            response_code = parts[0]
            valid_responses[response_code] = response.strip()
    
    return valid_responses

# Function to rebuild response text from accumulated responses
def rebuild_response_text(accumulated_responses):
    """
    Rebuild response text from accumulated valid responses.
    
    Args:
        accumulated_responses (dict): Dictionary of valid responses.
    
    Returns:
        str: Rebuilt response text with all accumulated responses.
    """
    if not accumulated_responses:
        return ""
    
    # Sort by response code to maintain consistency
    sorted_responses = [accumulated_responses[code] for code in sorted(accumulated_responses.keys(), key=int)]
    return '\n'.join(sorted_responses)

# Function to import the file
def file_import(working_directory, file):
    """
    Function to import the file from the working directory.
    
    Args:
        working_directory (str): The working directory path.
        file (str): The name of the file to import.
        
    Returns:
        current_data (pd.DataFrame): The current data DataFrame.
        current_data_abbriv (pd.DataFrame): The abbreviated current data DataFrame with only the desired columns.
    """
    
    # Specify path to file
    file_path = os.path.join(working_directory, "data", "outputs", file)
    
    # Import file
    try:
        current_data = pd.read_excel(file_path)
        print(f"File loaded successfully from {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # leave only the desired columns
    current_data_abbriv = current_data[['Response Code', 'MT Comments', 'VC Comments', 'TW Comments', 'A Comments']]
        
    # return current_data and current_data_abbriv
    return current_data, current_data_abbriv


# Function to save the response to a file
def save_response(cleaned_text, working_directory, current_data, file, output_file_path):
    """
    Function to save the response to a file.
    
    Args:
        response_text (str): The response text to save.
        working_directory (str): The working directory path.
        current_data (pd.DataFrame): The current data DataFrame.
        
    Returns:
        str: The path to the saved file.
    """
    
    # Define output directory and make dir if not present
    output_dir = os.path.join(working_directory, "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # Handle different response formats by extracting individual responses
    response_pattern = r'\[\d+,\s*[1-7],\s*[1-7],\s*[1-7],\s*[1-7]\]'
    individual_responses = re.findall(response_pattern, cleaned_text)
    
    # If no individual responses found, fall back to line-based parsing
    if not individual_responses:
        response_lines = cleaned_text.strip().split('\n')
        individual_responses = [line.strip() for line in response_lines if line.strip()]
    
    # Load the output.xlsx file
    try:
        output_file = pd.read_excel(output_file_path)
    except Exception as e:
        print(f"Failed to load {output_file_path} due to: {e}")
        return
    
    # Loop through the individual responses and process each one
    for response in individual_responses:
        
        # Skip empty responses
        if not response.strip():
            continue
            
        # Split the response into parts
        parts = response.strip().strip('[]').split(',')
        parts = [part.strip() for part in parts]  # Remove any whitespace from each part
        
        # Validate that we have exactly 5 parts (response_code + 4 sentiment scores)
        if len(parts) != 5:
            print(f"Invalid response format: {response}")
            continue
            
        try:
            # Save parts to variables
            response_code = int(parts[0])
            mt_sentiment = float(parts[1])
            vc_sentiment = float(parts[2])
            tw_sentiment = float(parts[3])
            a_sentiment = float(parts[4])
        except ValueError as e:
            print(f"Error parsing response '{response}': {e}")
            continue
        
        # Get response code index from current_data.
        response_code_index = current_data[current_data['Response Code'] == int(response_code)].index
        
        # Convert response_code_index to a interger
        response_code_index = response_code_index[0] if not response_code_index.empty else None
        
        # Get respomse code index from output_file
        excel_response_code_index = output_file[output_file['Response Code'] == int(response_code)].index
        
        # Convert excelresponse_code_index to a interger
        excel_response_code_index = excel_response_code_index[0] if not excel_response_code_index.empty else None
        
        # Raplace empty cells in current_data with the sentiment scores
        if response_code_index is not None:
            current_data.at[response_code_index, 'MT Sentiment'] = mt_sentiment
            current_data.at[response_code_index, 'VC Sentiment'] = vc_sentiment
            current_data.at[response_code_index, 'TW Sentiment'] = tw_sentiment
            current_data.at[response_code_index, 'A Sentiment'] = a_sentiment
            
            output_file.at[excel_response_code_index, 'MT Sentiment'] = mt_sentiment
            output_file.at[excel_response_code_index, 'VC Sentiment'] = vc_sentiment
            output_file.at[excel_response_code_index, 'TW Sentiment'] = tw_sentiment
            output_file.at[excel_response_code_index, 'A Sentiment'] = a_sentiment
            
            
        else:
            print(f"Response code {response_code} not found in current_data.")
            continue
            
        print(f"Processed response code {response_code}")
    
    # Save the .xlsx file with the sentiment scores
    try:
        current_data.to_excel(os.path.join(output_dir, file), index=False)
        print(f"Sentiment scores saved to {os.path.join(output_dir, file)}")
    except Exception as e:
        print(f"Failed to output individual sentiment scores: {e}")
    
    # Save output.xlsx
    try:
        output_file.to_excel(output_file_path, index=False)
        print("output.xlsx updated")
    except Exception as e:
        print(f"Failed to update output: {e}")

# Function to clean the response text
def clean_response_text(response_text, current_data):
    """
    Extract sentiment scores in [code, mt, vc, tw, a] format from response text.
    
    Args:
        response_text (str): The raw response text.
        current_data (pd.DataFrame): DataFrame with valid response codes.
    
    Returns:
        str: The cleaned response text containing only valid formatted responses.
    """
    
    print(f"Raw response: {response_text[:200]}...")  # Show first 200 chars
    
    # Find all potential score patterns [number, 1-7, 1-7, 1-7, 1-7]
    response_pattern = r'\[(\d+),\s*([1-7]),\s*([1-7]),\s*([1-7]),\s*([1-7])\]'
    matches = re.findall(response_pattern, response_text)
    
    if not matches:
        print("No valid response patterns found")
        return ""
    
    # Get valid response codes from input data
    valid_codes = set(map(str, current_data['Response Code'].astype(int)))
    
    # Filter matches to only include valid response codes
    valid_responses = []
    for match in matches:
        response_code = match[0]
        if response_code in valid_codes:
            # Reconstruct the formatted response
            formatted_response = f"[{match[0]}, {match[1]}, {match[2]}, {match[3]}, {match[4]}]"
            valid_responses.append(formatted_response)
            print(f"Valid response found: {formatted_response}")
        else:
            print(f"Invalid response code {response_code} ignored")
    
    cleaned_text = '\n'.join(valid_responses)
    print(f"Cleaned output: {cleaned_text}")
    
    return cleaned_text

# function to ensure output is in the correct format
def is_valid_format(text):
    # Updated pattern to match the actual output format: [123,4,5,6,7] or [123, 4, 5, 6, 7]
    # Allows for multi-digit response codes and single digit scores (1-7) with optional spaces
    pattern = r'^\[\d+,\s*[1-7],\s*[1-7],\s*[1-7],\s*[1-7]\]$'
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Return False if no lines found (empty text)
    if not lines:
        return False
    return all(re.match(pattern, line) for line in lines)

# function to get missing response codes
def get_missing_response_codes(cleaned_text, current_data):
    """
    Get the list of missing response codes from the cleaned text.
    
    Args:
        cleaned_text (str): The cleaned response text.
        current_data (pd.DataFrame): The current data DataFrame containing response codes.
    
    Returns:
        list: List of missing response codes.
    """
    # Get all expected response codes from the input data
    expected_codes = set(current_data['Response Code'].astype(int))
    
    # Extract response codes from the cleaned text
    response_pattern = r'\[(\d+),\s*[1-7],\s*[1-7],\s*[1-7],\s*[1-7]\]'
    found_matches = re.findall(response_pattern, cleaned_text)
    found_codes = set(int(code) for code in found_matches)
    
    # Return missing codes
    missing_codes = expected_codes - found_codes
    return sorted(list(missing_codes))

# function to check if all response IDs from input data are present in the cleaned text
def has_all_response_ids(cleaned_text, current_data):
    """
    Check if all response IDs from the input data are present in the cleaned text.
    
    Args:
        cleaned_text (str): The cleaned response text.
        current_data (pd.DataFrame): The current data DataFrame containing response codes.
    
    Returns:
        bool: True if all response IDs are present, False otherwise.
    """
    # Get all expected response codes from the input data
    expected_codes = set(current_data['Response Code'].astype(int))
    
    # Extract response codes from the cleaned text
    response_pattern = r'\[(\d+),\s*[1-7],\s*[1-7],\s*[1-7],\s*[1-7]\]'
    found_matches = re.findall(response_pattern, cleaned_text)
    found_codes = set(int(code) for code in found_matches)
    
    # Check if all expected codes are found
    missing_codes = expected_codes - found_codes
    if missing_codes:
        print(f"Missing response codes: {sorted(missing_codes)}")
        return False
    
    print(f"All {len(expected_codes)} response codes found in output.")
    return True
        


if __name__ == "__main__":
    # This script is meant to be imported and used by other scripts
    # Individual function calls should be made from the importing script
    pass
