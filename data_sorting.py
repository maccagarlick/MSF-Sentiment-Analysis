"""
This code is responsible for sorting the input data, which comes as a .xlsx file that contains all TAB responses.
"""

# Import necessary libraries
import pandas as pd
import os

# Function to load data from an Excel file
def load_data(working_directory):
    """
    Load data from an Excel file in the specified working directory.
    
    Args:
        working_directory (str): The working directory path containing the data folder.
        
    Returns:
        pd.DataFrame or None: The loaded input data as a DataFrame, or None if loading fails.
    """
     
     # Define input file path
    input_file = os.path.join(working_directory, 'data', 'input_data.xlsx')
    
    # Load the input data from the Excel file
    try:
        input_data = pd.read_excel(input_file)
        return input_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
# Function to create the main output file with additional columns for sentiment scores
def create_output_file(input_data, working_directory):
    """
    Create the main output file with additional columns for sentiment scores.
    
    Args:
        input_data (pd.DataFrame): The input data DataFrame to process.
        working_directory (str): The working directory path where the output file will be saved.
        
    Returns:
        str or None: The path to the created output file, or None if creation fails.
    """
    
    # Create a copy of the input data to modify
    output_file = input_data.copy() 
    
    # Create index of the first rating column
    MT_rating_index = output_file.columns.get_loc('MT Rating')
    
    # Add columns for the outputs of the sentiment score
    output_file.insert(current_index := MT_rating_index + 2, 'MT Sentiment', '')
    output_file.insert(current_index := current_index + 3, 'VC Sentiment', '')
    output_file.insert(current_index := current_index + 3, 'TW Sentiment', '')
    output_file.insert(current_index := current_index + 3, 'A Sentiment', '')
    
    # Define output file path
    output_file_path = os.path.join(working_directory, 'data', 'output.xlsx')
    
    # Save the modified data to a new Excel file
    try:
        output_file.to_excel(output_file_path, index=False)
        print(f"Data saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        return
    
    return output_file_path

# Function to sort data by ID and output individual files for each student
def output_individual_files(input_data, working_directory):
    """
    Sort data by student ID and create individual Excel files for each student.
    
    Args:
        input_data (pd.DataFrame): The input data DataFrame containing student data.
        working_directory (str): The working directory path where individual files will be saved.
        
    Returns:
        None: This function saves files to disk and doesn't return values.
    """
    
    # Loop to create list of unique IDs
    unique_ids = input_data['ID'].unique()
    
    # Create export directory if it does not exist
    individual_files_dir = os.path.join(working_directory, 'data', 'outputs')
    if not os.path.exists(individual_files_dir):
        os.makedirs(individual_files_dir)
    
    # Loop through each unique ID and create a separate file
    for student_id in unique_ids:
        # Filter the data for the current student ID
        student_data = input_data[input_data['ID'] == student_id]
        
        # Create index of the first rating column
        MT_rating_index = student_data.columns.get_loc('MT Rating')
        
        # Add columns for the outputs of the sentiment score
        student_data.insert(current_index := MT_rating_index + 2, 'MT Sentiment', '')
        student_data.insert(current_index := current_index + 3, 'VC Sentiment', '')
        student_data.insert(current_index := current_index + 3, 'TW Sentiment', '')
        student_data.insert(current_index := current_index + 3, 'A Sentiment', '')
        
        # Sort by year
        student_data = student_data.sort_values(by='Year')
        
        # Define output file path for the individual student
        output_file_path = os.path.join(individual_files_dir, f'student_{student_id}.xlsx')
        
        # Save the individual student's data to a new Excel file
        try:
            student_data.to_excel(output_file_path, index=False)
            print(f"Data for student {student_id} saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving data for student {student_id}: {e}")

# Keep only necessary columns in output.xlsx
def output_file_format(output_file_path, working_directory):
    """
    Create a simplified version of the output file with only necessary columns.
    
    Args:
        output_file_path (str): The path to the main output Excel file.
        working_directory (str): The working directory path where the simplified file will be saved.
        
    Returns:
        None: This function saves the simplified file to disk and doesn't return values.
    """
    
    # Load the file with only the selected columns
    try:
        selected_columns = ["Response Code", "ID", "MT Sentiment", "VC Sentiment", "TW Sentiment", "A Sentiment"]
        output_file_num_only = pd.read_excel(output_file_path, usecols=selected_columns)
    except Exception as e:
        print(f"Error loading the output_file.xlsx with selected columns: {e}")
        return
    
    # Save the new file
    try:
        output_file_num_only.to_excel(os.path.join(working_directory, "data", "output_file_num_only.xlsx"), index=False)
        print("output_file_num_only.xlsx saved")
    except Exception as e:
        print(f"Error saving output_file_num_only.xlsx file: {e}")
        return