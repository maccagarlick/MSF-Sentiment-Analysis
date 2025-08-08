# Importing Libraries
import pandas as pd
import re
import os
import random


def generate_random_id(TAB):
    
    """
    Generates random IDs for each row in the DataFrame.
    The IDs are unique and range from 100000 to 999999.
    The function updates the 'ID' column in the DataFrame with these new IDs.
    """
    
    print("Generating random IDs...")
    
    # Create a set of the origional IDs with a 2nd column for the new ID to ensure IDs are updated correctly for all records belonging to an individual
    ID_lookup = {
        "Origional_ID": (),
        "New_ID": ()
    }
    
    # Create a set to keep track of existing IDs to prevent duplication
    existing_ID = []
    existing_rand_ID = []
    
    # Generate random IDs for all rows
    new_ID = []
    for _ in range(len(TAB)):
        
        # Parse current ID into a variable
        current_ID = TAB['ID'].iloc[_]
        
        # Check if ID is in the existing IDs and if not then run the logic to generate a new ID
        if current_ID not in existing_ID:
            
            # Generate a random ID between 100000 and 999999
            rand_ID = random.randint(100000, 999999)
            
            # Ensure the random ID is unique
            while rand_ID in existing_rand_ID:
                rand_ID = random.randint(100000, 999999)
            
            # Append the new unique ID to the list
            
            # Add the new ID to the set to keep track of existing IDs
            existing_rand_ID.append(rand_ID)
            
            # Append the current_ID to existing_ID
            existing_ID.append(current_ID)
        
            # Append the new ID to the new_ID list
            ID_lookup["Origional_ID"] += (current_ID,)
            ID_lookup["New_ID"] += (rand_ID,)
        

    # Update the TAB DataFrame with the new random IDs
    TAB['ID'] = TAB['ID'].map(dict(zip(ID_lookup["Origional_ID"], ID_lookup["New_ID"])))


def name_remove(TAB):
    
    """
    Anonymizes names in specified comment columns and replaces with [NAME].
    Removes gender pronouns and gendered terms in the comments.
    The function modifies the DataFrame in place and deletes the "Names" column.
    Assumes the DataFrame has a "Names" column containing names to be anonymized.
    """
    
    print("Anonymizing names and removing gender pronouns...")

    # List of comment columns to anonymize
    comment_columns = [
        'MT Comments',
        'VC Comments',
        'TW Comments',
        'A Comments'
    ]

    # Loop through rows and anonymize names in comment columns
    for index, row in TAB.iterrows():
        # Check if "Names" is a string, otherwise skip
        if isinstance(row["Names"], str):
            # Create a list of names from the "Names" column
            names_list = re.split(r"\s*[;]\s*", row["Names"].strip())
            
            # Replace each name in all specified comment columns
            for name in names_list:
                for column in comment_columns:
                    # Replace names and gendered pronouns
                    TAB[column] = TAB[column].str.replace(fr'\b{name}\b', '[NAME]', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(he|she)\b', 'they', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(him|her)\b', 'them', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(his|her)\b', 'their', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(himself|herself)\b', 'themself', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r"\b(he's|she's)\b", "they're", case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r"\b(he'll|she'll)\b", "they'll", case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(man|woman)\b', 'person', case=False, regex=True)
                    TAB[column] = TAB[column].str.replace(r'\b(Mr|Mrs|Ms|Miss)\b', 'Mx', case=False, regex=True)

    # Delete the "Names" column
    TAB.drop(columns=["Names"], inplace=True)
    
def assessment_name(TAB):
    
    """
    Extracts the year of assessment from the Assesssment Name column.
    """
    
    print ("Extracting year from Assessment Name...")
    
    # Generate empty list to store years
    year_list = []
    
    # Iterate over rows
    for _ in range(len(TAB)):
        
        # Parse the assessment name
        assessment_name = TAB['Assessment Name'].iloc[_]
        
        # Extract the year from the assessment name
        year = re.search(r'\((.*?)\)', assessment_name)
        
        # append to the end of year_list
        if year:
            year_list.append(year.group(1))
        else:
            year_list.append(None)
    
    # Update the TAB DataFrame with the extracted years
    TAB.insert(2, "Year", year_list)
    
    # Drop assessment name column
    TAB.drop(columns=["Assessment Name"], inplace=True)
        
    
def main():
    
    # Setting working directory
    working_directory = os.getcwd()

    # Importing excel file
    print("Importing data...")
    TAB_path = os.path.join(working_directory, 'example_data.xlsx')
    TAB = pd.read_excel(TAB_path)
    
    # Run ID randomization function
    generate_random_id(TAB)
    
    # Run name removal function
    name_remove(TAB)
    
    # Run year extraction function
    assessment_name(TAB)
    
    # Export the updated DataFrame to a new Excel file
    output_path = os.path.join(working_directory, 'output.xlsx')
    TAB.to_excel(output_path, index=False)
    print(f"File saved to: \n {output_path}.")
    
if __name__ == "__main__":
    main()