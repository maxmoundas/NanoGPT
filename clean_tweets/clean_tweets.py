import pandas as pd
import re

def remove_links_from_text(filepath):
    # Load the CSV into a DataFrame
    df = pd.read_csv(filepath)
    
    # Define a function to remove links from a given text
    def remove_link(text):
        # Use regex to identify and remove links
        return re.sub(r'https\S+', '', text)
    
    # Define a function to keep only the specified characters in the text
    def filter_characters(text):
        # Use regex to keep only the specified characters
        return re.sub(r'[^a-zA-Z0-9!?@#$%&*()\-=+/\,.\'"\s]', '', text)

    # Apply the function to the 'text' column
    df['text'] = df['text'].apply(remove_link)

    # Apply the function to filter characters in the 'text' column
    df['text'] = df['text'].apply(filter_characters)
    
    # Filter out rows where the edited 'text' column starts with 'RT @'
    df = df[~df['text'].str.startswith('RT @')]
    
    # After removing 'RT @' rows, filter out rows where the 'text' column is empty or only contains whitespace
    df = df[df['text'].str.strip() != ""]
    
    # Define the output file path
    output_filepath = "clean_tweets/cleaned_tweets.csv"
    
    # Save the cleaned DataFrame to a new CSV
    df.to_csv(output_filepath, index=False)
    
    return output_filepath

# Example usage:
filepath = "clean_tweets/trump_tweets.csv"
output_filepath = remove_links_from_text(filepath)
print(f"Cleaned CSV saved to: {output_filepath}")
