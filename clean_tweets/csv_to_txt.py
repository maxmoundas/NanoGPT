import pandas as pd

def write_text_to_file():
    # Load the CSV into a DataFrame
    df = pd.read_csv("clean_tweets/cleaned_tweets.csv")

    # Open the .txt file for writing
    with open("clean_tweets/trump_tweets.txt", "w") as file:
        # Initialize a counter to help us avoid adding a divider after the last row
        count = len(df['text'])
        for text in df['text']:
            file.write(text)
            count -= 1
            # Only add the divider if it's not the last row
            if count:
                file.write("\n----------------\n")

# Execute the function to write the texts to the .txt file
write_text_to_file()
