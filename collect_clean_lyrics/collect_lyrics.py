import lyricsgenius
import time
import random
import csv
from dotenv import load_dotenv
import os

# you need to create a .env file and store your Genius API key in a var named GENIUS_KEY
load_dotenv()  # Load the environment variables from the .env file
GENIUS_KEY = os.getenv('GENIUS_KEY')  # Retrieve the GENIUS_KEY environment variable

artist_name = "Travis Scott"
output_file_name = f"{artist_name.replace(' ', '')}_lyrics.csv"

def save_lyrics(song_title, lyrics, filename):
    with open(filename, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([song_title, lyrics])

def main():
    # Replace 'your_access_token_here' with your actual Genius API access token
    genius = lyricsgenius.Genius(GENIUS_KEY, timeout=15, retries=3)

    # Search for and save all lyrics of Travis Scott
    artist = genius.search_artist(artist_name, max_songs=None, sort="title")
    if artist is None:
        print("Failed to retrieve artist.")
        return

    # Write headers to the CSV file
    with open(output_file_name, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Lyrics'])

    for song in artist.songs:
        save_lyrics(song.title, song.lyrics, output_file_name)
        # Delay for a random interval between requests to avoid hitting rate limits
        time.sleep(random.uniform(1, 3))

if __name__ == "__main__":
    main()

genius = lyricsgenius.Genius(GENIUS_KEY, timeout=15, retries=3)
