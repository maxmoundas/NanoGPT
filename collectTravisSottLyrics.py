import lyricsgenius
import time
import random
import csv

def save_lyrics(song_title, lyrics, filename):
    with open(filename, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([song_title, lyrics])

def main():
    # Replace 'your_access_token_here' with your actual Genius API access token
    genius = lyricsgenius.Genius("pCTkuy3XgZEUa-hrsx6siLMffUT8u8fyfD03gdPl-Z5uQwE_7fbItVr0TQ8GR9z-", timeout=15, retries=3)

    # Search for and save all lyrics of Travis Scott
    artist = genius.search_artist("Travis Scott", max_songs=None, sort="title")
    if artist is None:
        print("Failed to retrieve artist.")
        return

    # Write headers to the CSV file
    with open('travis_scott_lyrics.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Lyrics'])

    for song in artist.songs:
        save_lyrics(song.title, song.lyrics, 'travis_scott_lyrics.csv')
        # Delay for a random interval between requests to avoid hitting rate limits
        time.sleep(random.uniform(1, 3))

if __name__ == "__main__":
    main()


# num chars in shakespeare.txt     = 1,075,394
# num chars in travis scott lyrics =   438,670
genius = lyricsgenius.Genius("pCTkuy3XgZEUa-hrsx6siLMffUT8u8fyfD03gdPl-Z5uQwE_7fbItVr0TQ8GR9z-", timeout=15, retries=3)
