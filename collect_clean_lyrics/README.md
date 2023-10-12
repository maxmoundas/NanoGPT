This folder is a series of independent Python and PowerShell scripts that can be used to collect all of the lyrics of an artist (using the Genius API), clean the data, and convert it to a .txt file.

Store your Genius API key in a .env file in this directory (collect_clean_lyrics) in a variable named 'GENIUS_KEY'.

The songs are originally collected from the Genius API and put into a .csv file so that the user can easily remove entire redundant/fake/incomplete songs posted on Genius. 

Steps to clean a artist's discography of lyrics:
1. Store your Genius API key in a .env file in this directory (collect_clean_lyrics) in a variable named 'GENIUS_KEY'
2. To collect an artists lyrics, run: collect_clean_lyrics\collect_lyrics.py
3. Manually remove redundant/fake/incomplete songs from the .csv file
4. To automate the data cleaning process, run: collect_clean_lyrics\clean_lyrics.ps1 (you may need to tweak this script depending on your data)
5. To convert the .csv file to a .txt file the model can use, run: collect_clean_lyrics\csv_to_txt.ps1