# Define the input and output file paths
$inputFile = ""
$outputFile = ""

# Import the CSV file
$csvData = Import-Csv -Path $inputFile

# Create an array to hold the cleaned data
$cleanedData = @()

# Initialize a counter variable
$counter = 0

# Get the total number of rows
$totalRows = $csvData.Count

# Iterate through each row in the CSV
foreach ($row in $csvData) {
    # Increment the counter
    $counter++

    # Get the 'Lyrics' column value
    $lyrics = $row.Lyrics

    # Remove everything from the start of the cell until 'Lyrics' (inclusive)
    $lyrics = $lyrics -replace '.*?Lyrics', ''

    # Remove the phrase 'X Contributors' from the beginning of the cells
    # $lyrics = $lyrics -replace '^\d+ Contributors', ''

    # Remove text between 'See Travis Scott Live' and 'You might also like'
    $lyrics = $lyrics -replace '(?s)See Travis Scott Live.*?You might also like', ''

    # Remove the term 'Embed' if it exists
    $lyrics = $lyrics -replace 'Embed', ''

    # Remove characters that are not letters, numbers, or the specified special characters
    # Preserve newline characters by excluding them from the replace operation
    $lyrics = $lyrics -replace '[^a-zA-Z0-9 !$&,-.%:;?\r\n]+', ''

    # Add a newline character right before the term 'Lyrics' for the first time in a cell
    # $lyrics = $lyrics -replace 'Lyrics', "`nLyrics`n"

    # Create a new object with the cleaned lyrics and add it to the cleaned data array
    $cleanedRow = $row | Select-Object *
    $cleanedRow.Lyrics = $lyrics
    $cleanedData += $cleanedRow
    
    # Write progress to the console
    Write-Host "$counter of $totalRows completed"
}

# Export the cleaned data to a new CSV file
$cleanedData | Export-Csv -Path $outputFile -NoTypeInformation
