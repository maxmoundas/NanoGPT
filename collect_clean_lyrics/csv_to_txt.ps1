# Static paths
$csvFilePath = "CSV_FILE_PATH"
$outputTxtFilePath = "OUTPUT_FILE_PATH"

# Read the CSV
$csvContent = Import-Csv -LiteralPath $csvFilePath

# Convert CSV to desired format
$outputText = ""

foreach ($row in $csvContent) {
    $outputText += $row.Title + "`{`n" 
    $outputText += $row.Lyrics + "`n`}`n`n"
}

# Write the output to the .txt file
$outputText | Out-File -FilePath $outputTxtFilePath

Write-Output "File conversion completed!"
