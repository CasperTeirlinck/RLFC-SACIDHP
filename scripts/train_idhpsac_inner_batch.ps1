# Train batch
$batch_size = 10

for ($i = 1; $i -le $batch_size; $i++) {
    Start-Process -FilePath powershell.exe -ArgumentList "-NoExit", "-Command", "[console]::WindowWidth = 150; ./venv/Scripts/python.exe .\scripts\train_idhpsac_inner.py"
}
