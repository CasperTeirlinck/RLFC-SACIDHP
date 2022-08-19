$batch = @(
    "trained/SAC_citation_tracking_attitude_1659223622/496000"
    "trained/SAC_citation_tracking_attitude_1659223622/496000"
    "trained/SAC_citation_tracking_attitude_1659223622/496000"
    "trained/SAC_citation_tracking_attitude_1659223622/496000"
    "trained/SAC_citation_tracking_attitude_1659223622/496000"
)

foreach ($inner_save_dir in $batch) {
    Start-Process -FilePath powershell.exe -ArgumentList "-NoExit", "-Command", ("[console]::WindowWidth = 150; ./venv/Scripts/python.exe .\scripts\train_sac_outer.py " + $inner_save_dir)
}