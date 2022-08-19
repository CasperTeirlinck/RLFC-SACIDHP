# Evaluate batch
$batch = @(
    "trained/SAC_citation_tracking_attitude_1659223622/best_eval_r"
    "trained/SAC_citation_tracking_attitude_1659223623/best_eval_r"
    "trained/SAC_citation_tracking_attitude_1659223621/best_eval_r"
    "trained/SAC_citation_tracking_attitude_1659223620/best_eval_r"
    "trained/SAC_citation_tracking_attitude_1659223619/best_eval_r"
)

foreach ($save_dir in $batch) {
    Start-Process -FilePath powershell.exe -ArgumentList "-NoExit", "-Command", ("[console]::WindowWidth = 150; ./venv/Scripts/python.exe .\scripts\train_sac_inner.py " + $save_dir)
}

