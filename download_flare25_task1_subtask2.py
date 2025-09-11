import time

from huggingface_hub import snapshot_download

local_dir = "./FLARE-Task1-PancancerRECIST-to-3D"


while True:
    try:
        snapshot_download(
            repo_id="FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        time.sleep(60)  # Wait before the next download attempt due to rate limits.
    except Exception as e:
        print(e)
