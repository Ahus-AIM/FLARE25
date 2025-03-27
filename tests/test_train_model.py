import os
import subprocess


def test_train_ahus_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    command = [
        "python",
        "-m",
        "src.train_ahus_model",
        "--num_clicks",
        "1",
        "--batch_size",
        "1",
        "--task_name",
        "test",
        "--click_type",
        "challenge",
        "--device",
        "cpu",
        "--base_dir",
        "tests/test_data",
        "--val_dir",
        "tests/test_data",
        "--size_threshold",
        "32768",  # 16^3
        "--num_workers",
        "0",
        "--num_epochs",
        "1",
    ]

    result = subprocess.run(command, cwd=base_dir, capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed with output: {result.stderr}"
    assert "error" not in result.stderr.lower(), f"Error in stderr: {result.stderr, result.stdout}"
