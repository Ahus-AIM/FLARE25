import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from src.scripts.evaluate_recist import evaluate
from src.utils.recist_metrics import score_case
from src.utils.surface_dice import compute_dice_coefficient, compute_multi_class_dsc


def test_integer_labels_are_not_binary_weights():
    gt = np.array([1, 2, 3]).reshape(1, 1, 3)
    pred = np.array([1, 3, 2]).reshape(1, 1, 3)
    with pytest.raises(TypeError, match="boolean masks"):
        compute_dice_coefficient(gt, pred)
    result = score_case(gt, pred, [1, 1, 1])
    assert result["means"]["dsc"] == pytest.approx(1 / 3)
    # No background voxels: the smallest positive lesion must still be included.
    assert compute_multi_class_dsc(gt, pred) == pytest.approx(1 / 3)
    renaming = np.array([0, 11, 5, 27])
    renamed = score_case(renaming[gt], renaming[pred], [1, 1, 1])
    assert renamed["means"] == result["means"]


def test_spacing_affects_nsd_but_not_dsc():
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[2:4, 2:6, 2:6] = 1
    pred[3:5, 2:6, 2:6] = 1
    unit = score_case(gt, pred, [1, 1, 1])["means"]
    anisotropic = score_case(gt, pred, [1, 1, 5])["means"]
    assert unit["dsc"] == anisotropic["dsc"] == 0.5
    assert anisotropic["nsd"] < unit["nsd"]


def test_nsd_gate_is_applied_per_lesion_and_includes_boundary():
    gt = np.zeros((3, 3, 40), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[1, 1, :10] = 1
    pred[1, 1, 0] = 1
    gt[1, 1, 20:30] = 2
    pred[1, 1, 20:30] = 2
    result = score_case(gt, pred, [1, 1, 1])
    assert result["means"]["dsc"] > 0.2
    assert result["lesions"][0]["dsc"] < 0.2
    assert result["lesions"][0]["nsd"] == 0
    assert result["means"]["nsd"] == 0.5
    gt[1, 1, 9] = 0
    boundary = score_case(gt, pred, [1, 1, 1])["lesions"][0]
    assert boundary["dsc"] == 0.2
    assert boundary["nsd"] > 0


def test_missing_lesions_and_invalid_inputs():
    gt = np.ones((2, 2, 2), dtype=np.uint8)
    assert score_case(gt, np.zeros_like(gt), [1, 1, 1])["means"] == {"dsc": 0, "nsd": 0}
    with pytest.raises(ValueError, match="Empty reference"):
        score_case(np.zeros_like(gt), gt, [1, 1, 1])
    with pytest.raises(ValueError, match="matching"):
        score_case(gt, gt[:1], [1, 1, 1])
    with pytest.raises(ValueError, match="spacing"):
        score_case(gt, gt, [0, 1, 1])
    with pytest.raises(ValueError, match="integer"):
        score_case(gt, gt * 0.5, [1, 1, 1])


def test_cli_averages_cases_equally(tmp_path):
    inputs, predictions = tmp_path / "inputs", tmp_path / "predictions"
    inputs.mkdir()
    predictions.mkdir()
    np.savez(inputs / "one.npz", gts=np.ones((1, 1, 3)), spacing=[1, 1, 1])
    np.savez(predictions / "one.npz", segs=np.ones((1, 1, 3)))
    np.savez(inputs / "three.npz", gts=np.array([1, 2, 3]).reshape(1, 1, 3), spacing=[1, 1, 1])
    np.savez(predictions / "three.npz", segs=np.array([1, 3, 2]).reshape(1, 1, 3))
    output = tmp_path / "results.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.evaluate_recist",
            "--inputs",
            str(inputs),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["n_cases"] == 2 and result["n_lesions"] == 4
    assert result["summary"]["dsc"]["mean"] == pytest.approx(2 / 3)
    assert result["summary"]["dsc"]["sample_sd"] == pytest.approx(np.std([1, 1 / 3], ddof=1))
    (predictions / "three.npz").unlink()
    with pytest.raises(FileNotFoundError, match="three"):
        evaluate(inputs, predictions)


def test_nifti_scoring_uses_original_npz_spacing(tmp_path):
    inputs, predictions = tmp_path / "inputs", tmp_path / "predictions"
    inputs.mkdir()
    predictions.mkdir()
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[2:4, 2:6, 2:6] = 1
    pred[3:5, 2:6, 2:6] = 1
    np.savez(inputs / "sample.npz", gts=gt, spacing=[1, 1, 5])
    nib.save(nib.Nifti1Image(pred, np.eye(4)), predictions / "sample.nii.gz")
    result = evaluate(inputs, predictions)["cases"][0]
    assert result["means"] == score_case(gt, pred, [1, 1, 5])["means"]
    assert result["means"]["nsd"] != score_case(gt, pred, [1, 1, 1])["means"]["nsd"]
