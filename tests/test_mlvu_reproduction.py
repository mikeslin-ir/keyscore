import json
from pathlib import Path

import pytest
import torch

from keyscore import KeyScore
from scripts.videoqa.mlvu_qa_eval import build_subset, run, select_topk, summarize


TASK = "2_needle"


@pytest.fixture
def annotations(tmp_path):
    folder = tmp_path / "json"
    folder.mkdir()
    questions = [
        {"video": f"clip{i}.mp4", "question": "Which object?",
         "candidates": ["A", "B", "C", "D"], "answer": "B"}
        for i in range(3)
    ]
    (folder / f"{TASK}.json").write_text(json.dumps(questions))
    return tmp_path


def test_manifest_selects_exact_questions_and_overrides_default_limit(annotations):
    ids = [f"{TASK}/clip2.mp4/2", f"{TASK}/clip0.mp4/0"]
    items = build_subset(annotations, [TASK], 1, ids)
    assert [item["uid"] for item in items] == ids
    assert len(build_subset(annotations, [TASK], 1)) == 1


@pytest.mark.parametrize("ids", [
    [], ["missing"], [f"{TASK}/clip0.mp4/0"] * 2, [123], {"not": "a list"},
])
def test_invalid_manifests_fail(annotations, ids):
    with pytest.raises(ValueError):
        build_subset(annotations, [TASK], 1, ids)


def test_missing_manifest_videos_fail_before_loading_models(annotations, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([f"{TASK}/clip0.mp4/0"]))
    with pytest.raises(FileNotFoundError, match="Videos missing"):
        run(annotations, tmp_path / "out", [TASK], 1, 32, 8,
            "unused-model", tmp_path / "cache", ["uniform"], manifest)


def test_qa_selection_uses_public_api_and_temporal_order():
    frames = torch.eye(4)
    query = torch.tensor([4.0, 3.0, 2.0, 1.0])
    selector = KeyScore(alpha=1.0, beta=0.0, gamma=0.0)
    expected = selector.select(frames, query, 2).indices.sort().values.tolist()
    assert select_topk(frames, query[None], selector, 2) == expected


def test_qa_selection_rejects_multiple_questions():
    with pytest.raises(ValueError, match="one question"):
        select_topk(torch.eye(4), torch.ones(2, 4), KeyScore(), 2)


def write_checkpoint(path: Path, outcomes):
    records = [
        {"uid": f"{TASK}/clip.mp4/{i}", "task": TASK,
         "uniform": {"correct": a}, "keyscore": {"correct": b}}
        for i, (a, b) in enumerate(outcomes)
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in records))
    return records


def test_summary_uses_paired_questions_and_exact_mcnemar(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    write_checkpoint(checkpoint, [(0, 1), (0, 1), (1, 0), (1, 1)])
    summarize(checkpoint, tmp_path, ["uniform", "keyscore"], [TASK])
    result = json.loads((tmp_path / "mlvu_summary.json").read_text())
    assert result["overall"]["uniform"]["correct"] == 2
    assert result["overall"]["keyscore"]["correct"] == 3
    assert result["metadata"]["num_questions"] == 4
    assert result["metadata"]["num_videos"] == 1
    assert result["paired_mcnemar"]["uniform_vs_keyscore"] == {
        "a_only_correct": 1, "b_only_correct": 2, "two_sided_exact_p": 1.0,
    }


@pytest.mark.parametrize("case", ["empty", "duplicate", "missing_method", "invalid_correct"])
def test_invalid_checkpoints_fail_explicitly(tmp_path, case):
    path = tmp_path / "checkpoint.jsonl"
    records = write_checkpoint(path, [(0, 1)])
    if case == "empty":
        path.write_text("")
    elif case == "duplicate":
        path.write_text("\n".join(json.dumps(records[0]) for _ in range(2)))
    elif case == "missing_method":
        del records[0]["keyscore"]
        path.write_text(json.dumps(records[0]))
    else:
        records[0]["keyscore"]["correct"] = 2
        path.write_text(json.dumps(records[0]))
    with pytest.raises(ValueError):
        summarize(path, tmp_path, ["uniform", "keyscore"], [TASK])
