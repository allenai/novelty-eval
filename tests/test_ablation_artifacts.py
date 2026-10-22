import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.analysis.artifacts import _extract_instance_id_from_key

def test_extract_instance_id_from_key():
    # Plain integers
    assert _extract_instance_id_from_key("42") == "42"
    assert _extract_instance_id_from_key("-5") == "-5"
    
    # Pointwise format: ptw-{id}-{index}
    assert _extract_instance_id_from_key("ptw-problem-1-0") == "problem-1"
    assert _extract_instance_id_from_key("ptw-complex-id-with-hyphens-0") == "complex-id-with-hyphens"
    assert _extract_instance_id_from_key("ptw-42-5") == "42"
    
    # Pairwise format: cmp-{id}-{pidx}-{i0idx}-{i1idx}
    assert _extract_instance_id_from_key("cmp-problem-1-0-1-2") == "problem-1"
    assert _extract_instance_id_from_key("cmp-42-10-0-1") == "42"
    assert _extract_instance_id_from_key("cmp-my-awesome-problem-0-0-1") == "my-awesome-problem"

    # Invalid formats
    assert _extract_instance_id_from_key("random-string") is None
    assert _extract_instance_id_from_key("ptw-no-index") is None
    assert _extract_instance_id_from_key("cmp-too-few-indices-1-2") is None
