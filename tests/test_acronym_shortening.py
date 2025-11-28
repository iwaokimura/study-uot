"""Tests for the acronym shortening example."""

import numpy as np
from examples.acronym_shortening import (
    character_position_cost,
    find_acronym_mapping,
)


def test_character_position_cost_matching():
    """Test that matching characters have zero cost."""
    source = ["A", "B", "C"]
    target = ["A", "B", "C"]
    cost = character_position_cost(source, target)

    # Diagonal should be 0 (matching characters)
    assert cost[0, 0] == 0
    assert cost[1, 1] == 0
    assert cost[2, 2] == 0

    # Off-diagonal should be 1 (non-matching characters)
    assert cost[0, 1] == 1
    assert cost[1, 0] == 1


def test_character_position_cost_case_insensitive():
    """Test that character matching is case-insensitive."""
    source = ["a", "B"]
    target = ["A", "b"]
    cost = character_position_cost(source, target)

    assert cost[0, 0] == 0  # 'a' matches 'A'
    assert cost[1, 1] == 0  # 'B' matches 'b'


def test_find_acronym_mapping_ai():
    """Test mapping for Artificial Intelligence -> AI."""
    result = find_acronym_mapping("Artificial Intelligence", "AI")

    assert result["phrase"] == "Artificial Intelligence"
    assert result["acronym"] == "AI"
    assert result["source_chars"] == ["A", "I"]
    assert result["target_chars"] == ["A", "I"]
    assert result["transport_plan"].shape == (2, 2)

    # Check that the diagonal has higher transport mass
    transport = result["transport_plan"]
    assert transport[0, 0] > transport[0, 1]
    assert transport[1, 1] > transport[1, 0]


def test_find_acronym_mapping_nlp():
    """Test mapping for Natural Language Processing -> NLP."""
    result = find_acronym_mapping("Natural Language Processing", "NLP")

    assert result["source_chars"] == ["N", "L", "P"]
    assert result["target_chars"] == ["N", "L", "P"]
    assert result["transport_plan"].shape == (3, 3)

    # Check mappings are found
    assert len(result["mappings"]) >= 3


def test_find_acronym_mapping_returns_valid_transport():
    """Test that the transport plan is valid (non-negative)."""
    result = find_acronym_mapping("Unbalanced Optimal Transport", "UOT")

    transport = result["transport_plan"]
    assert np.all(transport >= 0)


if __name__ == "__main__":
    test_character_position_cost_matching()
    test_character_position_cost_case_insensitive()
    test_find_acronym_mapping_ai()
    test_find_acronym_mapping_nlp()
    test_find_acronym_mapping_returns_valid_transport()
    print("All tests passed!")
