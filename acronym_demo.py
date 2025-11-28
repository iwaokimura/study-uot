#!/usr/bin/env python3
"""
Demonstration: Making Long Words into Acronyms using Unbalanced Optimal Transport

This script demonstrates how to use POT's ot.unbalanced module to create
a soft mapping between a long phrase and its acronym. Unbalanced Optimal
Transport is particularly suited for this task because:

1. The source (full phrase) has many more characters than the target (acronym)
2. UOT allows for mass creation/destruction, handling the imbalance naturally
3. The transport plan shows which characters contribute to the acronym

Example: "Unbalanced Optimal Transport" -> "UOT"
"""

import numpy as np
import ot


def create_position_cost_matrix(source_len: int, target_len: int) -> np.ndarray:
    """
    Create a cost matrix based on relative positions.
    
    Characters at similar relative positions have lower transport cost.
    
    Parameters
    ----------
    source_len : int
        Length of the source string
    target_len : int
        Length of the target string (acronym)
    
    Returns
    -------
    np.ndarray
        Cost matrix of shape (source_len, target_len)
    """
    source_positions = np.arange(source_len) / max(source_len - 1, 1)
    target_positions = np.arange(target_len) / max(target_len - 1, 1)
    
    # Squared Euclidean distance between relative positions
    cost_matrix = (source_positions[:, np.newaxis] - target_positions[np.newaxis, :]) ** 2
    
    return cost_matrix


def create_character_cost_matrix(source: str, target: str) -> np.ndarray:
    """
    Create a cost matrix based on character matching.
    
    Matching characters have zero cost, non-matching have cost 1.
    
    Parameters
    ----------
    source : str
        Source string (the long phrase)
    target : str
        Target string (the acronym)
    
    Returns
    -------
    np.ndarray
        Cost matrix of shape (len(source), len(target))
    """
    source_chars = np.array(list(source.upper()))
    target_chars = np.array(list(target.upper()))
    
    # Use broadcasting to create boolean matrix of matching characters
    # Then convert to cost: 0 for match, 1 for non-match
    cost_matrix = (source_chars[:, np.newaxis] != target_chars[np.newaxis, :]).astype(float)
    
    return cost_matrix


def phrase_to_acronym_transport(
    phrase: str,
    acronym: str,
    reg: float = 0.1,
    reg_m: float = 1.0,
    alpha: float = 0.5
) -> tuple[np.ndarray, dict]:
    """
    Compute the unbalanced optimal transport plan between a phrase and its acronym.
    
    This function uses POT's sinkhorn_unbalanced to find a soft assignment
    between characters in the phrase and characters in the acronym.
    
    Parameters
    ----------
    phrase : str
        The full phrase (e.g., "Unbalanced Optimal Transport")
    acronym : str
        The acronym (e.g., "UOT")
    reg : float
        Entropic regularization parameter (default: 0.1)
    reg_m : float
        Marginal relaxation parameter for unbalanced OT (default: 1.0)
        Lower values allow more mass imbalance
    alpha : float
        Weight for combining position and character costs (default: 0.5)
        alpha=0 uses only character matching, alpha=1 uses only positions
    
    Returns
    -------
    tuple[np.ndarray, dict]
        - Transport plan matrix of shape (len(phrase), len(acronym))
        - Dictionary with additional information
    """
    # Remove spaces for character-level analysis
    phrase_chars = phrase.replace(" ", "")
    
    # Source distribution: uniform over phrase characters
    a = np.ones(len(phrase_chars)) / len(phrase_chars)
    
    # Target distribution: uniform over acronym characters
    b = np.ones(len(acronym)) / len(acronym)
    
    # Combined cost matrix
    pos_cost = create_position_cost_matrix(len(phrase_chars), len(acronym))
    char_cost = create_character_cost_matrix(phrase_chars, acronym)
    
    # Normalize costs to [0, 1] range
    if pos_cost.max() > 0:
        pos_cost = pos_cost / pos_cost.max()
    
    cost_matrix = alpha * pos_cost + (1 - alpha) * char_cost
    
    # Compute unbalanced optimal transport
    transport_plan = ot.unbalanced.sinkhorn_unbalanced(
        a, b, cost_matrix,
        reg=reg,
        reg_m=reg_m
    )
    
    info = {
        "phrase_chars": phrase_chars,
        "acronym": acronym,
        "source_distribution": a,
        "target_distribution": b,
        "cost_matrix": cost_matrix,
        "total_transported_mass": transport_plan.sum()
    }
    
    return transport_plan, info


def visualize_transport(transport_plan: np.ndarray, info: dict) -> str:
    """
    Create a text visualization of the transport plan.
    
    Parameters
    ----------
    transport_plan : np.ndarray
        The computed transport plan
    info : dict
        Information dictionary from phrase_to_acronym_transport
    
    Returns
    -------
    str
        Text representation of the transport plan
    """
    phrase_chars = info["phrase_chars"]
    acronym = info["acronym"]
    
    lines = []
    lines.append("=" * 60)
    lines.append("Unbalanced Optimal Transport: Phrase to Acronym")
    lines.append("=" * 60)
    lines.append(f"\nPhrase:  '{phrase_chars}'")
    lines.append(f"Acronym: '{acronym}'")
    lines.append(f"\nTotal transported mass: {info['total_transported_mass']:.4f}")
    
    lines.append("\n" + "-" * 60)
    lines.append("Transport Plan (top contributions per acronym character):")
    lines.append("-" * 60)
    
    for j, acr_char in enumerate(acronym):
        # Get contributions to this acronym character
        contributions = transport_plan[:, j]
        top_indices = np.argsort(contributions)[-5:][::-1]
        
        lines.append(f"\n'{acr_char}' receives mass from:")
        for idx in top_indices:
            if contributions[idx] > 1e-6:
                lines.append(
                    f"  '{phrase_chars[idx]}' (position {idx}): "
                    f"{contributions[idx]:.4f}"
                )
    
    lines.append("\n" + "-" * 60)
    lines.append("Per-character contribution summary:")
    lines.append("-" * 60)
    
    for i, char in enumerate(phrase_chars):
        total_sent = transport_plan[i, :].sum()
        if total_sent > 1e-6:
            destinations = []
            for j, acr_char in enumerate(acronym):
                if transport_plan[i, j] > 1e-6:
                    destinations.append(f"{acr_char}({transport_plan[i, j]:.3f})")
            if destinations:
                lines.append(f"  '{char}' (pos {i:2d}) -> {', '.join(destinations)}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def demo_acronym_examples():
    """Run demonstration with several example phrases."""
    
    examples = [
        ("Unbalanced Optimal Transport", "UOT"),
        ("Natural Language Processing", "NLP"),
        ("Artificial Intelligence", "AI"),
        ("Machine Learning", "ML"),
    ]
    
    print("\n" + "#" * 70)
    print("# DEMONSTRATION: Making Acronyms with Unbalanced Optimal Transport")
    print("#" * 70)
    
    for phrase, acronym in examples:
        print(f"\n\n>>> Processing: '{phrase}' -> '{acronym}'")
        
        transport_plan, info = phrase_to_acronym_transport(
            phrase, acronym,
            reg=0.1,
            reg_m=0.5,  # Allow significant mass imbalance
            alpha=0.3   # Favor character matching over position
        )
        
        visualization = visualize_transport(transport_plan, info)
        print(visualization)
    
    # Detailed example showing the effect of reg_m
    print("\n\n" + "#" * 70)
    print("# EFFECT OF MARGINAL RELAXATION (reg_m) PARAMETER")
    print("#" * 70)
    
    phrase = "Unbalanced Optimal Transport"
    acronym = "UOT"
    
    for reg_m in [0.1, 0.5, 1.0, 5.0]:
        transport_plan, info = phrase_to_acronym_transport(
            phrase, acronym,
            reg=0.1,
            reg_m=reg_m,
            alpha=0.3
        )
        print(f"\nreg_m = {reg_m}: Total mass transported = {transport_plan.sum():.4f}")
        print("  (Lower reg_m allows more mass imbalance, less mass is transported)")


if __name__ == "__main__":
    demo_acronym_examples()
