"""
Acronym Shortening Example using Unbalanced Optimal Transport

This example demonstrates how to use Unbalanced Optimal Transport (UOT) to find
a mapping between long words/phrases and their acronyms.

The key insight is that UOT is well-suited for this task because:
1. The source (full word) and target (acronym) have different lengths
2. We want to find which characters in the source map to characters in the target
3. The "unbalanced" nature allows for mass creation/destruction (some source
   characters don't map to any target character)
"""

import numpy as np
import ot


def character_position_cost(source_chars, target_chars):
    """
    Compute a cost matrix based on character matching.

    The cost is 0 if characters match (case-insensitive), otherwise 1.

    Parameters
    ----------
    source_chars : list
        List of characters from the source word/phrase
    target_chars : list
        List of characters from the acronym

    Returns
    -------
    np.ndarray
        Cost matrix of shape (len(source_chars), len(target_chars))
    """
    n_source = len(source_chars)
    n_target = len(target_chars)
    cost = np.ones((n_source, n_target))

    for i, s_char in enumerate(source_chars):
        for j, t_char in enumerate(target_chars):
            if s_char.upper() == t_char.upper():
                cost[i, j] = 0

    return cost


def find_acronym_mapping(phrase, acronym, reg=0.1, reg_m=1.0):
    """
    Use Unbalanced Optimal Transport to find the mapping between a phrase
    and its acronym.

    Parameters
    ----------
    phrase : str
        The full phrase (e.g., "Artificial Intelligence")
    acronym : str
        The acronym (e.g., "AI")
    reg : float, optional
        Entropy regularization parameter (default: 0.1)
    reg_m : float, optional
        Marginal relaxation parameter for unbalanced OT (default: 1.0)

    Returns
    -------
    dict
        A dictionary containing:
        - 'transport_plan': The optimal transport matrix
        - 'source_chars': List of source characters
        - 'target_chars': List of target characters
        - 'mappings': List of (source_idx, source_char, target_idx, target_char, weight)
    """
    # Extract starting characters of words (typical acronym pattern)
    words = phrase.split()
    source_chars = [word[0] for word in words if word]
    target_chars = list(acronym)

    # Create uniform distributions
    n_source = len(source_chars)
    n_target = len(target_chars)

    # Source distribution (uniform over word-starting characters)
    a = np.ones(n_source) / n_source

    # Target distribution (uniform over acronym characters)
    b = np.ones(n_target) / n_target

    # Compute cost matrix
    cost = character_position_cost(source_chars, target_chars)

    # Solve unbalanced optimal transport
    transport_plan = ot.unbalanced.sinkhorn_unbalanced(a, b, cost, reg, reg_m)

    # Extract significant mappings
    mappings = []
    threshold = 0.01  # Minimum weight to consider a mapping significant
    for i in range(n_source):
        for j in range(n_target):
            if transport_plan[i, j] > threshold:
                mappings.append(
                    (i, source_chars[i], j, target_chars[j], transport_plan[i, j])
                )

    return {
        "transport_plan": transport_plan,
        "source_chars": source_chars,
        "target_chars": target_chars,
        "mappings": mappings,
        "phrase": phrase,
        "acronym": acronym,
    }


def print_mapping(result):
    """
    Print the mapping result in a human-readable format.

    Parameters
    ----------
    result : dict
        The result dictionary from find_acronym_mapping
    """
    print(f"\nPhrase: '{result['phrase']}'")
    print(f"Acronym: '{result['acronym']}'")
    print(f"Source characters (word starters): {result['source_chars']}")
    print(f"Target characters (acronym): {result['target_chars']}")
    print("\nTransport Plan (word starters -> acronym):")
    print(result["transport_plan"])
    print("\nSignificant Mappings:")
    for src_idx, src_char, tgt_idx, tgt_char, weight in result["mappings"]:
        print(f"  '{src_char}' (word {src_idx + 1}) -> '{tgt_char}' (pos {tgt_idx + 1}): {weight:.4f}")


def main():
    """
    Run the acronym shortening example with several test cases.
    """
    print("=" * 60)
    print("Acronym Shortening with Unbalanced Optimal Transport")
    print("=" * 60)

    # Test cases: (phrase, acronym)
    test_cases = [
        ("Artificial Intelligence", "AI"),
        ("Machine Learning", "ML"),
        ("Natural Language Processing", "NLP"),
        ("Application Programming Interface", "API"),
        ("Unbalanced Optimal Transport", "UOT"),
        ("Central Processing Unit", "CPU"),
        ("Graphics Processing Unit", "GPU"),
        ("Random Access Memory", "RAM"),
    ]

    for phrase, acronym in test_cases:
        result = find_acronym_mapping(phrase, acronym)
        print_mapping(result)
        print("-" * 60)


if __name__ == "__main__":
    main()
