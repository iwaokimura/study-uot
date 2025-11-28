# Study of Unbalanced Optimal Transport

This repository contains a study of the unbalanced optimal transport with POT (Python Optimal Transport library).

## Installation

Install the required dependencies:

```bash
pip install pot numpy
```

## Examples

### 1. Acronym Shortening

The first example demonstrates how to use Unbalanced Optimal Transport (UOT) to find mappings between long words/phrases and their acronyms.

**Why UOT is well-suited for this task:**
- The source (full phrase) and target (acronym) have different lengths
- We want to find which characters in the source map to characters in the target
- The "unbalanced" nature allows for mass creation/destruction (some source characters don't map to any target character)

**Run the example:**

```bash
python examples/acronym_shortening.py
```

**Example output:**

```
Phrase: 'Artificial Intelligence'
Acronym: 'AI'
Source characters (word starters): ['A', 'I']
Target characters (acronym): ['A', 'I']

Significant Mappings:
  'A' (word 1) -> 'A' (pos 1): 0.4837
  'I' (word 2) -> 'I' (pos 2): 0.4837
```

## References

- [POT: Python Optimal Transport](https://pythonot.github.io/)
- [Unbalanced Optimal Transport](https://pythonot.github.io/gen_modules/ot.unbalanced.html)
