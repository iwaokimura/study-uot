# study-uot
Study of Unbalanced Optimal Transport.

## Demonstrations

### Acronym Generation with Unbalanced Optimal Transport

The `acronym_demo.py` script demonstrates how Unbalanced Optimal Transport (UOT) can be used to create a soft mapping between a long phrase and its acronym.

**Why UOT is suited for this task:**
- The source (full phrase) has many more characters than the target (acronym)
- UOT allows for mass creation/destruction, handling the imbalance naturally
- The transport plan shows which characters contribute to the acronym

**Run the demo:**
```bash
pip install POT numpy
python acronym_demo.py
```

**Example output:**
```
>>> Processing: 'Unbalanced Optimal Transport' -> 'UOT'

'U' receives mass from:
  'U' (position 0): 0.0517  # Highest contribution!
  
'O' receives mass from:
  'O' (position 10): 0.0467  # Highest contribution!
  
'T' receives mass from:
  't' (position 25): 0.0470
  'T' (position 17): 0.0446  # Multiple T's contribute
```

The demo also shows how the `reg_m` (marginal relaxation) parameter affects the transport:
- Lower `reg_m` allows more mass imbalance (less total mass transported)
- Higher `reg_m` enforces stricter mass conservation
