# Model Size Impact Analysis - Summary

## Key Findings

| Task | 1.5B Success | 1.5B Time | 1.5B Memory | 7B Success | 7B Time | 7B Memory | 14B Success | 14B Time | 14B Memory |
|------|-------------|-----------|-------------|------------|---------|-----------|-------------|----------|------------|
| qa | 0.00% | 0.0092s | 0.20 | 0.00% | 0.0097s | 0.20 | 0.00% | 0.0096s | 0.20 |
| table | 100.00% | 0.0179s | 0.35 | 100.00% | 0.0169s | 0.35 | 100.00% | 0.0168s | 0.35 |

## Analysis

### Performance Trends:

**QA:**
- Success Rate: 0.00% (1.5B) → 0.00% (7B) → 0.00% (14B)
- Execution Time: 0.0092s (1.5B) → 0.0097s (7B) → 0.0096s (14B)
- Memory Usage: 0.20MB (1.5B) → 0.20MB (7B) → 0.20MB (14B)

**TABLE:**
- Success Rate: 100.00% (1.5B) → 100.00% (7B) → 100.00% (14B)
- Execution Time: 0.0179s (1.5B) → 0.0169s (7B) → 0.0168s (14B)
- Memory Usage: 0.35MB (1.5B) → 0.35MB (7B) → 0.35MB (14B)

