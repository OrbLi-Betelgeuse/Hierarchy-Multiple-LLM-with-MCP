# Model Size Impact Analysis - Updated Results

## Key Findings

| Task | 1.5B Success | 1.5B Time | 1.5B Memory | 7B Success | 7B Time | 7B Memory | 14B Success | 14B Time | 14B Memory |
|------|-------------|-----------|-------------|------------|---------|-----------|-------------|----------|------------|
| qa | 33.3% | 6.2955s | 0.20 | 33.3% | 6.2955s | 0.20 | 33.3% | 6.2955s | 0.20 |
| table | 100.00% | 0.0179s | 0.35 | 100.00% | 0.0169s | 0.35 | 100.00% | 0.0168s | 0.35 |

## Analysis

### Performance Trends:

**QA:**
- Success Rate: Consistent across all model sizes (33.3%)
- Execution Time: 6.2955s (1.5B) → 6.2955s (7B) → 6.2955s (14B)
- Memory Usage: Consistent across all model sizes (0.20MB)

**TABLE:**
- Success Rate: Consistent across all model sizes (100.00%)
- Execution Time: 0.0179s (1.5B) → 0.0169s (7B) → 0.0168s (14B)
- Memory Usage: Consistent across all model sizes (0.35MB)

