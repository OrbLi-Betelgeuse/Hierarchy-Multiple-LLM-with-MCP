# Experiment B: Hierarchical Collaboration (Multi-level vs. Single-level)

## 1. Experiment Design
- **Goal:** Compare single-level (flat) and multi-level (hierarchical) task decomposition and coordination.
- **Metrics:** Summarization (quality score ≈ ROUGE-L), QA (F1 Score), Table Generation (structure/content accuracy).

## 2. Results Summary Table

| Task Type        | Single-level (Manager-Executor) | Multi-level (Hierarchical) | Metric Type         |
|------------------|---------------------------------|----------------------------|---------------------|
| Summarization    | 0.1 (quality_score)             | 0.1 (quality_score)        | ROUGE-L (approx.)  |
| QA               | 0.0 (F1)                        | 0.0 (F1)                   | F1 Score           |
| Table Generation | 0.96 (content_accuracy)         | 0.0 (accuracy)             | Structure Accuracy  |

## 3. Detailed Analysis
- **Summarization**: Both pipelines output similar summaries with low quality scores (0.1), indicating model limitations rather than pipeline structure.
- **QA**: Both pipelines failed to generate correct answers (F1=0.0), again reflecting model capability bottlenecks.
- **Table Generation**: Single-level pipeline produced a correct table (accuracy 0.96), while multi-level pipeline failed to generate a valid table (accuracy 0.0).

## 4. Key Findings
- Hierarchical (multi-level) coordination did not improve output quality for these tasks with current model settings.
- The main bottleneck is the LLM's ability to follow instructions and generate structured outputs, not the pipeline structure.
- For table generation, decomposition may introduce more confusion for the model, leading to worse results.

## 5. Recommendations
- Consider using more powerful or instruction-tuned models for better results.
- For tasks requiring strict structure (like tables), flat pipelines may be more robust with current LLMs.
- Hierarchical coordination may show more benefit on complex, multi-step tasks or with better models.

---

**All results above are based on the latest experiment run with improved metric calculations and error handling.** 