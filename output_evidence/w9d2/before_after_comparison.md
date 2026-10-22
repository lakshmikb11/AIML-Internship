# W9D2 Baseline vs Tool-Assisted Comparison

## Topic
Practical applications of generative AI in software development

## Baseline
- Uses the local LLM's existing knowledge.
- No live web search is claimed.
- Evidence: `baseline_results.txt` if the run completed.

## Tool-Assisted Attempt
- Web search status: skipped: SERPER_API_KEY is not configured
- Evidence: `tool_assisted_results.txt` and, if created, `tool_assisted_error.txt`.
- Treat results as web-grounded only if the output shows that real search results were retrieved.

## Code Execution Tool Requirement
- The script performs a controlled Python syntax check using `compile()`.
- It does not execute arbitrary code produced by the LLM.
- Evidence: `code_validation_results.txt`.

## Honest Assessment
Do not claim the tool-assisted output is more accurate or current unless the run actually retrieved live sources and the report demonstrates a verifiable improvement.
