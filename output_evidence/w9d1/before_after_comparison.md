# W9D1 Before/After Comparison

## Baseline: Knowledge-Only Crew

The baseline CrewAI workflow performed the research without live web search.

### Baseline Result

The baseline crew successfully completed three sequential stages:

1. AI Researcher
2. Technical Writer
3. Research Reviewer

The researcher produced findings from the local LLM's existing knowledge. The writer converted those findings into a technical report, and the reviewer checked factual consistency, completeness, clarity, organization, and unsupported claims.

### Baseline Observation

The baseline workflow completed successfully, but the generated research was relatively general and included older examples such as StyleGAN and GAN/VAE-based approaches. The reviewer also identified unsupported or overly broad claims and recommended stronger evidence and source verification.

---

## Enhanced: Web-Enabled Crew

The enhanced workflow was configured to use the CrewAI Serper web-search tool for current web research.

### Web Search Execution

The Serper tool was successfully invoked, but the API request returned:

`403 Client Error: Forbidden`

with the response indicating:

`Unauthorized`

Therefore, the web-enabled run completed through the CrewAI workflow, but live web search results were not successfully retrieved during this execution.

### Web-Enabled Observation

Because the Serper request was unauthorized, the generated report should not be treated as genuinely web-grounded research. The writer explicitly indicated that it could not obtain specific research findings from the failed web search and produced a general report from its existing model knowledge.

---

## Before/After Workflow Comparison

| Aspect                      | Baseline                          | Web-Enabled Attempt                           |
| --------------------------- | --------------------------------- | --------------------------------------------- |
| CrewAI multi-agent workflow | Successful                        | Successful                                    |
| AI Researcher               | Successful                        | Successful                                    |
| Technical Writer            | Successful                        | Successful                                    |
| Research Reviewer           | Successful                        | Successful                                    |
| Live web search configured  | No                                | Yes                                           |
| Serper tool invoked         | No                                | Yes                                           |
| Live web results retrieved  | Not applicable                    | No — 403 Unauthorized                         |
| Source verification         | Model knowledge only              | Not completed because search failed           |
| Evidence generated          | Yes                               | Yes                                           |
| Main limitation             | Knowledge may be stale or general | Serper authorization prevented live retrieval |

## Observed Result

The W9D1 implementation demonstrates a CrewAI multi-agent research pipeline with separate researcher, writer, and reviewer roles.

The baseline workflow demonstrates knowledge-only research successfully.

The enhanced workflow demonstrates how a live-search tool can be integrated into the researcher agent. However, the Serper API authorization failure prevented actual web results from being retrieved in this run.

Therefore, this execution should be documented as a successful **web-search integration attempt**, not as a successful web-grounded research benchmark.

## Improvement Required

For a future run, a valid Serper API configuration should be provided and the workflow should be executed again. The web-enabled evidence can then be compared with the baseline using actual retrieved sources.
