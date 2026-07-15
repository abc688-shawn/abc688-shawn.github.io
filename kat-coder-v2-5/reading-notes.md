# KAT-Coder-V2.5 working notes

## Scope and thesis

- Route: technical-report.
- Reader: engineers and researchers who understand LLM agents but have not read the report.
- Central question: does the report make a credible case that coding-agent capability is bottlenecked by training infrastructure more than model scale?
- Narrative spine: system decomposition with an evidence audit.
- Thesis: the report offers a coherent infrastructure-first recipe and a strong target-task capability profile, but does not close the causal chain because it lacks component-level ablations and key reproducibility details.

## Claim ledger

| Claim | Source | Evidence strength | Caveat |
| --- | --- | --- | --- |
| AutoBuilder yields 100K+ verifiable environments across 12 languages and raises construction success from 16.5% to 57.2% | §2.1, p.4; Fig.2, p.3 | Internal measured statistic | Dataset list, sample counts by language, variance, and external reproduction are absent |
| Hints raise pass rate on previously zero-pass tasks to about 20%, followed by hint-free replay | §2.2, p.5 | Internal pipeline statistic | No task count or baseline distribution is reported |
| KwaiClawEnv retains 100K+ tasks, averages 15 tool calls, and exceeds 100 steps at the long end | §3.3.2, p.8 | Internal scale statistic | Candidate-to-retained distribution and evaluation error rates are not reported |
| Token drift affects about 40% of approximately 200-turn samples | §4.2.1, p.11 | Internal observation | No sample size or severity distribution |
| Sandbox feedback error drops from about 16% to below 2% | §4.2.2, pp.11–12 | Before/after internal audit | No confidence interval; multiple fixes are bundled |
| KAT-Coder-V2.5 scores 65.2 SWE-Bench Pro, 53.1 KAT Code Bench, 94.9 PinchBench, 85.5 KAT Claw Bench, 60.7 Terminal-Bench 2.1, 50.3 SciCode | Table 4, p.19 | Direct benchmark table under a stated unified harness | Two benchmarks are internal; no repeated-run variance is shown |
| The full infrastructure causes the final benchmark profile | Abstract; §7 | System-level author claim | No component-level ablation establishes independent causal contributions |

## Method map

| Stage | Input | Mechanism | Training/evaluation signal | Output |
| --- | --- | --- | --- | --- |
| AutoBuilder | Repositories, issues, PRs, commits, code/test patches | Structured task regeneration, clarity filtering, sandboxed build/verify loop | Structured test parsing, >90% expected test collection, reproducible P2P/F2P | Verifiable SWE environments |
| Data flywheel | Failed, near-miss, and passing trajectories | Hint-boost, hint-free replay, process filtering, harness rewriting | Test outcomes plus exploration/localization/fidelity/minimality/verification/honesty | Robust SWE trajectories |
| KwaiClawEnv | Human skills, generated services, real-task seeds | Service/Task/Eval loop with complexity controls | Hard rules plus LLM judge | Multi-tool trajectories |
| Agentic RL | Multiple white/black-box harnesses | Gateway, reliable sandbox, asymmetric PPO, hindsight critic | Core tests, behavioral constraints, partial-progress incentives, GRM | Domain experts |
| MOPD | Five experts and student on-policy trajectories | Reverse-KL teacher supervision, cold start, drift-aware truncation | Token-level teacher/student compatibility | Unified student policy |

## Figure shortlist

- Fig.1 / PDF p.1: six-benchmark capability profile.
- Fig.2 / PDF p.3: AutoBuilder plus data scaling flywheel.
- Fig.3 / PDF p.6: KwaiClawEnv three-layer loop.
- Fig.4 / PDF p.10: agentic RL infrastructure.
- Fig.5 / PDF p.11: SWE reward curve.
- Table 4 / PDF p.19: exact benchmark scores.

## Limitation log

- Model parameter count, base-model recipe, pretraining data, post-training compute, and serving cost are not disclosed.
- No systematic ablation isolates AutoBuilder, KwaiClawEnv, harness scaling, sandbox fixes, asymmetric PPO, reward components, or MOPD.
- KAT Code Bench and KAT Claw Bench are internal; task inventories and full evaluation artifacts are not public in the report.
- Internal environment and reliability statistics omit sample sizes, confidence intervals, and detailed error distributions.
- Fig.5 shows a rising reward curve but does not compare algorithms or reward variants.
- Cross-harness generalization is motivated but not reported as a per-harness transfer matrix.
- Terminal and scientific coding remain weaker, suggesting specialization trade-offs that are not diagnosed further.
