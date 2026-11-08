# Benchmarks

## AgentFault-500: trace grading and root-cause localization

[AgentFault-500](https://ieee-dataport.org/documents/agentfault) (Oct 2026, CC BY 4.0) has 160
clean and 500 fault-injected trajectories from a mock-LLM, four-agent LangGraph workflow, with
the injected step marked. It contains trajectories only: no task to rerun and no end state. So it
tests agent-breakpoint's **trace** grading and root-cause localization, not the state oracle,
which is what decides most grades in a real run.

```bash
agent-breakpoint bench agentfault-500      # downloads the two files, verifies their SHA-256
```

The data is fetched from the generator repo
([AgentFault-Infra](https://github.com/Skywalkr-dev/AgentFault-Infra)) at run time and cached in
`~/.cache/agent-breakpoint/`. It is not vendored: the repo has no license file, while DataPort
lists CC BY 4.0. The checksums are pinned to the release manifest (content hash `d5d2cca9…`).

**Method.** Groups are split by a hash of the baseline id: 80% dev, 20% held out, so all fault
variants of one baseline stay together. The trace rules were written looking only at dev
examples, then frozen and run once on the held-out groups. Intervals are 95% bootstrap,
resampling baseline groups (1,000 resamples).

| Held-out (116 trajectories, 89 injected, 27 groups) | Value | 95% CI |
| --- | --- | --- |
| Detection AUROC, claim oracle alone | 0.50 | 0.50–0.50 |
| Detection AUROC, unpaired trace rules | **0.82** | 0.76–0.88 |
| Root-cause step, unpaired trace rules (exact) | 0.66 | 0.56–0.75 |
| Root-cause step, paired first divergence (exact) | **0.92** | 0.88–0.97 |

All 660 trajectories: trace AUROC 0.80 (0.77–0.83), paired root cause 0.92 (0.90–0.94).
Top-3 accuracy equals exact here: the rules rarely flag more than one step per trajectory.

| Family (held-out) | n | Trace AUROC | Unpaired top-3 | Paired exact |
| --- | --- | --- | --- | --- |
| Control | 10 | 0.92 | 0.80 | 0.90 |
| Knowledge | 13 | 0.86 | 0.77 | 1.00 |
| Multi-agent | 22 | 0.84 | 0.68 | 1.00 |
| Plan | 16 | 0.77 | 0.75 | 0.75 |
| Security | 21 | 0.72 | 0.43 | 1.00 |
| Tool | 7 | 1.00 | 0.71 | 0.71 |

**What this says.**

* **The claim oracle alone is at chance.** Injected answers sound as confident as clean ones.
  That is the false-success problem this tool exists for, and why grades come from the end
  state, not from the agent's word.
* **Paired localization is the strong signal.** agent-breakpoint always has the baseline (fork-at-
  fault replay), so its report blames the first step where behaviour diverged, at 0.92 here.
  It misses most on Plan (the dropped step is labelled earlier than where the runs part) and
  Tool (the mutated call is mocked back to the same result).
* **Security is the weak unpaired family.** Cross-user leakage ("data belonging to another
  user") has no generic trace signal. In a live run the canary oracle catches it instead
  (`canary_leak`, or the whoseit pack).
* **Caveats.** The data is synthetic (mock LLM and tools), its outcome labels are injected rather
  than observed, and `PLAN_INVALID_DEPENDENCY` has no records. These numbers say the rules and
  the localizer work on clearly labelled faults; they don't measure real-world recall.

`tests/test_rootcause_bench.py` fails if the held-out numbers drop below the ones quoted here.
