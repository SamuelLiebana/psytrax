# psytrax Toolbox Master Prompt

Use this prompt from the `psytrax` repository when preparing toolbox-side work
for the NeurIPS response.

```text
You are working in the psytrax toolbox repository. This is separate from the
paper/response repository. Treat this repo as production-facing package and
Streamlit app code.

Use the repo instructions in AGENTS.md. Use GPT-5.6 for the main coordinator
when available. Use subagents for independent read-heavy planning, app testing,
and result auditing. Do not let multiple agents edit the same files at the same
time.

Goal:
Prepare psytrax for reviewer-response experiments and app/code changes while
preserving clean git history and human review gates.

Initial steps:
1. Inspect git status, repo layout, tests, app entry points, and experiment
   scaffolding.
2. Confirm the baseline with:
   conda run -n psytrax python -m pytest
3. Spawn psytrax_experiment_planner to map reviewer concerns to minimal toolbox
   experiments and approval packets.
4. If app behavior is touched, spawn psytrax_streamlit_tester to propose or add
   AppTest coverage.
5. Wait for subagent outputs and consolidate a plan.

Hard approval gates:
Stop before downloading datasets, accepting data terms, moving data into
long-term storage, launching full Slurm jobs, expanding sweeps, pushing branches,
merging to main, or making paper/rebuttal claims from new results.

Experiment workflow:
1. Produce a dataset/experiment approval packet before new data or fitting.
2. After explicit approval, implement only the approved scripts/configs.
3. Run local smoke tests before full Slurm jobs where feasible.
4. Submit Slurm jobs only after scoped approval.
5. Record provenance in experiments/manifests/.
6. Export small paper-ready outputs to experiments/results/.

Testing:
- Run conda run -n psytrax python -m pytest after source changes.
- Run focused tests first for narrow changes.
- Add Streamlit AppTest coverage for changed app behavior where practical.

Before returning:
Summarize changed files, test commands and outcomes, branch state, Slurm jobs
submitted or pending, result manifests, unresolved risks, and exact decisions
needed from the human.
```
