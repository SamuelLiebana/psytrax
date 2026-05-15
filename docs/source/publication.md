# Publication

The most natural software venue is the
[Journal of Open Source Software (JOSS)](https://joss.readthedocs.io/en/latest/submitting.html),
but the project should look obviously useful and sustainable before submission.

## JOSS readiness checklist

| Requirement | psytrax status | Next action |
| --- | --- | --- |
| Open source repository | Repository exists | Confirm public visibility and issue access |
| OSI-approved license | Not visible in this checkout | Choose and add `LICENSE` |
| Installation instructions | README has pip and local app instructions | Add docs install page and test in clean env |
| Example usage | README and examples exist | Curate examples into docs gallery |
| API documentation | Docstrings exist unevenly | Generate Sphinx API reference and improve public docstrings |
| Tests | Pytest suite exists | Add CI and minimum/latest dependency jobs |
| Community guidelines | Initial guidance added | Add issue templates, code of conduct, support channel |
| Research impact | NeurIPS short paper and community outreach underway | Gather labs, projects, or analyses using psytrax |
| Statement of need | README has a concise package purpose | Write a JOSS-focused statement comparing alternatives |
| Archival DOI | Not visible in this checkout | Make a release and archive with Zenodo |

## Evidence to collect

- Labs or projects currently using psytrax.
- Example analyses that would be hard to reproduce without trial-varying model
  trajectories.
- Benchmarks showing when psytrax is fast enough for real datasets.
- Community feedback that led to concrete improvements.
- Comparison to related behavioural modelling and state-space packages.

## Suggested paper message

psytrax fills a practical gap between bespoke behavioural state-space analyses
and fully custom probabilistic modelling: it lets researchers write a
JAX-compatible per-trial likelihood while reusing a tested Empirical Bayes
inference pipeline for trial-varying decision parameters.
