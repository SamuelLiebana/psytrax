# Maintenance

psytrax is currently pre-1.0 research software. The maintenance goal is to be
honest about what is supported, while keeping the package usable for students
and experimental labs.

## Support policy

- Supported Python versions and dependency lower bounds should be declared in
  `pyproject.toml`.
- The test suite should run against the declared minimum supported environment
  and a recent environment.
- Lower bounds should be revisited on a regular schedule rather than changed
  reactively.
- Upper bounds should be avoided unless an upstream release is known to break
  psytrax.
- GPU support should be documented separately from the core CPU install because
  JAX backend installation differs by platform.

## SPEC 0 alignment

[Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/)
recommends a time-based policy for dropping old Python and core dependency
versions: Python versions after 3 years, and core package dependency versions
after 2 years. psytrax should use that as the default policy for NumPy, SciPy,
pandas, and matplotlib unless accessibility for a specific user group requires
a temporary exception.

JAX is a core dependency for psytrax but not part of the SPEC 0 core package
table. For JAX, psytrax should follow the actively maintained JAX release line,
test the minimum declared version in CI, and keep installation notes current
for CPU, CUDA, and Apple Silicon users.

## Immediate decisions

- Decide whether the next release should keep Python `>=3.10` for accessibility
  or move toward the current SPEC 0 window.
- Add CI for at least one minimum-dependency environment and one latest
  environment.
- Add a release checklist with tests, docs build, version tag, and archive DOI.
- Add a public policy for how long old saved fit files and result dictionaries
  remain loadable.
