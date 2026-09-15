# Release Procedure

Push reviewed changes to `main`, then create and push a new stable version tag
(`vMAJOR.MINOR.PATCH`) on the intended commit. Never move an existing release tag.

`.github/workflows/runtime.yml` builds `psytrainer-ml-skill.zip` from committed
files only. Every push and pull request tests that archive in fresh environments
across Windows, macOS and Linux with Python 3.12, 3.13 and 3.14. Each environment
runs dependency checks, the full test suite and real training/prediction/report
smoke tests. Development `tests/` and `fixtures/` are excluded from downloads;
CI adds them from the matching source revision for release QA only. Users do not
run sample training during setup. The ZIP comment identifies its exact source commit.

A stable tag push publishes only after all nine environments succeed. The publish
job uploads the same tested archive to a draft Release, downloads it to confirm
byte-for-byte identity, then publishes it as Latest. Source pushes alone do not
update Release downloads. A failed build or test leaves the previous Latest intact.

Check the Actions run and Release attachment before announcing availability.
If publication fails after creating a draft, inspect that draft and the failed
step before retrying; the workflow never overwrites an existing release.

Local packaging: `python scripts/build_skill.py --ref HEAD --output /path/to/psytrainer-ml-skill.zip`.
The destination must not exist. Local uncommitted and untracked files are excluded.
This package contains the Skill files; the runtime setup script installs Python
dependencies on demand on the destination machine. Release QA explicitly uses
`scripts/install_runtime.py --all` to exercise every supported algorithm.
