# 13 — Git Documentation

Git workflow, branch strategy, commit conventions, and merge processes.

---

## Contents

| Document | Description |
|----------|-------------|
| [Git Workflow](GIT_WORKFLOW.md) | Complete Git branching strategy, commit conventions, and merge process |

---

## Quick Reference

| Branch | Purpose | Merge Target |
|--------|---------|-------------|
| `main` | Production-ready releases | *(protected)* |
| `development` | Integration branch | `main` (via release PR) |
| `feature/<phase>-<task>-<desc>` | Individual task work | `development` (via PR) |

---

## Related Documentation

- [Engineering Rules](../03_Engineering/ENGINEERING_RULES.md)
- [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
- [Review Process](../17_Reviews/README.md)
