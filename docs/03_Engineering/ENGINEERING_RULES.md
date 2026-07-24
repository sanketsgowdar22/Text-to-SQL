# Engineering Rules

> **Document Status**: ✅ MANDATORY
> **Effective Date**: 2026-07-25
> **Scope**: All development work on the ADIP repository

> [!CAUTION]
> Before starting ANY implementation work, you MUST read this document in full. Violations of these rules require immediate remediation before the work can be merged.

---

## Rule 1 — One Task = One Branch

Every implementation task MUST be performed on its own feature branch.

- Branch from `development`
- Branch name: `feature/<phase-number>-<task-id>-<short-description>`
- Example: `feature/02-AUTH-001-user-orm-model`
- Never combine multiple tasks in a single branch
- Never reuse a branch for a different task

---

## Rule 2 — Never Implement Outside Task Scope

A feature branch MUST only contain changes related to its assigned task.

- Do not fix unrelated bugs
- Do not refactor unrelated code
- Do not update unrelated documentation
- If you discover an issue outside your task scope, create a separate issue

---

## Rule 3 — Never Modify Unrelated Files

Only modify files that are directly relevant to the current task.

- Check the task document for the list of expected files
- If you need to modify a file not listed, update the task document first
- Incidental formatting changes to unrelated files are prohibited

---

## Rule 4 — Always Update Documentation First

Before writing implementation code:

1. Ensure the task document exists and is complete
2. Update any relevant documentation (API spec, architecture, etc.)
3. Then proceed to implementation

Documentation changes MUST be in the same PR as the implementation.

---

## Rule 5 — Always Update CHANGELOG

Every PR that changes functionality MUST include a `CHANGELOG.md` update.

- Use [Keep a Changelog](https://keepachangelog.com/) format
- Categorize: Added, Changed, Deprecated, Removed, Fixed, Security
- Reference the Task ID

---

## Rule 6 — Always Update API Documentation

If your task adds or modifies API endpoints:

1. Update `docs/api_specification.md`
2. Verify the OpenAPI schema reflects the changes
3. Include request/response examples

---

## Rule 7 — Always Run Tests

Before committing:

1. Run the full test suite: `pytest -v`
2. All existing tests MUST pass
3. New tests MUST be written for new functionality
4. Coverage must not decrease below 80%

---

## Rule 8 — Always Create a Pull Request

All code reaches `development` through a Pull Request. No exceptions.

- Use the [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
- Fill in ALL sections — do not skip any
- Link the task document in the PR description

---

## Rule 9 — Always Create a Review Summary

Every PR must include:

- Summary of changes
- Testing evidence (test output, coverage report)
- Screenshots for UI changes
- Risks and mitigation

---

## Rule 10 — Always Push to GitHub

Commits must be pushed to the remote repository. Local-only branches are not acceptable.

- Push regularly (at least once per work session)
- Use descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/)

---

## Rule 11 — Always Merge Through Pull Request

The ONLY way to merge code to `development` or `main` is through a Pull Request.

- `development` ← `feature/*` branches (via PR)
- `main` ← `development` (via release PR)

---

## Rule 12 — Never Push Directly to Development

Direct pushes to `development` are prohibited.

- All changes go through feature branches → PR → merge
- Emergency hotfixes still require a branch and PR

---

## Rule 13 — Never Push Directly to Main

Direct pushes to `main` are prohibited.

- `main` only receives merges from `development` via release PRs
- `main` represents production-ready code only

---

## Rule 14 — Never Skip Testing

Every implementation MUST include tests.

- Unit tests for business logic
- Integration tests for API endpoints
- Edge case coverage for validation and error handling
- Skipping tests is grounds for PR rejection

---

## Rule 15 — Never Skip Documentation

Every implementation MUST have corresponding documentation.

- Task document completed before implementation begins
- Code comments for non-obvious logic
- Docstrings for all public classes and functions
- API documentation for all endpoints
- README updates when applicable

---

## Rule 16 — Never Skip Review

Every PR MUST be reviewed before merging.

- Reviewer checks the [Review Checklist](../17_Reviews/README.md)
- At minimum, verify: tests pass, scope is correct, docs updated
- Self-review is acceptable for solo developers but must follow the checklist

---

## Rule 17 — Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `build`

Examples:
- `feat(auth): implement user registration endpoint`
- `fix(database): resolve connection pool leak`
- `docs(phase-02): add authentication phase document`
- `test(auth): add JWT token expiration tests`

---

## Rule 18 — Delete Feature Branches After Merge

After a PR is merged:

1. Delete the feature branch from remote
2. Delete the local branch
3. Pull latest `development`

This keeps the branch list clean and prevents confusion.

---

## Enforcement

These rules are enforced through:

1. **PR Template**: Requires checklist completion
2. **CI Pipeline**: Automated lint, type-check, and test verification
3. **Code Review**: Reviewers verify compliance
4. **Branch Protection**: `main` and `development` require PR-based merges

---

## Exceptions

There are NO exceptions to these rules. If a rule seems inappropriate for a specific situation, propose a rule change through an ADR before proceeding.
