# 18 — Release Process Documentation

Release management, versioning strategy, and deployment procedures.

---

## Versioning Strategy

This project follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

| Component | Increment When |
|-----------|---------------|
| **MAJOR** | Breaking API changes |
| **MINOR** | New features (backwards compatible) |
| **PATCH** | Bug fixes (backwards compatible) |

### Current Version: `0.1.0`

---

## Release Process

1. All phase tasks merged to `development`
2. Create release PR: `development` → `main`
3. Update `CHANGELOG.md` with release notes
4. Update version in `pyproject.toml` and `app/core/config.py`
5. Tag the release: `git tag -a v<version> -m "Release v<version>"`
6. Push tag: `git push origin v<version>`

---

## Related Documentation

- [CHANGELOG](../../CHANGELOG.md)
- [Git Workflow](../13_Git/GIT_WORKFLOW.md)
- [Release Report Template](../16_Templates/RELEASE_REPORT_TEMPLATE.md)
