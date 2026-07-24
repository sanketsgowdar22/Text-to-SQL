# 17 — Review Process Documentation

Code review standards, checklists, and approval workflows.

---

## Code Review Process

### Review Requirements

| Criteria | Requirement |
|----------|-------------|
| **Minimum Reviewers** | 1 (recommended: 2 for architectural changes) |
| **Review Scope** | All files changed in the PR |
| **Testing Evidence** | All tests passing (CI pipeline green) |
| **Documentation** | Relevant docs updated in the same PR |

### Review Checklist

Every reviewer should verify:

- [ ] Code follows [Coding Standards](../15_Standards/README.md)
- [ ] All existing tests pass
- [ ] New tests cover the changes
- [ ] Documentation is updated
- [ ] No unrelated files modified
- [ ] Commit messages follow [Git Workflow](../13_Git/GIT_WORKFLOW.md) conventions
- [ ] PR description is complete per the [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
- [ ] Security implications considered
- [ ] Performance implications considered
- [ ] Error handling is comprehensive

### Review Outcomes

| Outcome | Meaning | Action |
|---------|---------|--------|
| ✅ Approved | Meets all criteria | Merge to `development` |
| 🔄 Changes Requested | Issues found | Address feedback, re-request review |
| ❌ Rejected | Fundamental issues | Close PR, redesign approach |

---

## Related Documentation

- [Engineering Rules](../03_Engineering/ENGINEERING_RULES.md)
- [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
- [Review Report Template](../16_Templates/REVIEW_REPORT_TEMPLATE.md)
