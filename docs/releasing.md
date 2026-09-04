# Releasing Mazinger

Releases go to PyPI through GitHub Actions using **PyPI Trusted Publishing**
(OIDC). No API token is stored in this repository or in GitHub Secrets: PyPI is
configured to trust one specific workflow file, in one specific repository,
running inside one specific GitHub Environment, and hands that job a
short-lived (15 minute), project-scoped token at upload time.

The workflow lives at [.github/workflows/publish.yml](../.github/workflows/publish.yml).

---

## One-time setup

Do these three steps once. After that, releasing is just "tag and publish a
GitHub Release".

### 1. Create the GitHub Environments

In the repo: **Settings → Environments → New environment**.

Create two, named exactly:

| Environment | Used by |
| ----------- | ------- |
| `pypi`      | Real releases (`https://pypi.org/p/mazinger`) |
| `testpypi`  | Manual dry runs (`https://test.pypi.org/p/mazinger`) |

For `pypi`, add protection rules — this is the main reason to use an
environment at all:

- **Required reviewers** → add yourself (and any co-maintainers). Every upload
  to the real index then waits for a human click.
- **Deployment branches and tags** → restrict to `Protected tags` (or the
  `v*` tag pattern) so the environment cannot be reached from an arbitrary
  branch.

No secrets go in either environment.

### 2. Add the Trusted Publisher on PyPI

Because `mazinger` **already exists** on PyPI, use the per-project form:

1. Go to <https://pypi.org/manage/projects/>.
2. Click **Manage** on `mazinger`.
3. Choose **Publishing** in the sidebar.
4. Under "Add a new publisher", pick **GitHub**, and fill in:

   | Field | Value |
   | ----- | ----- |
   | Owner | `bakrianoo` |
   | Repository name | `mazinger` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

5. Click **Add**.

The environment name is technically optional, but PyPI strongly recommends it
and this workflow depends on it — leaving it blank would let *any* job in the
repo publish, bypassing the required-reviewer gate from step 1.

> **Note on the workflow filename.** PyPI matches on the filename only
> (`publish.yml`), not the full path. Renaming or moving the workflow breaks
> publishing until the trusted publisher entry is updated to match.

### 3. (Optional) Do the same on TestPyPI

Repeat step 2 at <https://test.pypi.org/manage/projects/>, using environment
name `testpypi`. If `mazinger` does not exist on TestPyPI yet, use the
**pending publisher** form at
<https://test.pypi.org/manage/account/publishing/> instead — it reserves the
name and converts to a normal trusted publisher on the first successful upload.

Skip this entirely if you don't want dry runs.

---

## Cutting a release

1. **Bump the version** in `pyproject.toml`:

   ```toml
   [project]
   version = "1.9.6"
   ```

   `mazinger.__version__` is read from the installed distribution metadata, so
   there is nothing else to edit.

2. **Commit and push** to `master`.

3. **Tag and publish a GitHub Release** with the tag `v<version>`:

   ```bash
   gh release create v1.9.6 --title "v1.9.6" --generate-notes
   ```

   The tag must match `pyproject.toml`; the build job fails loudly if it
   doesn't, before anything is uploaded.

4. **Approve the deployment.** If you configured required reviewers, the
   `Publish to PyPI` job pauses until approved from the Actions run page.

5. Done — the release appears at <https://pypi.org/project/mazinger/>.

### Dry run against TestPyPI

**Actions → Publish to PyPI → Run workflow**, leaving the target as
`testpypi`. Then verify:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ mazinger
```

(The extra index is needed because Mazinger's dependencies only live on the
real PyPI.)

---

## What the workflow actually does

| Job | What it does | Permissions |
| --- | ------------ | ----------- |
| `build` | Verifies tag ↔ version, builds sdist + wheel, runs `twine check --strict`, installs the wheel into a clean venv and checks `import mazinger` and `mazinger --help` work | `contents: read` |
| `publish-testpypi` | Uploads to TestPyPI (manual dispatch only) | `id-token: write` |
| `publish-pypi` | Uploads to PyPI | `id-token: write` |

Building and publishing are separate jobs on purpose: only the tiny upload jobs
ever hold `id-token: write`, so the OIDC credential is never exposed to build
scripts or dependency code.

Uploads also carry [PEP 740 attestations](https://peps.python.org/pep-0740/),
which `pypa/gh-action-pypi-publish` generates automatically under Trusted
Publishing — PyPI records which workflow run produced each file.

---

## Troubleshooting

**`invalid-publisher` / "not a valid publisher for this project"**
The OIDC claims don't match the entry on PyPI. Check owner, repo name,
workflow *filename*, and environment name — all four must match exactly, and
the job must actually be running in the `pypi` environment.

**`Missing credentials` / no OIDC token**
The job is missing `permissions: id-token: write`, or it's running from a fork
(pull requests from forks cannot mint OIDC tokens — that's intended).

**`File already exists`**
That version was already uploaded; PyPI never allows re-uploading a version.
Bump the version and cut a new release.

**Tag/version mismatch**
Delete the tag and release, fix `pyproject.toml`, then re-tag:

```bash
git tag -d v1.9.6 && git push origin :refs/tags/v1.9.6
gh release delete v1.9.6 --yes
```
