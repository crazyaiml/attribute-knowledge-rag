# Contributing to AK-RAG

Thanks for your interest in contributing. This project is developed entirely through pull requests — direct pushes to `main` are disabled by repository ruleset, so the workflow below is required for every change, including from maintainers.

## Workflow

1. **Fork** the repository (or create a branch if you have write access).
2. **Branch** off `main` with a descriptive name, e.g. `fix/rrf-stopword-match` or `feat/bedrock-embedding-batching`.
3. **Make your change**, keeping it scoped to one concern per PR.
4. **Add or update tests** under `tests/` for any behavior change. PRs that change retrieval, decision, or contract logic without a test are unlikely to be merged as-is.
5. **Run the checks locally** before opening the PR (see below).
6. **Open a pull request** against `main` using the PR template. Link any related issue.
7. Address review feedback with new commits — no need to force-push or squash until a maintainer asks for it.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sentence]"   # sentence-transformers keeps embedding local, no API key needed
```

Run the backend:

```bash
python3 -m pytest
ruff check src tests
```

Run the frontend:

```bash
cd frontend
npm install
npm run lint
npm run build
```

## Pull Request Expectations

- **One logical change per PR.** Unrelated fixes or refactors should be split out.
- **Tests pass.** CI runs `pytest`, `ruff check`, and the frontend build/lint on every PR — all must be green before merge.
- **No secrets.** Never commit `.env`, API keys, or real credentials. `.env.example` documents the shape of config without values.
- **Follow existing patterns.** New LLM/embedding/search providers should implement the existing abstract base classes (`akrag.llm.base.LLMProvider`, `akrag.embeddings.base.EmbeddingProvider`, `akrag.search.base.SearchBackend`) rather than introducing parallel abstractions.
- **Update docs** when behavior, configuration, or provider support changes — particularly the Provider Matrix in `README.md`.

## Reporting Bugs / Requesting Features

Use the issue templates. For bugs, include the exact query/phrase, expected vs. actual attribute resolution, and your provider configuration (LLM/embedding/search) — most retrieval bugs are provider- or threshold-specific.

## Security Issues

Do not open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful and constructive.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), consistent with the rest of the project.
