# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AK-RAG, please **do not** open a
public GitHub issue. Instead:

1. Use GitHub's [private vulnerability reporting](../../security/advisories/new) for this repository, or
2. Contact the maintainers directly through the repository owner's GitHub profile.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal repro is ideal).
- The affected provider configuration (LLM/embedding/search backend), if relevant.

We will acknowledge reports within a reasonable timeframe and keep you updated as the issue is triaged and fixed.

## Scope Notes

AK-RAG is a reference architecture that, in production deployments, may sit in front of governed enterprise data (including PHI/PII depending on the attribute catalog loaded). Security-relevant areas to flag include:

- Bypass of the governance layer (PHI/consent/channel/minimum-cell-size checks) before DSL generation.
- Injection risks in query construction against the configured search backend.
- Leakage of API keys or credentials through logs, error messages, or API responses.
- Any path by which the LLM could cause an ungoverned `attribute_id` to appear in output.

## Supported Versions

This project is in early development (pre-1.0). Security fixes are applied to `main` only; there is no long-term support branch yet.
