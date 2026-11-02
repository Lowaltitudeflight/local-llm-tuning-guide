# Security Policy

## What this project covers

This repository contains documentation, benchmark data, and small standard
library tools. It does not ship model weights, container images, or a network
service. The relevant risks are therefore mostly:

- a benchmark JSON or recipe that leaks a private hostname, IP, token, or key;
- a tool that executes an untrusted path or command;
- a documentation change that recommends an insecure deployment (open port,
  no authentication, secrets in a compose file).

## Reporting

Please **do not open a public issue** for a security problem. Use GitHub’s
private vulnerability reporting (Security → Advisories → “Report a
vulnerability”) or contact a maintainer directly on GitHub.

Include:

- the file or tool involved;
- a minimal reproduction (command, input file);
- the impact and any suggested fix.

We aim to acknowledge a report within a few days. There is no bug bounty.

## Handling data and secrets

- Benchmark JSON must not contain real hostnames, usernames, passwords, API
  keys, or internal IPs. Replace them with placeholders before submitting.
- Recipe commands should use `<your-model>`, `$API_KEY`, or similar
  placeholders, never a literal secret.
- If a secret was committed, rotate it first; removing the file from the
  current commit is not enough because history retains it.

## Supported versions

The project is documentation-first; fixes land on `main`. There are no
long-lived release branches yet.
