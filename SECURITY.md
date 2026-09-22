# Security Policy

Do not report secrets, credentials, API keys, restart secrets, model tokens or
private notebook URLs in public issues.

For normal security-sensitive configuration:

- keep `.env` out of version control;
- require API authentication on exposed endpoints;
- use a separate restart secret;
- treat Quick Tunnel as demonstration ingress, not a permanent security
  boundary;
- rotate any credential that may have appeared in logs or screenshots.

If a vulnerability can be described safely without exposing credentials, open
a GitHub issue with the minimum reproduction details needed to investigate.
