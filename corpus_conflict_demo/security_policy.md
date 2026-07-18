# Nimbus Cloud Storage — Security Policy

## Password Requirements

All employee accounts must use passwords that are at least 14 characters long
and include a mix of uppercase, lowercase, numbers, and symbols. Passwords
must be rotated every 180 days. Reusing any of the last 10 passwords is
prohibited.

## Multi-Factor Authentication

MFA is mandatory for all accounts with access to production systems or
customer data. Approved MFA methods are: hardware security keys (preferred),
authenticator apps (TOTP), or SMS (only when the first two are unavailable).

## Data Classification

Customer data is classified into three tiers:

- **Tier 1 (Public)**: Marketing material, public documentation.
- **Tier 2 (Internal)**: Internal metrics, non-sensitive logs.
- **Tier 3 (Restricted)**: Customer files, billing information, authentication
  credentials. Access to Tier 3 data requires manager approval and is logged
  and audited monthly.

## Incident Reporting

Any suspected security incident — including lost devices, phishing attempts,
or unauthorized access — must be reported to security@nimbus.example within
1 hour of discovery. Do not attempt to independently investigate a suspected
breach; escalate immediately to the security team.
