# Nimbus Cloud Storage — Incident Response Runbook

## Severity Levels

- **SEV1**: Full outage or customer data exposure. Page on-call immediately,
  notify leadership within 15 minutes, public status page updated within 30
  minutes.
- **SEV2**: Partial outage or major feature degradation affecting a subset of
  customers. Page on-call; status page updated within 1 hour.
- **SEV3**: Minor bug or degraded performance with a workaround available. No
  page required; ticket filed for next business day.

## On-Call Rotation

Engineering on-call rotates weekly, Monday 9am to the following Monday 9am.
The on-call engineer must acknowledge a page within 10 minutes; if
unacknowledged, PagerDuty escalates to the secondary on-call after 10 minutes,
then to the engineering manager after 20 minutes.

## Postmortems

Every SEV1 and SEV2 incident requires a blameless postmortem published within
5 business days of resolution. Postmortems must include a timeline, root
cause, and at least one concrete action item with an owner and due date.

## Rollback Procedure

If a deploy is suspected to have caused an incident, the on-call engineer
should roll back immediately using the `nimbus-deploy rollback` command
rather than attempting a forward fix, unless the rollback itself is judged
riskier than the current impact.
