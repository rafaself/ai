# Retry-After bug reproductions

This branch adds two reproduction specs to upstream nestjs/ai commit
cf58ae9254b62fd56337f97f6237a63fec00d7e6. It changes no library code or dependencies.

Both specs call AiService.generateText() with FakeModel and fake timers. They make
no provider requests and need no credentials. Assertions describe expected behavior;
failures are the reproduction result, not a test setup failure.

## Run

Use Node.js >=22, as required by the repository.

```bash
git clone --single-branch --branch repro/retry-after-handling https://github.com/rafaself/ai.git ai-retry-after-repro
cd ai-retry-after-repro
npm ci
npm run test:e2e -- tests/repro-retry-after-custom-backoff.spec.ts --reporter=verbose
TZ=America/Sao_Paulo npm run test:e2e -- tests/repro-retry-after-http-date.spec.ts --reporter=verbose
```

## Results before a fix

- Custom-backoff spec: two controls pass with object backoff; two function-backoff
  cases fail because the observed wait is 0ms instead of 2000ms. Covers both
  Retry-After and retry-after-ms.
- HTTP-date spec under America/Sao_Paulo: IMF-fixdate and ordinary RFC850 controls
  pass. asctime waits 10802000ms rather than 2000ms when maxDelay permits it;
  with the default cap it selects the fallback unnecessarily. The RFC850 year-75
  case retries immediately rather than rejecting the excessive provider delay.
  Three assertions fail and two controls pass.

The date tests freeze the clock at 2026-11-13T12:00:00Z. Multi-hour waits advance
instantly through fake timers. The cutoff itself is existing policy, not the bug.

Validated on Linux, Node.js v25.9.0, Vitest 5.1.0, with installed @nestjs/core
12.4.0, @nestjs/resilience 0.0.3, and ai 7.0.165. The source package version is
@nestjs/ai 0.0.2. npm ci uses the upstream repository's committed lockfile.
