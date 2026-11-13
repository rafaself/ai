import { APICallError } from 'ai';
import { FakeModel } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

// No provider credentials or network requests. Run with the repository's Vitest config.
describe('Retry-After with custom backoff', () => {
  afterEach(() => vi.useRealTimers());

  const headerCases: Record<string, string>[] = [
    { 'retry-after': '2' },
    { 'retry-after-ms': '2000' },
  ];
  for (const headers of headerCases) {
    for (const custom of [false, true]) {
      it(`${JSON.stringify(headers)} with ${custom ? 'function' : 'object'} backoff waits 2000ms`, async () => {
        const times: number[] = [];
        const failure = new APICallError({
          message: 'Rate limited',
          url: 'https://provider.invalid',
          requestBodyValues: {},
          statusCode: 429,
          isRetryable: true,
          responseHeaders: headers,
        });
        const model = new FakeModel(() => {
          times.push(Date.now());
          return times.length === 1 ? failure : 'ok';
        });
        const { moduleRef, ai } = await createAi({
          models: { default: model },
        });
        try {
          vi.useFakeTimers();
          const pending = ai.generateText({
            prompt: 'Hi',
            timeout: false,
            retry: { attempts: 2, backoff: custom ? () => 0 : { delay: 0 } },
          });
          await vi.runAllTimersAsync();
          expect((await pending).text).toBe('ok');
          expect(times).toHaveLength(2);
          expect(times[1]! - times[0]!).toBe(2000);
        } finally {
          vi.useRealTimers();
          await moduleRef.close();
        }
      });
    }
  }
});
