import { APICallError } from 'ai';
import { AiModelError } from '../lib/index.js';
import { FakeModel } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

// Start Node with TZ=America/Sao_Paulo. No provider credentials or network requests.
const now = Date.parse('2026-11-13T12:00:00Z');
const failure = (value: string) =>
  new APICallError({
    message: 'Rate limited',
    url: 'https://provider.invalid',
    requestBodyValues: {},
    statusCode: 429,
    isRetryable: true,
    responseHeaders: { 'retry-after': value },
  });

describe('Retry-After HTTP dates', () => {
  afterEach(() => vi.useRealTimers());

  it.each([
    ['IMF-fixdate', 'Fri, 13 Nov 2026 12:00:02 GMT'],
    ['RFC850', 'Friday, 13-Nov-26 12:00:02 GMT'],
    ['asctime', 'Fri Nov 13 12:00:02 2026'],
  ])('%s represents a two-second wait in UTC', async (_format, header) => {
    const times: number[] = [];
    const model = new FakeModel(() => {
      times.push(Date.now());
      return times.length === 1 ? failure(header) : 'ok';
    });
    const { moduleRef, ai } = await createAi({ models: { default: model } });
    try {
      vi.useFakeTimers();
      vi.setSystemTime(now);
      const pending = ai.generateText({
        prompt: 'Hi',
        timeout: false,
        // Expose the parsed wait; the default 30s cap would reject the retry instead.
        retry: { attempts: 2, backoff: { delay: 0, maxDelay: '4h' } },
      });
      await vi.runAllTimersAsync();
      expect((await pending).text).toBe('ok');
      expect(times[1]! - times[0]!).toBe(2000);
    } finally {
      vi.useRealTimers();
      await moduleRef.close();
    }
  });

  it('asctime does not trigger an unnecessary fallback at the default cap', async () => {
    const primary = new FakeModel([
      failure('Fri Nov 13 12:00:02 2026'),
      'primary answer',
    ]);
    const backup = new FakeModel(['backup answer']);
    const { moduleRef, ai } = await createAi({
      models: { default: { model: primary, fallback: 'backup' }, backup },
    });
    try {
      vi.useFakeTimers();
      vi.setSystemTime(now);
      const pending = ai.generateText({
        prompt: 'Hi',
        timeout: false,
        retry: { attempts: 2, backoff: { delay: 0 } },
      });
      await vi.runAllTimersAsync();
      expect((await pending).text).toBe('primary answer');
      expect(backup.calls).toHaveLength(0);
    } finally {
      vi.useRealTimers();
      await moduleRef.close();
    }
  });

  it('RFC850 year 75 means 2075 when the clock is in 2026', async () => {
    const future = new Date('2075-11-13T12:00:02Z');
    const weekday = [
      'Sunday',
      'Monday',
      'Tuesday',
      'Wednesday',
      'Thursday',
      'Friday',
      'Saturday',
    ][future.getUTCDay()];
    const model = new FakeModel([
      failure(`${weekday}, 13-Nov-75 12:00:02 GMT`),
      'too early',
    ]);
    const { moduleRef, ai } = await createAi({ models: { default: model } });
    try {
      vi.useFakeTimers();
      vi.setSystemTime(now);
      const pending = ai
        .generateText({
          prompt: 'Hi',
          timeout: false,
          retry: { attempts: 2, backoff: { delay: 0 } },
        })
        .catch((error: unknown) => error);
      await vi.runAllTimersAsync();
      // The valid date is far beyond maxDelay, so this model must not be retried.
      expect(await pending).toBeInstanceOf(AiModelError);
      expect(model.calls).toHaveLength(1);
    } finally {
      vi.useRealTimers();
      await moduleRef.close();
    }
  });
});
