/**
 * Model calls: options and their precedence, retries, fallback, timeouts, the caller's abort, usage events, cost
 * and budgets. Fake models only.
 */
import * as z from 'zod';
import { Logger } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import { APICallError } from 'ai';
import {
  AiBudgetExceededError,
  AiConfigurationError,
  AiModelError,
  type AiUsageEvent,
} from '../lib/index.js';
import { ResilienceTimeoutError } from '@nestjs/resilience';
import { FakeModel, providerError } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

const noWait = { attempts: 3, backoff: { delay: 0, jitter: 'none' as const } };

function apiErrorWithHeaders(responseHeaders: Record<string, string>): APICallError {
  return new APICallError({
    message: 'The provider answered 429',
    url: 'https://fake-provider.test/v1/messages',
    requestBodyValues: {},
    statusCode: 429,
    responseHeaders,
  });
}

describe('model calls', () => {
  let moduleRef: TestingModule | undefined;

  afterEach(async () => {
    vi.useRealTimers();
    await moduleRef?.close();
    moduleRef = undefined;
  });

  describe('options', () => {
    it('sends the prompt, the instructions and the generation settings to the model', async () => {
      const model = new FakeModel(['Billing']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;

      const result = await ai.generateText({
        instructions: 'Classify the ticket: Billing, Shipping or Other.',
        prompt: 'I was charged twice',
        temperature: 0,
        maxOutputTokens: 5,
        seed: 7,
        stopSequences: ['\n'],
        providerOptions: { fake: { mode: 'strict' } },
        headers: { 'x-team': 'support' },
      });

      expect(result).toMatchObject({ text: 'Billing', finishReason: 'stop', steps: [{ index: 0, attempts: 1 }] });
      expect(result.model).toEqual({ name: 'default', provider: 'fake', modelId: 'fake-model' });
      const call = model.lastCall!;
      expect(call.instructions).toBe('Classify the ticket: Billing, Shipping or Other.');
      expect(call.lastUserMessage).toBe('I was charged twice');
      expect(call.options).toMatchObject({
        temperature: 0,
        maxOutputTokens: 5,
        seed: 7,
        stopSequences: ['\n'],
        providerOptions: { fake: { mode: 'strict' } },
      });
      expect(call.options.headers).toMatchObject({ 'x-team': 'support' });
    });

    it('takes the call settings over the model settings over the module defaults', async () => {
      const model = new FakeModel(() => 'ok');
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model, temperature: 0.5, maxOutputTokens: 100 } },
        defaults: { temperature: 1, maxOutputTokens: 1000, providerOptions: { fake: { from: 'defaults' } } },
      });
      moduleRef = ref;

      await ai.generateText({ prompt: 'a' });
      expect(model.lastCall!.options).toMatchObject({ temperature: 0.5, maxOutputTokens: 100, providerOptions: { fake: { from: 'defaults' } } });
      await ai.generateText({ prompt: 'b', temperature: 0 });
      expect(model.lastCall!.options).toMatchObject({ temperature: 0, maxOutputTokens: 100 });
    });

    it('sends a conversation as it is, and refuses a call with both or neither', async () => {
      const model = new FakeModel(() => 'ok');
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;

      await ai.generateText({
        messages: [
          { role: 'user', content: 'Where is my order?' },
          { role: 'assistant', content: 'Which one?' },
          { role: 'user', content: 'ord_1' },
        ],
      });
      expect(model.lastCall!.options.prompt.map((message) => message.role)).toEqual(['user', 'assistant', 'user']);
      expect(model.lastCall!.lastUserMessage).toBe('ord_1');

      await expect(ai.generateText({ prompt: 'a', messages: [{ role: 'user', content: 'b' }] })).rejects.toThrow(AiConfigurationError);
      await expect(ai.generateText({})).rejects.toThrow('Pass `prompt`');
    });

    it('accepts a LanguageModel in place of a name', async () => {
      const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel([]) } });
      moduleRef = ref;
      const adhoc = new FakeModel(['ad hoc'], { provider: 'other', modelId: 'm-1' });
      const result = await ai.generateText({ model: adhoc, prompt: 'Hi' });
      expect(result.text).toBe('ad hoc');
      expect(result.model).toEqual({ provider: 'other', modelId: 'm-1' });
    });
  });

  describe('retries', () => {
    it('retries what the provider marks retryable, then answers', async () => {
      const model = new FakeModel([providerError(429), providerError(503), 'third time']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: noWait } });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Hi' });
      expect(result.text).toBe('third time');
      expect(result.steps[0]!.attempts).toBe(3);
      expect(model.calls).toHaveLength(3);
    });

    it('does not retry a client error, and reports every attempt', async () => {
      const model = new FakeModel([providerError(400, { message: 'Bad request' }), 'never']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: noWait } });
      moduleRef = ref;

      const error = await ai.generateText({ prompt: 'Hi' }).catch((e: unknown) => e);
      expect(error).toBeInstanceOf(AiModelError);
      expect((error as AiModelError).attempts).toHaveLength(1);
      expect((error as AiModelError).message).toContain('Bad request');
      expect((error as AiModelError).cause).toMatchObject({ statusCode: 400 });
      expect(model.calls).toHaveLength(1);
    });

    it('gives up after `attempts`', async () => {
      const model = new FakeModel(() => providerError(500));
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: { ...noWait, attempts: 2 } } });
      moduleRef = ref;

      const error = (await ai.generateText({ prompt: 'Hi' }).catch((e: unknown) => e)) as AiModelError;
      expect(error).toBeInstanceOf(AiModelError);
      expect(error.attempts.map((attempt) => attempt.attempt)).toEqual([1, 2]);
      expect(model.calls).toHaveLength(2);
    });

    it("honors the provider's retry-after with object backoff and enforces maxDelay", async () => {
      vi.useFakeTimers();
      const model = new FakeModel([providerError(429, { retryAfter: 0.05 }), 'after the wait']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: noWait } });
      moduleRef = ref;

      const generation = ai.generateText({ prompt: 'Hi' });
      await vi.advanceTimersByTimeAsync(49);
      expect(model.calls).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect((await generation).text).toBe('after the wait');
      expect(model.calls).toHaveLength(2);

      const limited = new FakeModel([providerError(429, { retryAfter: 120 }), 'never']);
      const limitedGeneration = ai
        .generateText({ model: limited, prompt: 'Hi', retry: { ...noWait, backoff: { delay: 0, maxDelay: '1s' } } })
        .catch((e: unknown) => e);
      await vi.advanceTimersByTimeAsync(0);
      const error = (await limitedGeneration) as AiModelError;
      expect(error.attempts).toHaveLength(1);
      expect(limited.calls).toHaveLength(1);
    });

    it.each([
      { header: 'retry-after', error: providerError(429, { retryAfter: 0.05 }) },
      { header: 'retry-after-ms', error: apiErrorWithHeaders({ 'retry-after-ms': '50' }) },
    ])('honors $header ahead of function backoff', async ({ error }) => {
      vi.useFakeTimers();
      const model = new FakeModel([error, 'after the wait']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;
      const backoff = vi.fn(() => 0);

      const generation = ai.generateText({ prompt: 'Hi', retry: { attempts: 2, backoff } });
      await vi.advanceTimersByTimeAsync(49);
      expect(model.calls).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect((await generation).text).toBe('after the wait');
      expect(backoff).not.toHaveBeenCalled();
      expect(model.calls).toHaveLength(2);
    });

    it('uses function backoff when retry-after is invalid', async () => {
      vi.useFakeTimers();
      const model = new FakeModel([apiErrorWithHeaders({ 'retry-after': 'invalid' }), 'after the wait']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;
      const backoff = vi.fn(() => 25);

      const generation = ai.generateText({ prompt: 'Hi', retry: { attempts: 2, backoff } });
      await vi.advanceTimersByTimeAsync(24);
      expect(model.calls).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect((await generation).text).toBe('after the wait');
      expect(backoff).toHaveBeenCalledOnce();
      expect(model.calls).toHaveLength(2);
    });

    it('does not retry when retryIf returns false', async () => {
      vi.useFakeTimers();
      const model = new FakeModel([providerError(429, { retryAfter: 0.05 }), 'never']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;
      const backoff = vi.fn(() => 0);
      const retryIf = vi.fn(() => false);

      const generation = ai
        .generateText({ prompt: 'Hi', retry: { attempts: 2, backoff, retryIf } })
        .catch((error: unknown) => error);
      await vi.advanceTimersByTimeAsync(0);

      expect(await generation).toBeInstanceOf(AiModelError);
      expect(retryIf).toHaveBeenCalledOnce();
      expect(backoff).not.toHaveBeenCalled();
      expect(model.calls).toHaveLength(1);
    });

    it('cancels function backoff without beginning another request', async () => {
      vi.useFakeTimers();
      const model = new FakeModel([providerError(503), 'never']);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;
      const controller = new AbortController();
      const backoff = vi.fn(() => 5_000);
      const generation = ai.generateText({
        prompt: 'Hi',
        abortSignal: controller.signal,
        retry: { attempts: 2, backoff },
      });
      await vi.advanceTimersByTimeAsync(0);
      expect(backoff).toHaveBeenCalledOnce();
      expect(model.calls).toHaveLength(1);

      const reason = new Error('retry delay cancelled');
      controller.abort(reason);
      await expect(generation).rejects.toBe(reason);
      expect(model.calls).toHaveLength(1);
    });

    it('retries what retryIf says, and nothing with retry: false', async () => {
      const model = new FakeModel([new Error('flaky'), 'ok', new Error('flaky')]);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;

      const retryIf = vi.fn((error: unknown) => (error as Error).message === 'flaky');
      expect((await ai.generateText({ prompt: 'a', retry: { ...noWait, retryIf } })).text).toBe('ok');
      expect(retryIf).toHaveBeenCalledWith(expect.objectContaining({ message: 'flaky' }), 1);

      await expect(ai.generateText({ prompt: 'b', retry: false })).rejects.toThrow(AiModelError);
    });
  });

  describe('fallback', () => {
    it("tries the model's fallback when it fails after its retries", async () => {
      const primary = new FakeModel(() => providerError(529));
      const backup = new FakeModel(['from the backup']);
      const usage: AiUsageEvent[] = [];
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model: primary, fallback: 'backup' }, backup: { model: backup, retry: false } },
        defaults: { retry: { ...noWait, attempts: 2 } },
        onUsage: (event) => void usage.push(event),
      });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Hi' });
      expect(result.text).toBe('from the backup');
      expect(result.model.name).toBe('backup');
      expect(result.steps[0]!.attempts).toBe(3);
      expect(primary.calls).toHaveLength(2);
      expect(usage).toEqual([expect.objectContaining({ model: expect.objectContaining({ name: 'backup' }), fallback: true, attempt: 1 })]);
    });

    it('tries separately configured models that share a language model, with their own settings and retries', async () => {
      const warn = vi.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined);
      try {
        const shared = new FakeModel([providerError(503), providerError(503), 'from patient']);
        const { moduleRef: ref, ai } = await createAi({
          models: {
            default: { model: shared, fallback: 'patient', temperature: 0.2, retry: false },
            patient: { model: shared, temperature: 0.8, retry: { ...noWait, attempts: 2 } },
          },
        });
        moduleRef = ref;

        const result = await ai.generateText({ prompt: 'Hi' });

        expect(result.text).toBe('from patient');
        expect(result.model.name).toBe('patient');
        expect(result.steps[0]!.attempts).toBe(3);
        expect(shared.calls.map((call) => call.options.temperature)).toEqual([0.2, 0.8, 0.8]);
        expect(warn.mock.calls).toEqual([
          ['The model "default" failed (APICallError, 503: The provider answered 503); falling back to "patient".'],
        ]);
      } finally {
        warn.mockRestore();
      }
    });

    it('tries a direct fallback that shares the configured model instance', async () => {
      const shared = new FakeModel([providerError(503), 'from the direct fallback']);
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model: shared, temperature: 0.2 } },
        defaults: { retry: false },
      });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Hi', fallback: shared });

      expect(result.text).toBe('from the direct fallback');
      expect(result.model).toEqual({ provider: 'fake', modelId: 'fake-model' });
      expect(shared.calls).toHaveLength(2);
    });

    it('deduplicates repeated configured names and direct model references', async () => {
      const primary = new FakeModel(() => providerError(500));
      const backup = new FakeModel([providerError(500), 'would answer if tried twice']);
      const directPrimary = new FakeModel(() => providerError(500));
      const directBackup = new FakeModel([providerError(500), 'would answer if tried twice']);
      const { moduleRef: ref, ai } = await createAi({
        models: {
          default: { model: primary, fallback: ['backup', 'backup'] },
          backup,
        },
        defaults: { retry: false },
      });
      moduleRef = ref;

      await expect(ai.generateText({ prompt: 'configured' })).rejects.toBeInstanceOf(AiModelError);
      expect(primary.calls).toHaveLength(1);
      expect(backup.calls).toHaveLength(1);

      await expect(
        ai.generateText({ model: directPrimary, fallback: [directBackup, directBackup], prompt: 'direct' }),
      ).rejects.toBeInstanceOf(AiModelError);
      expect(directPrimary.calls).toHaveLength(1);
      expect(directBackup.calls).toHaveLength(1);
    });

    it('takes a fallback per call, or none', async () => {
      const primary = new FakeModel(() => providerError(500));
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model: primary, fallback: 'backup' }, backup: new FakeModel(() => 'backup'), other: new FakeModel(() => 'other') },
        defaults: { retry: false },
      });
      moduleRef = ref;

      expect((await ai.generateText({ prompt: 'a', fallback: ['other'] })).text).toBe('other');
      expect((await ai.generateText({ prompt: 'b', fallback: new FakeModel(['inline']) })).text).toBe('inline');
      const error = (await ai.generateText({ prompt: 'c', fallback: false }).catch((e: unknown) => e)) as AiModelError;
      expect(error.attempts.map((attempt) => attempt.model.name)).toEqual(['default']);
    });

    it('logs every fallback, with the model that failed, the next one and the cause, a 401 included', async () => {
      const warn = vi.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined);
      try {
        const { moduleRef: ref, ai } = await createAi({
          models: {
            default: { model: new FakeModel(() => providerError(401, { message: 'Invalid API key' })), fallback: ['slow', 'backup'] },
            slow: new FakeModel(() => ({ text: 'too late', delay: 1_000 })),
            backup: new FakeModel(() => 'from the backup'),
          },
          defaults: { retry: false, timeout: 50 },
        });
        moduleRef = ref;

        // A revoked key does not hide behind the fallback that answers.
        expect((await ai.generateText({ prompt: 'Hi' })).text).toBe('from the backup');
        expect(warn.mock.calls).toEqual([
          ['The model "default" failed (APICallError, 401: Invalid API key); falling back to "slow".'],
          ['The model "slow" failed (ResilienceTimeoutError: Timed out after 50ms); falling back to "backup".'],
        ]);
      } finally {
        warn.mockRestore();
      }
    });

    it('falls back only for what fallbackIf accepts', async () => {
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model: new FakeModel(() => providerError(400)), fallback: 'backup' }, backup: new FakeModel(() => 'backup') },
        defaults: { retry: false },
      });
      moduleRef = ref;

      const fallbackIf = (error: unknown) => (error as { statusCode?: number }).statusCode !== 400;
      await expect(ai.generateText({ prompt: 'a', fallbackIf })).rejects.toThrow(AiModelError);
      expect((await ai.generateText({ prompt: 'b' })).text).toBe('backup');
    });
  });

  describe('timeouts and aborts', () => {
    it('times an attempt out, retries it, and reports the timeout', async () => {
      const model = new FakeModel([{ text: 'too slow', delay: 1_000 }, { text: 'fast enough' }]);
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { retry: noWait } });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Hi', timeout: 50 });
      expect(result.text).toBe('fast enough');
      expect(result.steps[0]!.attempts).toBe(2);

      const slow = new FakeModel(() => ({ text: 'too slow', delay: 1_000 }));
      const error = (await ai.generateText({ model: slow, prompt: 'Hi', timeout: '30ms', retry: { ...noWait, attempts: 2 } }).catch((e: unknown) => e)) as AiModelError;
      expect(error).toBeInstanceOf(AiModelError);
      expect(error.cause).toBeInstanceOf(ResilienceTimeoutError);
      expect(error.attempts).toHaveLength(2);
    });

    it("rejects with the caller's abort at once, without retrying or falling back", async () => {
      const model = new FakeModel(() => ({ text: 'slow', delay: 5_000 }));
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model, fallback: 'backup' }, backup: new FakeModel(() => 'backup') },
        defaults: { retry: noWait },
      });
      moduleRef = ref;

      const controller = new AbortController();
      const call = ai.generateText({ prompt: 'Hi', abortSignal: controller.signal });
      await vi.waitFor(() => expect(model.calls).toHaveLength(1));
      controller.abort(new Error('the user left'));
      await expect(call).rejects.toThrow('the user left');
      expect(model.calls).toHaveLength(1);

      const aborted = AbortSignal.abort(new Error('already'));
      await expect(ai.generateText({ prompt: 'Hi', abortSignal: aborted })).rejects.toThrow('already');
    });

    it('refuses a malformed duration', async () => {
      const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel(() => 'ok') } });
      moduleRef = ref;
      await expect(ai.generateText({ prompt: 'Hi', timeout: '5 minutes' as never })).rejects.toThrow('`timeout` must be');
    });
  });

  describe('usage, cost and budgets', () => {
    it('reports each model call to the module and the call, with its cost', async () => {
      const model = new FakeModel([{ text: 'Billing', usage: { inputTokens: 1_000, outputTokens: 200, cachedInputTokens: 400 } }]);
      const moduleEvents: AiUsageEvent[] = [];
      const callEvents: AiUsageEvent[] = [];
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model, pricing: { input: 3, output: 15, cachedInput: 0.3 } } },
        onUsage: (event) => void moduleEvents.push(event),
      });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Triage this', label: 'triage', onUsage: (event) => void callEvents.push(event) });
      // 600 uncached input tokens at 3, 400 cached at 0.3 and 200 output at 15, per million
      const cost = (600 * 3 + 400 * 0.3 + 200 * 15) / 1_000_000;
      expect(result.usage).toEqual({ inputTokens: 1_000, outputTokens: 200, totalTokens: 1_200, cachedInputTokens: 400, reasoningTokens: 0 });
      expect(result.cost).toBeCloseTo(cost, 12);
      expect(moduleEvents).toEqual([
        expect.objectContaining({
          label: 'triage',
          model: { name: 'default', provider: 'fake', modelId: 'fake-model' },
          step: 0,
          attempt: 1,
          fallback: false,
          stream: false,
          finishReason: 'stop',
          usage: result.usage,
          cost: expect.closeTo(cost, 12),
          durationMs: expect.any(Number),
        }),
      ]);
      expect(callEvents).toEqual(moduleEvents);
    });

    it('logs what a usage listener throws, and answers anyway', async () => {
      const { moduleRef: ref, ai } = await createAi({
        models: { default: new FakeModel(() => 'ok') },
        onUsage: () => {
          throw new Error('metrics are down');
        },
      });
      moduleRef = ref;
      expect((await ai.generateText({ prompt: 'Hi', onUsage: async () => Promise.reject(new Error('also down')) })).text).toBe('ok');
    });

    it('reports no cost for a model without pricing', async () => {
      const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel(() => 'ok') } });
      moduleRef = ref;
      const result = await ai.generateText({ prompt: 'Hi' });
      expect(result.cost).toBeUndefined();
      expect(result.steps[0]).not.toHaveProperty('cost');
    });

    describe.each(['maxTokens', 'maxCost'] as const)('budget.%s', (name) => {
      it.each([-1, NaN, Infinity, -Infinity])('rejects invalid limit %s before calling the provider', async (value) => {
        const model = new FakeModel(() => 'ok');
        const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
        moduleRef = ref;
        await expect(ai.generateText({ prompt: 'Hi', budget: { [name]: value } })).rejects.toBeInstanceOf(AiConfigurationError);
        expect(() => ai.streamText({ prompt: 'Hi', budget: { [name]: value } })).toThrow(AiConfigurationError);
        expect(model.calls).toHaveLength(0);
      });

      it('validates module defaults and allows a valid call override', async () => {
        const model = new FakeModel(() => 'ok');
        const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { budget: { [name]: NaN } } });
        moduleRef = ref;
        await expect(ai.generateText({ prompt: 'Hi' })).rejects.toBeInstanceOf(AiConfigurationError);
        expect(model.calls).toHaveLength(0);
        await expect(ai.generateText({ prompt: 'Hi', budget: { maxTokens: 100, maxCost: 0.01 } })).resolves.toMatchObject({ text: 'ok' });
      });

      it('preserves zero as an exhausted budget', async () => {
        const model = new FakeModel(() => 'ok');
        const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
        moduleRef = ref;
        await expect(ai.generateText({ prompt: 'Hi', budget: { [name]: 0 } })).rejects.toBeInstanceOf(AiBudgetExceededError);
        expect(model.calls).toHaveLength(0);
      });
    });

    it('rejects fractional token budgets for structured output before calling the provider', async () => {
      const model = new FakeModel(() => 'ok');
      const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
      moduleRef = ref;
      await expect(ai.generateObject({ prompt: 'Hi', schema: z.object({ text: z.string() }), budget: { maxTokens: 1.5 } })).rejects.toBeInstanceOf(AiConfigurationError);
      expect(model.calls).toHaveLength(0);
    });

    it('stops before a model call over the budget, with what was spent', async () => {
      // A model that keeps calling a tool it was not given: each call is an error it reads, and calls again.
      const model = new FakeModel(() => ({ toolCalls: [{ name: 'missing' }], usage: { inputTokens: 600, outputTokens: 100 } }));
      const { moduleRef: ref, ai } = await createAi({ models: { default: { model, pricing: { input: 1, output: 1 } } } });
      moduleRef = ref;

      const tokens = (await ai.generateText({ prompt: 'Hi', budget: { maxTokens: 1_000 } }).catch((e: unknown) => e)) as AiBudgetExceededError;
      expect(tokens).toBeInstanceOf(AiBudgetExceededError);
      expect(tokens.usage.totalTokens).toBe(1_400);
      expect(model.calls).toHaveLength(2);

      // 700 tokens at 1 per million per step
      const cost = (await ai.generateText({ prompt: 'Hi', budget: { maxCost: 0.002 } }).catch((e: unknown) => e)) as AiBudgetExceededError;
      expect(cost).toBeInstanceOf(AiBudgetExceededError);
      expect(cost.cost).toBeCloseTo(0.0021, 12);
      expect(model.calls).toHaveLength(5);
    });

    it('stops at maxSteps while the model still asks for tools', async () => {
      const model = new FakeModel(() => ({ toolCalls: [{ name: 'missing' }] }));
      const { moduleRef: ref, ai } = await createAi({ models: { default: model }, defaults: { maxSteps: 3 } });
      moduleRef = ref;

      const result = await ai.generateText({ prompt: 'Hi' });
      expect(result.finishReason).toBe('tool-calls');
      expect(result.steps).toHaveLength(3);
      expect(result.usage.totalTokens).toBe(60);
      await expect(ai.generateText({ prompt: 'Hi', maxSteps: 0 })).rejects.toThrow('`maxSteps`');
    });
  });
});
