import {
  FallbackPolicy,
  ResilienceContext,
  ResilienceError,
  ResiliencePolicy,
  ResilienceTimeoutError,
  RetryPolicy,
  TimeoutPolicy,
  type RetryOptions,
} from '@nestjs/resilience';
import { Logger } from '@nestjs/common';
import { APICallError, RetryError } from 'ai';
import { Observable } from 'rxjs';
import {
  AiBudgetExceededError,
  AiConfigurationError,
  AiModelError,
  AiOutputError,
  type AiAttemptFailure,
} from '../errors/ai.errors.js';
import type { AiResilienceSettings } from '../interfaces/ai-module-options.interface.js';
import type { ResolvedModel } from '../models/ai-models.service.js';
import type { AiModelInfo } from '../interfaces/ai-result.interface.js';
import { toMilliseconds } from '../utils/duration.util.js';

/** The identity the policy layer needs; generation and embedding registries stay separately typed. */
export interface PolicyModel {
  info: AiModelInfo;
}

/** What one attempt of a model call receives. */
export interface StepAttempt<M extends PolicyModel = ResolvedModel> {
  model: M;
  /** 1-based, per model. */
  attempt: number;
  fallback: boolean;
  /** Aborted by the caller, or by the attempt's timeout (reason: `ResilienceTimeoutError`). */
  signal: AbortSignal;
  /** Commits the attempt: no timeout, retry or fallback from now on. A step whose tools start calls it. */
  commit(): void;
  /** Sends a stream chunk to the consumer, which commits the attempt too. */
  emit(chunk: unknown): void;
}

/** How a call's models are retried, timed out and replaced. Internal. */
export interface PolicyPlan<M extends PolicyModel = ResolvedModel> {
  chain: M[];
  /** The caller's signal. */
  signal: AbortSignal;
  /** Labels the policies' events (`source`). */
  source: string;
  /** The model's own policy: an application's, or the one built from its `retry` and `timeout`. */
  policyFor(model: M): ResiliencePolicy | undefined;
  fallbackIf: (error: unknown) => boolean;
}

type StepEvent<T, M extends PolicyModel> = { type: 'commit' } | { type: 'chunk'; chunk: unknown } | { type: 'outcome'; value: T; attempt: StepAttempt<M> };

const logger = new Logger('AiModule');

/** `ai`'s argument errors, which no retry or other model fixes. */
const CALLER_MISTAKES = new Set(['AI_InvalidPromptError', 'AI_InvalidArgumentError', 'AI_InvalidMessageRoleError', 'AI_MessageConversionError']);

/**
 * Runs one model call (a step) under the call's policies: the model's own (retry, timeout, or the application's),
 * then each fallback model's, through `executeObservable()`, so that the outcome is committed once a chunk reaches
 * the consumer or the step's tools start. Chunks go to `emit` as they come. When the call fails: `AiModelError` with
 * every attempt, the caller's abort reason, or what is not a model failure (an `AiOutputError`, a circuit breaker's
 * rejection) as it is.
 */
export function runStep<T, M extends PolicyModel = ResolvedModel>(
  plan: PolicyPlan<M>,
  run: (attempt: StepAttempt<M>) => Promise<T>,
  emit?: (chunk: unknown) => void,
): Promise<{ value: T; attempt: StepAttempt<M>; attempts: number }> {
  const failures: AiAttemptFailure[] = [];
  let attempts = 0;

  const attemptsOf = (index: number) => (context: { signal: AbortSignal; attempt: number }) =>
    new Observable<StepEvent<T, M>>((subscriber) => {
      attempts++;
      const model = plan.chain[index]!;
      let settled = false;
      let committed = false;
      const record = (error: unknown) => {
        if (!settled) {
          settled = true;
          // An attempt aborted for its timeout failed for the timeout, whatever the request threw.
          const reason = context.signal.aborted ? context.signal.reason : undefined;
          failures.push({ model: model.info, attempt: context.attempt, error: reason instanceof ResilienceTimeoutError ? reason : unwrap(error) });
        }
      };
      const attempt: StepAttempt<M> = {
        model,
        attempt: context.attempt,
        fallback: index > 0,
        signal: context.signal,
        commit: () => {
          if (!committed && !subscriber.closed) {
            committed = true;
            subscriber.next({ type: 'commit' });
          }
        },
        emit: (chunk) => {
          committed = true;
          subscriber.next({ type: 'chunk', chunk });
        },
      };
      run(attempt).then(
        (value) => {
          settled = true;
          subscriber.next({ type: 'outcome', value, attempt });
          subscriber.complete();
        },
        (error: unknown) => {
          record(error);
          subscriber.error(unwrap(error));
        },
      );
      // Unsubscribed before it settled: its timeout ran out, or the caller went away.
      return () => record(context.signal.reason);
    });

  const policyAt = (index: number): ResiliencePolicy => {
    const model = plan.chain[index]!;
    const own = plan.policyFor(model);
    const policies: ResiliencePolicy[] = [];
    if (index < plan.chain.length - 1) {
      policies.push(
        new FallbackPolicy<StepEvent<T, M>>(
          (error, context) => {
            // A fallback that works must not hide why the model failed: a revoked key, a refused request.
            logger.warn(`The model "${labelOf(model)}" failed (${describeError(error)}); falling back to "${labelOf(plan.chain[index + 1]!)}".`);
            return policyAt(index + 1).executeObservable(attemptsOf(index + 1), { signal: context.signal, source: plan.source });
          },
          { name: `${nameOf(model)} fallback`, handleIf: (error) => isFallbackable(error) && plan.fallbackIf(error) },
        ),
      );
    }
    if (own) {
      policies.push(own);
    }
    return ResiliencePolicy.wrap(...policies);
  };

  // Inside a resilient entrypoint (or `ResilienceContext.run()`), its attempt's signal stops the call too.
  const ambient = new ResilienceContext().signal;
  const signal = ambient ? AbortSignal.any([plan.signal, ambient]) : plan.signal;

  return new Promise((resolve, reject) => {
    let outcome: { value: T; attempt: StepAttempt<M> } | undefined;
    policyAt(0)
      .executeObservable(attemptsOf(0), { signal, source: plan.source })
      .subscribe({
        next: (event) => {
          if (event.type === 'chunk') {
            emit?.(event.chunk);
          } else if (event.type === 'outcome') {
            outcome = { value: event.value, attempt: event.attempt };
          }
        },
        error: (error: unknown) => reject(finalError(error, failures, signal)),
        complete: () => (outcome ? resolve({ ...outcome, attempts }) : reject(finalError(new Error('The model call ended without an answer'), failures, signal))),
      });
  });
}

/**
 * The model's own policy from its settings: `RetryPolicy` (honoring the provider's `retry-after`) around
 * `TimeoutPolicy`, either one left out when it is `false`.
 */
export function modelPolicy(
  model: PolicyModel,
  { retry, timeout }: { retry: RetryOptions | false | undefined; timeout: number | string | false | undefined },
  namespace = 'ai',
): ResiliencePolicy | undefined {
  const policies: ResiliencePolicy[] = [];
  if (retry !== false) {
    policies.push(retryPolicy(model, retry ?? {}, namespace));
  }
  if (timeout !== false) {
    policies.push(new TimeoutPolicy({ timeout: toMilliseconds(timeout ?? DEFAULT_TIMEOUT, 'timeout'), name: `${nameOf(model, namespace)} timeout` }));
  }
  return policies.length > 0 ? ResiliencePolicy.wrap(...policies) : undefined;
}

/** The levels a setting can come from, highest priority first. */
export type PolicyLevels = ReadonlyArray<{ name: string; settings: AiResilienceSettings }>;

const isSet = (value: unknown) => value !== undefined && value !== null;

/**
 * Rejects a `retry` or `timeout` set at the level of the custom policy that applies, or above it: the policy replaces
 * the ones built from them, so they would be ignored without a word.
 */
export function assertPolicyLevels(where: string, levels: PolicyLevels): void {
  const selected = levels.findIndex(({ settings }) => isSet(settings.resilience));
  if (selected === -1) {
    return;
  }
  const conflicts = levels
    .slice(0, selected + 1)
    .flatMap(({ name, settings }) =>
      (['retry', 'timeout'] as const).filter((key) => isSet(settings[key])).map((key) => `\`${key}\` at ${name}`),
    );
  if (conflicts.length > 0) {
    throw new AiConfigurationError(
      `${where} cannot use ${conflicts.join(' and ')} with \`resilience\` at ${levels[selected]!.name}. ` +
        'Remove the listed `retry`/`timeout` settings or remove the selected `resilience` policy. ' +
        'A custom policy cannot replace settings at the same or a higher-priority level.',
    );
  }
}

/** The policy a model's settings select: a custom one, or else the one built from `retry` and `timeout`. */
export function resolveModelPolicy(
  model: PolicyModel,
  levels: PolicyLevels,
  preset: (name: string) => ResiliencePolicy,
  namespace = 'ai',
): ResiliencePolicy | undefined {
  const pick = <K extends keyof AiResilienceSettings>(key: K) =>
    levels.find(({ settings }) => isSet(settings[key]))?.settings[key];
  const resilience = pick('resilience');
  if (resilience === undefined) {
    return modelPolicy(model, { retry: pick('retry'), timeout: pick('timeout') }, namespace);
  }
  return typeof resilience === 'string' ? preset(resilience) : resilience;
}

const DEFAULT_TIMEOUT = '60s';
const DEFAULT_BACKOFF = { delay: 1_000, factor: 2, maxDelay: 30_000 };

function retryPolicy(model: PolicyModel, retry: RetryOptions, namespace: string): RetryPolicy {
  const backoff = retry.backoff;
  const maxDelay =
    typeof backoff === 'object' && backoff.maxDelay !== undefined ? toMilliseconds(backoff.maxDelay, 'retry.backoff.maxDelay') : DEFAULT_BACKOFF.maxDelay;
  const retryIf = retry.retryIf ?? isRetryable;
  const configuredBackoff = (attempt: number, error: unknown) =>
    typeof backoff === 'function'
      ? backoff(attempt, error)
      : exponential(attempt, { ...DEFAULT_BACKOFF, ...backoff });
  return new RetryPolicy({
    name: `${nameOf(model, namespace)} retry`,
    attempts: retry.attempts ?? 3,
    ...(retry.retryOnResult && { retryOnResult: retry.retryOnResult }),
    // A provider that asks for a longer wait than the backoff allows: on to the fallback.
    retryIf: (error, attempt) =>
      !(error instanceof AiOutputError) && (retryAfter(error) ?? 0) <= maxDelay && retryIf(error, attempt),
    backoff: (attempt, error) => retryAfter(error) ?? configuredBackoff(attempt, error),
  });
}

/** What is retried by default: what the provider marks retryable (with network failures), and timeouts. */
export function isRetryable(error: unknown): boolean {
  if (error instanceof ResilienceTimeoutError) {
    return true;
  }
  return APICallError.isInstance(error) && error.isRetryable;
}

function isFallbackable(error: unknown): boolean {
  return !(error instanceof AiOutputError) && !(error instanceof Error && CALLER_MISTAKES.has(error.name));
}

function finalError(error: unknown, failures: AiAttemptFailure[], signal: AbortSignal): unknown {
  if (signal.aborted) {
    return signal.reason;
  }
  if (error instanceof AiOutputError || error instanceof AiBudgetExceededError || error instanceof AiConfigurationError) {
    return error;
  }
  // A circuit breaker's, bulkhead's or rate limit's rejection: the application's policy said no.
  if (error instanceof ResilienceError && !(error instanceof ResilienceTimeoutError)) {
    return error;
  }
  return new AiModelError(failures.length > 0 ? failures : [{ model: { provider: 'unknown', modelId: 'unknown' }, attempt: 1, error }]);
}

/** `ai` wraps errors it retried itself; with `maxRetries: 0` it should not, but the last one is what failed. */
function unwrap(error: unknown): unknown {
  return RetryError.isInstance(error) ? (error.lastError ?? error) : error;
}

function exponential(
  attempt: number,
  { delay, factor = 2, maxDelay, jitter = 'full' }: { delay?: number | string; factor?: number; maxDelay?: number | string; jitter?: 'full' | 'equal' | 'none' },
): number {
  const max = toMilliseconds(maxDelay ?? DEFAULT_BACKOFF.maxDelay, 'retry.backoff.maxDelay');
  const computed = Math.min(max, toMilliseconds(delay ?? DEFAULT_BACKOFF.delay, 'retry.backoff.delay') * Math.max(1, factor) ** (attempt - 1));
  switch (jitter) {
    case 'none':
      return computed;
    case 'equal':
      return computed / 2 + Math.random() * (computed / 2);
    default:
      return Math.random() * computed;
  }
}

/** The provider's `retry-after-ms` or `retry-after`, in milliseconds. */
function retryAfter(error: unknown): number | undefined {
  if (!APICallError.isInstance(error) || !error.responseHeaders) {
    return undefined;
  }
  const headers = Object.fromEntries(Object.entries(error.responseHeaders).map(([name, value]) => [name.toLowerCase(), value]));
  const ms = Number.parseFloat(headers['retry-after-ms'] ?? '');
  if (Number.isFinite(ms) && ms >= 0) {
    return ms;
  }
  const value = headers['retry-after'];
  if (value === undefined) {
    return undefined;
  }
  const seconds = Number.parseFloat(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1_000;
  }
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}

function nameOf(model: PolicyModel, namespace = 'ai'): string {
  return `${namespace}:${labelOf(model)}`;
}

export function labelOf(model: PolicyModel): string {
  return model.info.name ?? `${model.info.provider}:${model.info.modelId}`;
}

/** The error's class, its HTTP status if it has one, and its message: `APICallError, 401: Invalid API key`. */
function describeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }
  const { statusCode, status } = error as { statusCode?: unknown; status?: unknown };
  const code = typeof statusCode === 'number' ? statusCode : typeof status === 'number' ? status : undefined;
  return `${error.constructor.name}${code !== undefined ? `, ${code}` : ''}: ${error.message}`;
}
