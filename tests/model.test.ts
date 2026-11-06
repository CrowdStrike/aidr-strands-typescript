/**
 * Unit tests for AIGuardModel.
 *
 * All AIDR calls and downstream model interactions are mocked — no live
 * services are required.
 */

import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest'
import { AIGuard } from '@crowdstrike/aidr'
import {
  type BaseModelConfig,
  Message,
  Model,
  type ModelStreamEvent,
  type StreamOptions,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { AIGuardBlockedError } from '../src/errors.js'
import { AIGuardModel } from '../src/model.js'
import { messagesToOpenAI, toolSpecsToOpenAI } from '../src/convert.js'

// ---------------------------------------------------------------------------
// Mock the AIDR SDK
// ---------------------------------------------------------------------------

vi.mock('@crowdstrike/aidr', () => {
  const mockGuardChatCompletions = vi.fn()
  const MockAIGuard = vi.fn().mockImplementation(function () {
    return { guardChatCompletions: mockGuardChatCompletions }
  })
  return { AIGuard: MockAIGuard }
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a minimal Strands Message. */
function userMsg(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function assistantMsg(text: string): Message {
  return new Message({ role: 'assistant', content: [new TextBlock(text)] })
}

/** Build a simple ModelStreamEvent sequence for a text response. */
function makeTextEvents(text: string): ModelStreamEvent[] {
  return [
    { type: 'modelMessageStartEvent', role: 'assistant' },
    { type: 'modelContentBlockStartEvent' },
    { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text } },
    { type: 'modelContentBlockStopEvent' },
    { type: 'modelMessageStopEvent', stopReason: 'endTurn' },
  ]
}

/** AIDR pass-through result. */
const passThroughResult = { result: { blocked: false, transformed: false } }

/** AIDR blocked result. */
function blockedResult(summary = 'blocked by policy') {
  return { result: { blocked: true, transformed: false, summary } }
}

/** AIDR transformed result. */
function transformedResult(replacementText: string) {
  return {
    result: {
      blocked: false,
      transformed: true,
      guard_output: {
        messages: [{ role: 'assistant', content: replacementText }],
      },
    },
  }
}

/** A mock downstream Model that yields the given events. */
class MockModel extends Model<BaseModelConfig> {
  updateConfig = vi.fn()
  getConfig = vi.fn().mockReturnValue({ modelId: 'mock-model' })
  private _events: ModelStreamEvent[]

  constructor(events: ModelStreamEvent[] = makeTextEvents('hello')) {
    super()
    this._events = events
  }

  async *stream(_messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    for (const event of this._events) {
      yield event
    }
  }
}

/** Collect all events from an async iterable. */
async function collectEvents(iterable: AsyncIterable<ModelStreamEvent>): Promise<ModelStreamEvent[]> {
  const events: ModelStreamEvent[] = []
  for await (const event of iterable) {
    events.push(event)
  }
  return events
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

let mockGuardChatCompletions: Mock
let model: AIGuardModel
let downstream: MockModel

beforeEach(() => {
  vi.clearAllMocks()
  // Access the mock function on the AIGuard instance
  mockGuardChatCompletions = (AIGuard as unknown as Mock).mock.results[0]?.value?.guardChatCompletions ?? vi.fn()
  downstream = new MockModel()
  // Re-create model after clearing mocks to get fresh AIGuard mock
  model = new AIGuardModel(downstream, {}, { token: 'test-token', guardInput: true, guardOutput: true })
  // Now get the mock from the freshly created instance
  const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
  mockGuardChatCompletions = aidrInstance.guardChatCompletions
})

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe('constructor', () => {
  it('throws when no token is provided', () => {
    const savedEnv = process.env['CS_AIDR_TOKEN']
    delete process.env['CS_AIDR_TOKEN']
    expect(() => new AIGuardModel(downstream, {}, {})).toThrow('AIDR token is required')
    if (savedEnv !== undefined) process.env['CS_AIDR_TOKEN'] = savedEnv
  })

  it('accepts token from env var', () => {
    process.env['CS_AIDR_TOKEN'] = 'env-token'
    expect(() => new AIGuardModel(downstream)).not.toThrow()
    delete process.env['CS_AIDR_TOKEN']
  })

  it('defaults guardInput and guardOutput to true', () => {
    const m = new AIGuardModel(downstream, {}, { token: 'tok' })
    expect(m.guardInput).toBe(true)
    expect(m.guardOutput).toBe(true)
  })

  it('accepts guardInput=false and guardOutput=false', () => {
    const m = new AIGuardModel(downstream, {}, { token: 'tok', guardInput: false, guardOutput: false })
    expect(m.guardInput).toBe(false)
    expect(m.guardOutput).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// Config delegation
// ---------------------------------------------------------------------------

describe('getConfig / updateConfig', () => {
  it('delegates getConfig to downstream', () => {
    expect(model.getConfig()).toEqual({ modelId: 'mock-model' })
  })

  it('delegates updateConfig to downstream', () => {
    model.updateConfig({ modelId: 'new-model' })
    expect(downstream.updateConfig).toHaveBeenCalledWith({ modelId: 'new-model' })
  })
})

// ---------------------------------------------------------------------------
// Pass-through (no block / transform)
// ---------------------------------------------------------------------------

describe('stream — pass-through', () => {
  it('yields all downstream events when both guards pass', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)
    const events = await collectEvents(model.stream([userMsg('hello')]))

    // No redaction event expected
    expect(events.filter((e) => e.type === 'modelRedactionEvent')).toHaveLength(0)
    // Downstream events passed through
    const textDeltas = events.filter((e) => e.type === 'modelContentBlockDeltaEvent')
    expect(textDeltas).toHaveLength(1)
    expect(mockGuardChatCompletions).toHaveBeenCalledTimes(2) // input + output
  })

  it('calls input guard with converted messages', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)
    await collectEvents(model.stream([userMsg('hello')]))

    const inputCall = mockGuardChatCompletions.mock.calls[0][0]
    expect(inputCall.event_type).toBe('input')
    expect(inputCall.guard_input.messages).toEqual([{ role: 'user', content: 'hello' }])
  })

  it('calls output guard with conversation + assistant message', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)
    await collectEvents(model.stream([userMsg('hello')]))

    const outputCall = mockGuardChatCompletions.mock.calls[1][0]
    expect(outputCall.event_type).toBe('output')
    // Should include user message + assistant response
    expect(outputCall.guard_input.messages.length).toBeGreaterThanOrEqual(2)
    const assistantMsgInCall = outputCall.guard_input.messages.find(
      (m: { role: string }) => m.role === 'assistant',
    )
    expect(assistantMsgInCall?.content).toBe('hello')
  })
})

// ---------------------------------------------------------------------------
// Input blocked
// ---------------------------------------------------------------------------

describe('stream — input blocked', () => {
  it('throws AIGuardBlockedError and does not call downstream', async () => {
    mockGuardChatCompletions.mockResolvedValue(blockedResult('malicious prompt'))

    let downstreamCalled = false
    const customDownstream = new MockModel()
    const origStream = customDownstream.stream.bind(customDownstream)
    customDownstream.stream = async function* (msgs, opts) {
      downstreamCalled = true
      yield* origStream(msgs, opts)
    }

    const m = new AIGuardModel(customDownstream, {}, { token: 'test-token' })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
    aidrInstance.guardChatCompletions.mockResolvedValue(blockedResult('malicious prompt'))

    await expect(collectEvents(m.stream([userMsg('hack the world')]))).rejects.toThrow(AIGuardBlockedError)
    expect(downstreamCalled).toBe(false)
  })

  it('AIGuardBlockedError has eventType=input and correct summary', async () => {
    mockGuardChatCompletions.mockResolvedValue(blockedResult('jailbreak attempt'))
    try {
      await collectEvents(model.stream([userMsg('bad input')]))
      expect.fail('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(AIGuardBlockedError)
      const guardErr = err as AIGuardBlockedError
      expect(guardErr.eventType).toBe('input')
      expect(guardErr.summary).toBe('jailbreak attempt')
    }
  })
})

// ---------------------------------------------------------------------------
// Output blocked
// ---------------------------------------------------------------------------

describe('stream — output blocked', () => {
  it('throws AIGuardBlockedError after downstream runs', async () => {
    // input passes, output blocks
    mockGuardChatCompletions
      .mockResolvedValueOnce(passThroughResult)
      .mockResolvedValueOnce(blockedResult('harmful output'))

    await expect(collectEvents(model.stream([userMsg('hello')]))).rejects.toThrow(AIGuardBlockedError)
    expect(mockGuardChatCompletions).toHaveBeenCalledTimes(2)
  })

  it('AIGuardBlockedError has eventType=output', async () => {
    mockGuardChatCompletions
      .mockResolvedValueOnce(passThroughResult)
      .mockResolvedValueOnce(blockedResult('pii detected'))
    try {
      await collectEvents(model.stream([userMsg('hello')]))
      expect.fail('should have thrown')
    } catch (err) {
      expect(err).toBeInstanceOf(AIGuardBlockedError)
      expect((err as AIGuardBlockedError).eventType).toBe('output')
    }
  })
})

// ---------------------------------------------------------------------------
// Output transformed
// ---------------------------------------------------------------------------

describe('stream — output transformed', () => {
  it('replaces text delta with replacement text', async () => {
    mockGuardChatCompletions
      .mockResolvedValueOnce(passThroughResult)
      .mockResolvedValueOnce(transformedResult('[REDACTED]'))

    const events = await collectEvents(model.stream([userMsg('hello')]))
    const textDeltas = events
      .filter((e) => e.type === 'modelContentBlockDeltaEvent')
      .map((e) => {
        if (e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'textDelta') return e.delta.text
        return null
      })
      .filter(Boolean)

    expect(textDeltas).toEqual(['[REDACTED]'])
  })

  it('emits modelRedactionEvent with outputRedaction after transform', async () => {
    mockGuardChatCompletions
      .mockResolvedValueOnce(passThroughResult)
      .mockResolvedValueOnce(transformedResult('[REDACTED]'))

    const events = await collectEvents(model.stream([userMsg('hello')]))
    const redactionEvents = events.filter((e) => e.type === 'modelRedactionEvent')
    expect(redactionEvents).toHaveLength(1)
    const re = redactionEvents[0]
    if (re.type === 'modelRedactionEvent') {
      expect(re.outputRedaction?.replaceContent).toBe('[REDACTED]')
    }
  })

  it('does not emit modelRedactionEvent on pass-through', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)
    const events = await collectEvents(model.stream([userMsg('hello')]))
    expect(events.filter((e) => e.type === 'modelRedactionEvent')).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Input transformed
// ---------------------------------------------------------------------------

describe('stream — input transformed', () => {
  it('substitutes messages before calling downstream', async () => {
    const inputTransform = {
      result: {
        blocked: false,
        transformed: true,
        guard_output: {
          messages: [{ role: 'user', content: '[REDACTED INPUT]' }],
        },
      },
    }
    mockGuardChatCompletions
      .mockResolvedValueOnce(inputTransform)
      .mockResolvedValueOnce(passThroughResult)

    let capturedMessages: Message[] | null = null
    const customDownstream = new MockModel()
    customDownstream.stream = async function* (msgs, opts) {
      capturedMessages = msgs
      for (const event of makeTextEvents('response')) yield event
    }

    const m = new AIGuardModel(customDownstream, {}, { token: 'test-token' })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
    aidrInstance.guardChatCompletions.mockResolvedValueOnce(inputTransform).mockResolvedValueOnce(passThroughResult)

    await collectEvents(m.stream([userMsg('original message')]))

    expect(capturedMessages).not.toBeNull()
    const firstBlock = capturedMessages![0].content[0]
    expect(firstBlock.type).toBe('textBlock')
    if (firstBlock.type === 'textBlock') {
      expect(firstBlock.text).toBe('[REDACTED INPUT]')
    }
  })
})

// ---------------------------------------------------------------------------
// Guard flag overrides
// ---------------------------------------------------------------------------

describe('guardInput=false', () => {
  it('skips input AIDR call', async () => {
    const m = new AIGuardModel(downstream, {}, { token: 'test-token', guardInput: false, guardOutput: true })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
    aidrInstance.guardChatCompletions.mockResolvedValue(passThroughResult)

    await collectEvents(m.stream([userMsg('hello')]))

    // Only one call: output guard
    expect(aidrInstance.guardChatCompletions).toHaveBeenCalledTimes(1)
    expect(aidrInstance.guardChatCompletions.mock.calls[0][0].event_type).toBe('output')
  })
})

describe('guardOutput=false', () => {
  it('skips output AIDR call', async () => {
    const m = new AIGuardModel(downstream, {}, { token: 'test-token', guardInput: true, guardOutput: false })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
    aidrInstance.guardChatCompletions.mockResolvedValue(passThroughResult)

    await collectEvents(m.stream([userMsg('hello')]))

    // Only one call: input guard
    expect(aidrInstance.guardChatCompletions).toHaveBeenCalledTimes(1)
    expect(aidrInstance.guardChatCompletions.mock.calls[0][0].event_type).toBe('input')
  })
})

describe('guardInput=false guardOutput=false', () => {
  it('makes no AIDR calls', async () => {
    const m = new AIGuardModel(downstream, {}, { token: 'test-token', guardInput: false, guardOutput: false })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value

    await collectEvents(m.stream([userMsg('hello')]))
    expect(aidrInstance.guardChatCompletions).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// Tool specs forwarding
// ---------------------------------------------------------------------------

describe('tool specs forwarding', () => {
  it('forwards tool specs to AIDR input call', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)

    await collectEvents(
      model.stream([userMsg('use a tool')], {
        toolSpecs: [{ name: 'search', description: 'search the web', inputSchema: { type: 'object', properties: {} } }],
      }),
    )

    const inputCall = mockGuardChatCompletions.mock.calls[0][0]
    expect(inputCall.guard_input.tools).toBeDefined()
    expect(inputCall.guard_input.tools[0].function.name).toBe('search')
  })

  it('forwards tool specs to AIDR output call', async () => {
    mockGuardChatCompletions.mockResolvedValue(passThroughResult)

    await collectEvents(
      model.stream([userMsg('use a tool')], {
        toolSpecs: [{ name: 'search', description: 'search the web' }],
      }),
    )

    const outputCall = mockGuardChatCompletions.mock.calls[1][0]
    expect(outputCall.guard_input.tools).toBeDefined()
    expect(outputCall.guard_input.tools[0].function.name).toBe('search')
  })
})

// ---------------------------------------------------------------------------
// AIGuardModelConfig forwarded to AIDR
// ---------------------------------------------------------------------------

describe('AIGuardModelConfig fields forwarded', () => {
  it('sends configured metadata on each AIDR call', async () => {
    const m = new AIGuardModel(
      downstream,
      {
        appId: 'my-app',
        userId: 'user-123',
        tenantId: 'tenant-abc',
        llmProvider: 'anthropic',
        model: 'claude-3-5-sonnet',
      },
      { token: 'test-token' },
    )
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value
    aidrInstance.guardChatCompletions.mockResolvedValue(passThroughResult)

    await collectEvents(m.stream([userMsg('hello')]))

    const call = aidrInstance.guardChatCompletions.mock.calls[0][0]
    expect(call.app_id).toBe('my-app')
    expect(call.user_id).toBe('user-123')
    expect(call.tenant_id).toBe('tenant-abc')
    expect(call.llm_provider).toBe('anthropic')
    expect(call.model).toBe('claude-3-5-sonnet')
  })
})

// ---------------------------------------------------------------------------
// AIDR returns null / no result (null-safety)
// ---------------------------------------------------------------------------

describe('null result handling', () => {
  it('continues when input guard returns no result', async () => {
    mockGuardChatCompletions
      .mockResolvedValueOnce({ result: null })
      .mockResolvedValueOnce(passThroughResult)

    const events = await collectEvents(model.stream([userMsg('hello')]))
    expect(events.some((e) => e.type === 'modelMessageStopEvent')).toBe(true)
  })

  it('continues when output guard returns no result', async () => {
    mockGuardChatCompletions
      .mockResolvedValueOnce(passThroughResult)
      .mockResolvedValueOnce({ result: null })

    const events = await collectEvents(model.stream([userMsg('hello')]))
    expect(events.some((e) => e.type === 'modelMessageStopEvent')).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// messagesToOpenAI unit tests
// ---------------------------------------------------------------------------

describe('messagesToOpenAI', () => {
  it('converts a simple text message', () => {
    const result = messagesToOpenAI([userMsg('hello')])
    expect(result).toEqual([{ role: 'user', content: 'hello' }])
  })

  it('includes a string system prompt', () => {
    const result = messagesToOpenAI([userMsg('hi')], 'You are helpful')
    expect(result[0]).toEqual({ role: 'system', content: 'You are helpful' })
  })

  it('includes a SystemContentBlock array system prompt', () => {
    const result = messagesToOpenAI([userMsg('hi')], [new TextBlock('Be helpful'), new TextBlock('Be concise')])
    expect(result[0]).toEqual({ role: 'system', content: 'Be helpful Be concise' })
  })

  it('converts tool use block to tool_calls', () => {
    const msg = new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ name: 'search', toolUseId: 'call-1', input: { query: 'test' } })],
    })
    const result = messagesToOpenAI([msg])
    expect(result[0].tool_calls).toBeDefined()
    expect(result[0].tool_calls![0].function.name).toBe('search')
    expect(JSON.parse(result[0].tool_calls![0].function.arguments)).toEqual({ query: 'test' })
  })

  it('converts tool result block', () => {
    const msg = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'call-1',
          status: 'success',
          content: [new TextBlock('search result')],
        }),
      ],
    })
    const result = messagesToOpenAI([msg])
    expect(result[0].role).toBe('tool')
    expect(result[0].tool_call_id).toBe('call-1')
    expect(result[0].content).toBe('search result')
  })

  it('emits main message before tool results', () => {
    const msg = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({ name: 'search', toolUseId: 'call-1', input: {} }),
        new ToolResultBlock({ toolUseId: 'call-1', status: 'success', content: [new TextBlock('result')] }),
      ],
    })
    const result = messagesToOpenAI([msg])
    expect(result[0].role).toBe('assistant')
    expect(result[1].role).toBe('tool')
  })

  it('omits messages with no text or tool_calls', () => {
    // A message that only has a ToolResultBlock should produce only tool messages
    const msg = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'call-1',
          status: 'success',
          content: [new TextBlock('result')],
        }),
      ],
    })
    const result = messagesToOpenAI([msg])
    // Only the tool result message, no empty user message
    expect(result.every((m) => m.role !== 'user')).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// toolSpecsToOpenAI unit tests
// ---------------------------------------------------------------------------

describe('toolSpecsToOpenAI', () => {
  it('converts tool spec to OpenAI function format', () => {
    const result = toolSpecsToOpenAI([{ name: 'search', description: 'search the web', inputSchema: { type: 'object' } }])
    expect(result).toEqual([
      {
        type: 'function',
        function: {
          name: 'search',
          description: 'search the web',
          parameters: { type: 'object' },
        },
      },
    ])
  })

  it('defaults parameters to empty object when inputSchema is missing', () => {
    const result = toolSpecsToOpenAI([{ name: 'noop', description: 'does nothing' }])
    expect(result[0].function.parameters).toEqual({})
  })
})

// ---------------------------------------------------------------------------
// Guard timeout
// ---------------------------------------------------------------------------

describe('guard timeout', () => {
  it('defaults guardTimeoutMs to 30000', () => {
    const m = new AIGuardModel(downstream, {}, { token: 'test-token' })
    expect(m.guardTimeoutMs).toBe(30_000)
  })

  it('accepts custom guardTimeoutMs', () => {
    const m = new AIGuardModel(downstream, {}, { token: 'test-token', guardTimeoutMs: 5_000 })
    expect(m.guardTimeoutMs).toBe(5_000)
  })

  it('rejects with AbortError when guard hangs past timeout', async () => {
    vi.useFakeTimers()

    const m = new AIGuardModel(downstream, {}, { token: 'test-token', guardTimeoutMs: 100 })
    const aidrInstance = (AIGuard as unknown as Mock).mock.results.at(-1)?.value

    // Simulate a guard call that never resolves
    aidrInstance.guardChatCompletions.mockImplementation(
      () => new Promise<never>(() => {}),
    )

    const streamPromise = collectEvents(m.stream([userMsg('hello')]))

    // Advance timers past the timeout
    await vi.advanceTimersByTimeAsync(200)

    await expect(streamPromise).rejects.toThrow()

    vi.useRealTimers()
  })
})
