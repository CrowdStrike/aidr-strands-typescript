/**
 * AIGuardModel — wraps any Strands Model with CrowdStrike AIDR security guardrails.
 *
 * The wrapper intercepts stream() on both sides:
 *
 * - **Input guard** (event_type="input"): runs before the downstream model is called.
 *   Blocked requests throw AIGuardBlockedError; transformed content is substituted.
 * - **Output guard** (event_type="output"): runs after the downstream model responds.
 *   Blocked responses throw AIGuardBlockedError; transformed content is substituted
 *   and a modelRedactionEvent is emitted.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'
 * import { AIGuardModel } from '@crowdstrike/strands-aidr'
 *
 * const downstream = new AnthropicModel({ modelId: 'claude-3-5-sonnet-20241022' })
 * const model = new AIGuardModel(downstream, {}, { token: process.env.CS_AIDR_TOKEN })
 * const agent = new Agent({ model })
 * await agent.invoke('Hello!')
 * ```
 */

import { AIGuard } from '@crowdstrike/aidr'
import type { ChatCompletionsGuard } from '@crowdstrike/aidr'
import {
  type BaseModelConfig,
  Message,
  Model,
  type ModelStreamEvent,
  type StreamOptions,
  TextBlock,
} from '@strands-agents/sdk'
import type { ToolSpec } from '@strands-agents/sdk'
import { AIGuardBlockedError, AIGuardUnavailableError, AIGuardTransformFailedError } from './errors.js'
import { messagesToOpenAI, toolSpecsToOpenAI } from './convert.js'
import type { OpenAIMessage } from './convert.js'

/**
 * AIDR metadata forwarded on every guard_chat_completions call.
 * All fields are optional.
 */
export interface AIGuardModelConfig {
  /** Identifier of the source application / agent. */
  appId?: string
  /** AIDR collector instance identifier. */
  collectorInstanceId?: string
  /** Identifier of the user or entity initiating the AI interaction. */
  userId?: string
  /** Tenant identifier for multi-tenant applications. */
  tenantId?: string
  /** Name of the LLM provider (e.g. "anthropic", "openai"). */
  llmProvider?: string
  /** Name of the AI model (e.g. "claude-3-5-sonnet"). */
  model?: string
  /** Version identifier for the AI model. */
  modelVersion?: string
  /** IP address of the client making the request. */
  sourceIp?: string
  /** Geographic location of the request origin (e.g. "US-CA", "EU-FR"). */
  sourceLocation?: string
  /** Additional metadata for AIDR logging. */
  extraInfo?: Record<string, unknown>
}

/** Parsed result from an AIDR guardChatCompletions response. */
interface AidrResult {
  blocked?: boolean
  transformed?: boolean
  guard_output?: { [key: string]: unknown }
}

/**
 * A Strands Model wrapper that adds CrowdStrike AIDR security guardrails
 * around any downstream model provider.
 */
export class AIGuardModel<T extends BaseModelConfig = BaseModelConfig> extends Model<T> {
  readonly guardInput: boolean
  readonly guardOutput: boolean
  readonly failOpen: boolean

  private readonly downstream: Model<T>
  private readonly aidrConfig: AIGuardModelConfig
  private readonly aidr: AIGuard

  constructor(
    downstream: Model<T>,
    aidrConfig: AIGuardModelConfig = {},
    options: {
      token?: string
      baseURLTemplate?: string
      guardInput?: boolean
      guardOutput?: boolean
      failOpen?: boolean
    } = {},
  ) {
    super()
    const token = options.token ?? process.env['CS_AIDR_TOKEN'] ?? ''
    if (!token) {
      throw new Error('AIDR token is required. Pass token option or set CS_AIDR_TOKEN env var.')
    }
    const baseURLTemplate =
      options.baseURLTemplate ??
      process.env['CS_AIDR_BASE_URL'] ??
      'https://api.crowdstrike.com/aidr/{SERVICE_NAME}'

    this.downstream = downstream
    this.aidrConfig = aidrConfig
    this.guardInput = options.guardInput ?? true
    this.guardOutput = options.guardOutput ?? true
    this.failOpen = options.failOpen ?? false
    this.aidr = new AIGuard({ token, baseURLTemplate })
  }

  // ---------------------------------------------------------------------------
  // Model interface
  // ---------------------------------------------------------------------------

  updateConfig(config: T): void {
    this.downstream.updateConfig(config)
  }

  getConfig(): T {
    return this.downstream.getConfig()
  }

  async *stream(messages: Message[], options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    let effectiveMessages = messages

    // 1. Input guard
    if (this.guardInput) {
      effectiveMessages = await this.runInputGuard(effectiveMessages, options)
    }

    // 2. Buffer all downstream events so we can inspect the complete response
    //    before applying the output guard.
    const bufferedEvents: ModelStreamEvent[] = []
    for await (const event of this.downstream.stream(effectiveMessages, options)) {
      bufferedEvents.push(event)
    }

    // 3. Output guard
    if (this.guardOutput) {
      const { events, replacementText } = await this.runOutputGuard(effectiveMessages, options, bufferedEvents)
      for (const event of events) {
        yield event
      }
      // Emit a modelRedactionEvent so the Strands framework knows output was redacted.
      if (replacementText !== null) {
        yield {
          type: 'modelRedactionEvent',
          outputRedaction: { replaceContent: replacementText },
        }
      }
      return
    }

    // 4. Yield buffered events unchanged when output guard is disabled.
    for (const event of bufferedEvents) {
      yield event
    }
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private buildAidrBody(
    guardInput: { [key: string]: unknown },
    eventType: 'input' | 'output',
  ): ChatCompletionsGuard {
    const cfg = this.aidrConfig
    return {
      guard_input: guardInput,
      event_type: eventType,
      ...(cfg.appId && { app_id: cfg.appId }),
      ...(cfg.collectorInstanceId && { collector_instance_id: cfg.collectorInstanceId }),
      ...(cfg.userId && { user_id: cfg.userId }),
      ...(cfg.tenantId && { tenant_id: cfg.tenantId }),
      ...(cfg.llmProvider && { llm_provider: cfg.llmProvider }),
      ...(cfg.model && { model: cfg.model }),
      ...(cfg.modelVersion && { model_version: cfg.modelVersion }),
      ...(cfg.sourceIp && { source_ip: cfg.sourceIp }),
      ...(cfg.sourceLocation && { source_location: cfg.sourceLocation }),
      ...(cfg.extraInfo && { extra_info: cfg.extraInfo }),
    }
  }

  private async callAidr(eventType: 'input' | 'output', guardInput: { [key: string]: unknown }): Promise<AidrResult | null> {
    const body = this.buildAidrBody(guardInput, eventType)
    const response = await this.aidr.guardChatCompletions(body)
    // MaybeAcceptedResponse<T> has a .result property when the request completes.
    return (response as { result?: AidrResult }).result ?? null
  }

  private async runInputGuard(messages: Message[], options?: StreamOptions): Promise<Message[]> {
    const oaiMessages = messagesToOpenAI(messages, options?.systemPrompt)
    const guardInput: { [key: string]: unknown } = { messages: oaiMessages }
    if (options?.toolSpecs && options.toolSpecs.length > 0) {
      guardInput['tools'] = toolSpecsToOpenAI(options.toolSpecs as ToolSpec[])
    }

    const result = await this.callAidr('input', guardInput)
    if (result === null) {
      if (!this.failOpen) {
        throw new AIGuardUnavailableError('input')
      }
      return messages
    }

    if (result.blocked) {
      throw new AIGuardBlockedError('input', getSummary(result))
    }

    if (result.transformed && result.guard_output) {
      return substituteMessages(messages, result.guard_output)
    }

    if (result.transformed) {
      throw new AIGuardTransformFailedError('input')
    }

    return messages
  }

  private async runOutputGuard(
    inputMessages: Message[],
    options: StreamOptions | undefined,
    bufferedEvents: ModelStreamEvent[],
  ): Promise<{ events: ModelStreamEvent[]; replacementText: string | null }> {
    const assistantMsg = reconstructAssistantMessage(bufferedEvents)
    const conversationOai = messagesToOpenAI(inputMessages, options?.systemPrompt)
    conversationOai.push(assistantMsg)

    const guardInput: { [key: string]: unknown } = { messages: conversationOai }
    if (options?.toolSpecs && options.toolSpecs.length > 0) {
      guardInput['tools'] = toolSpecsToOpenAI(options.toolSpecs as ToolSpec[])
    }

    const result = await this.callAidr('output', guardInput)
    if (result === null) {
      if (!this.failOpen) {
        throw new AIGuardUnavailableError('output')
      }
      return { events: bufferedEvents, replacementText: null }
    }

    if (result.blocked) {
      throw new AIGuardBlockedError('output', getSummary(result))
    }

    if (result.transformed && result.guard_output) {
      const { events, replacementText } = rebuildEventsFromGuardOutput(bufferedEvents, result.guard_output)
      return { events, replacementText }
    }

    if (result.transformed) {
      throw new AIGuardTransformFailedError('output')
    }

    return { events: bufferedEvents, replacementText: null }
  }
}

// ---------------------------------------------------------------------------
// StreamEvent reconstruction helpers
// ---------------------------------------------------------------------------

/**
 * Reconstruct an OpenAI-format assistant message from buffered ModelStreamEvents.
 * Used to build the output-guard AIDR payload.
 */
function reconstructAssistantMessage(events: ModelStreamEvent[]): OpenAIMessage {
  const textParts: string[] = []
  const toolCalls: Array<{
    type: 'function'
    id: string
    function: { name: string; arguments: string }
  }> = []
  let currentToolCall: (typeof toolCalls)[0] | null = null
  const currentInputParts: string[] = []

  for (const event of events) {
    if (event.type === 'modelContentBlockStartEvent') {
      if (event.start?.type === 'toolUseStart') {
        currentToolCall = {
          type: 'function',
          id: event.start.toolUseId,
          function: { name: event.start.name, arguments: '' },
        }
        currentInputParts.length = 0
      } else {
        currentToolCall = null
        currentInputParts.length = 0
      }
    } else if (event.type === 'modelContentBlockDeltaEvent') {
      if (event.delta.type === 'textDelta') {
        textParts.push(event.delta.text)
      } else if (event.delta.type === 'toolUseInputDelta' && currentToolCall !== null) {
        currentInputParts.push(event.delta.input)
      }
    } else if (event.type === 'modelContentBlockStopEvent') {
      if (currentToolCall !== null) {
        currentToolCall.function.arguments = currentInputParts.join('')
        toolCalls.push(currentToolCall)
        currentToolCall = null
        currentInputParts.length = 0
      }
    }
  }

  const msg: OpenAIMessage = { role: 'assistant' }
  if (textParts.length > 0) msg.content = textParts.join('')
  if (toolCalls.length > 0) msg.tool_calls = toolCalls
  return msg
}

/**
 * Rebuild a ModelStreamEvent list with text content replaced by AIDR's guard_output.
 *
 * Non-text events (tool calls, metadata, message start/stop) are preserved.
 * Text content blocks are replaced with the assistant content from guard_output.messages.
 *
 * Returns both the rebuilt event list and the replacement text (for emitting a
 * modelRedactionEvent afterwards).
 */
function rebuildEventsFromGuardOutput(
  originalEvents: ModelStreamEvent[],
  guardOutput: { [key: string]: unknown },
): { events: ModelStreamEvent[]; replacementText: string | null } {
  const messages = (guardOutput['messages'] as Array<{ role?: string; content?: string }>) ?? []
  let replacementText: string | null = null
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m.role === 'assistant' && typeof m.content === 'string' && m.content) {
      replacementText = m.content
      break
    }
  }

  if (replacementText === null) {
    return { events: originalEvents, replacementText: null }
  }

  const rebuilt: ModelStreamEvent[] = []
  let textBlockActive = false
  let textEmitted = false

  for (const event of originalEvents) {
    if (event.type === 'modelContentBlockStartEvent') {
      // Tool-use blocks are not text blocks.
      textBlockActive = event.start?.type !== 'toolUseStart'
      rebuilt.push(event)
    } else if (event.type === 'modelContentBlockDeltaEvent' && textBlockActive) {
      if (event.delta.type === 'textDelta') {
        if (!textEmitted) {
          // Emit replacement as a single delta.
          rebuilt.push({ type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: replacementText } })
          textEmitted = true
        }
        // Skip subsequent original text deltas.
      } else {
        rebuilt.push(event)
      }
    } else if (event.type === 'modelContentBlockStopEvent') {
      textBlockActive = false
      rebuilt.push(event)
    } else {
      rebuilt.push(event)
    }
  }

  return { events: rebuilt, replacementText }
}

// ---------------------------------------------------------------------------
// Private helpers: substitute Strands messages from AIDR guard_output
// ---------------------------------------------------------------------------

/**
 * Return a copy of original with text content replaced from guard_output.
 *
 * The indexing mirrors messagesToOpenAI: all text blocks in a Strands message
 * map to a single OAI message (so they share one oaiIdx), each toolResult
 * block emits an independent OAI message, and messages with only tool-use
 * blocks (no text) also consume one index.
 */
function substituteMessages(
  original: Message[],
  guardOutput: { [key: string]: unknown },
): Message[] {
  const updatedOai = (guardOutput['messages'] as Array<{ role?: string; content?: unknown }>) ?? []
  const contentByIndex = new Map<number, string>()
  updatedOai.forEach((m, idx) => {
    if (typeof m.content === 'string') {
      contentByIndex.set(idx, m.content)
    }
  })

  const result: Message[] = []
  let oaiIdx = 0

  for (const msg of original) {
    // Classify blocks to mirror messagesToOpenAI emission order.
    const hasText = msg.content.some((b) => b.type === 'textBlock')
    const hasToolUse = msg.content.some((b) => b.type === 'toolUseBlock')
    const toolResultBlocks = msg.content.filter((b) => b.type === 'toolResultBlock')

    // messagesToOpenAI emits one OAI message for text+toolUse (if either exists)
    const mainMsgIdx = (hasText || hasToolUse) ? oaiIdx++ : -1

    // Each toolResult emits a separate OAI message
    const toolResultStartIdx = oaiIdx
    oaiIdx += toolResultBlocks.length

    // Rebuild blocks, applying text replacement from the main message index
    const replacement = mainMsgIdx >= 0 ? contentByIndex.get(mainMsgIdx) : undefined
    const newBlocks = msg.content.map((block) => {
      if (block.type === 'textBlock' && replacement !== undefined) {
        // All text blocks in this message were collapsed into one OAI content string.
        // Replace the first text block with the full replacement; clear the rest.
        return block
      }
      return block
    })

    // If there's a replacement, apply it to the text blocks
    if (replacement !== undefined) {
      const textIndices = newBlocks.reduce<number[]>((acc, b, i) => {
        if (b.type === 'textBlock') acc.push(i)
        return acc
      }, [])
      if (textIndices.length > 0) {
        newBlocks[textIndices[0]] = new TextBlock(replacement)
        for (let i = 1; i < textIndices.length; i++) {
          newBlocks[textIndices[i]] = new TextBlock('')
        }
      }
    }

    result.push(new Message({ role: msg.role, content: newBlocks }))
  }
  return result
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function getSummary(result: AidrResult): string {
  // The TypeScript AIDR SDK types don't expose a summary field, so we fall
  // back to policy or a generic message.
  return (result as { summary?: string }).summary ?? (result as { policy?: string }).policy ?? 'blocked'
}
