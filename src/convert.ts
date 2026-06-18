/**
 * Conversion helpers: Strands SDK types → OpenAI-format dicts for AIDR.
 *
 * AIDR's guard_chat_completions endpoint expects OpenAI-format messages and
 * tool specs.  These helpers perform that conversion from Strands' native
 * Message / ToolSpec types.
 */

import type { Message, SystemPrompt, TextBlock } from '@strands-agents/sdk'
import type { ToolSpec } from '@strands-agents/sdk'

/**
 * OpenAI-format message as expected by AIDR's guard_input.messages.
 */
export interface OpenAIMessage {
  role: string
  content?: string
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
}

export interface OpenAIToolCall {
  type: 'function'
  id: string
  function: {
    name: string
    arguments: string
  }
}

export interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: unknown
  }
}

/**
 * Convert Strands Message[] (plus optional system prompt) to OpenAI-format
 * messages for AIDR's guard_input payload.
 *
 * Only text, tool-use, and tool-result blocks are included.  Rich media
 * (images, video, documents) is silently dropped because AIDR does not
 * process binary content.
 *
 * When systemPrompt is a string, it is emitted as a {role: "system"} message.
 * When systemPrompt is an array of SystemContentBlocks, TextBlock text is
 * concatenated (CachePointBlocks and GuardContentBlocks are skipped).
 */
export function messagesToOpenAI(messages: Message[], systemPrompt?: SystemPrompt): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  // Emit system prompt first.
  if (systemPrompt) {
    let systemText: string
    if (typeof systemPrompt === 'string') {
      systemText = systemPrompt
    } else {
      systemText = systemPrompt
        .filter((b) => b.type === 'textBlock')
        .map((b) => (b as TextBlock).text)
        .join(' ')
    }
    if (systemText) {
      result.push({ role: 'system', content: systemText })
    }
  }

  for (const msg of messages) {
    const textParts: string[] = []
    const toolCalls: OpenAIToolCall[] = []
    const toolResults: OpenAIMessage[] = []

    for (const block of msg.content) {
      if (block.type === 'textBlock') {
        textParts.push(block.text)
      } else if (block.type === 'toolUseBlock') {
        toolCalls.push({
          type: 'function',
          id: block.toolUseId,
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          },
        })
      } else if (block.type === 'toolResultBlock') {
        let contentStr = ''
        for (const item of block.content) {
          if (item.type === 'textBlock') {
            contentStr += item.text
          } else if (item.type === 'jsonBlock') {
            contentStr += JSON.stringify(item.json)
          }
        }
        toolResults.push({
          role: 'tool',
          tool_call_id: block.toolUseId,
          content: contentStr,
        })
      }
    }

    // Emit the main message before tool results to preserve conversation order.
    const oaiMsg: OpenAIMessage = { role: msg.role }
    if (textParts.length > 0) oaiMsg.content = textParts.join(' ')
    if (toolCalls.length > 0) oaiMsg.tool_calls = toolCalls
    if (textParts.length > 0 || toolCalls.length > 0) {
      result.push(oaiMsg)
    }
    for (const tr of toolResults) {
      result.push(tr)
    }
  }

  return result
}

/**
 * Convert Strands ToolSpec[] to OpenAI function-format tool descriptors for
 * AIDR's guard_input.tools.
 */
export function toolSpecsToOpenAI(toolSpecs: ToolSpec[]): OpenAITool[] {
  return toolSpecs.map((spec) => ({
    type: 'function' as const,
    function: {
      name: spec.name,
      description: spec.description,
      parameters: spec.inputSchema ?? {},
    },
  }))
}
