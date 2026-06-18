/**
 * Basic usage example for AIGuardModel.
 *
 * Requires:
 *   CS_AIDR_TOKEN=pts_... npx tsx examples/basic-usage.ts
 */

import { AIGuardBlockedError, AIGuardModel } from '../src/index.js'

// Import your preferred Strands model provider here, e.g.:
// import { BedrockModel } from '@strands-agents/sdk'
// const downstream = new BedrockModel({ modelId: 'us.amazon.nova-lite-v1:0' })

// For this example we use a stub that returns a fixed response.
import type { BaseModelConfig, Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import { Model, Message as StrandsMessage, TextBlock } from '@strands-agents/sdk'

class StubModel extends Model<BaseModelConfig> {
  updateConfig(_config: BaseModelConfig) {}
  getConfig(): BaseModelConfig {
    return { modelId: 'stub' }
  }
  async *stream(_messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    yield { type: 'modelMessageStartEvent', role: 'assistant' }
    yield { type: 'modelContentBlockStartEvent' }
    yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: 'Hello, world!' } }
    yield { type: 'modelContentBlockStopEvent' }
    yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
  }
}

async function main() {
  const downstream = new StubModel()
  const model = new AIGuardModel(
    downstream,
    { appId: 'basic-usage-example' },
    {
      token: process.env['CS_AIDR_TOKEN'],
      baseURLTemplate: process.env['CS_AIDR_BASE_URL'],
    },
  )

  console.log('Streaming through AIDR guards...')

  try {
    for await (const event of model.stream([
      new StrandsMessage({
        role: 'user',
        content: [new TextBlock('Hello!')],
      }),
    ])) {
      if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
        process.stdout.write(event.delta.text)
      }
    }
    console.log('\nDone.')
  } catch (err) {
    if (err instanceof AIGuardBlockedError) {
      console.error(`AIDR blocked ${err.eventType}: ${err.summary}`)
    } else {
      throw err
    }
  }
}

main().catch(console.error)
