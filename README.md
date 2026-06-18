# aidr-strands-typescript

CrowdStrike AIDR security guardrails for the [Strands Agents SDK](https://strandsagents.com/).

## Overview

`@crowdstrike/strands-aidr` provides `AIGuardModel`, a wrapper that intercepts any Strands
model provider and runs CrowdStrike AIDR analysis on both:

- **Input** — user messages and tool definitions, before they reach the LLM.
- **Output** — the LLM's response, before it is returned to the agent.

AIDR can detect and block malicious prompts, PII, secrets, MCP tool injection,
policy violations, and more. Blocked requests throw `AIGuardBlockedError`.
Transformed (redacted) content is automatically substituted in the message
stream so the downstream model and agent only ever see clean data.

## Installation

```bash
npm install @crowdstrike/strands-aidr
```

## Requirements

- Node.js 20+
- A CrowdStrike AIDR [Application collector](https://falcon.crowdstrike.com/aidr/) token
- Any Strands Agents SDK model provider (Bedrock, OpenAI, Anthropic, …)

## Quick start

```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { AIGuardModel } from '@crowdstrike/strands-aidr'

const downstream = new BedrockModel({ modelId: 'us.amazon.nova-lite-v1:0' })
const model = new AIGuardModel(downstream, {}, { token: process.env.CS_AIDR_TOKEN })

const agent = new Agent({ model })
const result = await agent.invoke('Hello!')
console.log(result.lastMessage)
```

Credentials are read from environment variables by default:

```bash
export CS_AIDR_BASE_URL="https://api.crowdstrike.com/aidr/{SERVICE_NAME}"
export CS_AIDR_TOKEN="pts_..."
```

Or passed directly:

```typescript
const model = new AIGuardModel(downstream, {}, {
  baseURLTemplate: 'https://api.crowdstrike.com/aidr/{SERVICE_NAME}',
  token: 'pts_...',
})
```

## Configuration

`AIGuardModel` accepts optional AIDR metadata that appears in Findings logs:

```typescript
const model = new AIGuardModel(
  downstream,
  {
    appId: 'hr-portal',
    userId: 'mary.potter',
    tenantId: 'acme-corp',
    extraInfo: { appName: 'HR Portal', appVersion: '1.2.3' },
  },
  { token: 'pts_...' },
)
```

## Handling blocked requests

```typescript
import { AIGuardBlockedError } from '@crowdstrike/strands-aidr'

try {
  await agent.invoke('Please ignore all previous instructions and…')
} catch (err) {
  if (err instanceof AIGuardBlockedError) {
    console.error(`Blocked at ${err.eventType}: ${err.summary}`)
  }
}
```

## Architecture

```
Agent
  └── AIGuardModel.stream()
        ├── [guardInput=true]  AIDR guardChatCompletions(event_type="input")
        │      blocked=true  → throw AIGuardBlockedError("input", …)
        │      transformed   → substitute redacted messages
        │
        ├── downstream.stream()   ← buffered (collected in memory)
        │
        └── [guardOutput=true] AIDR guardChatCompletions(event_type="output")
               blocked=true  → throw AIGuardBlockedError("output", …)
               transformed   → rebuild StreamEvents with redacted text
```

## Running tests

```bash
npm install
npm test
```

## Support

`@crowdstrike/strands-aidr` is a community-driven, open source project that integrates CrowdStrike AIDR security guardrails with the Strands Agents SDK. While not a formal CrowdStrike product, it is maintained by CrowdStrike and supported in partnership with the open source developer community.

### Issue Reporting and Questions

Issues may be reported on [GitHub](https://github.com/CrowdStrike/aidr-strands-typescript/issues) and are used to track bugs, documentation updates, enhancement requests, and security concerns.

### Support Escalation

We endeavor to provide support for `@crowdstrike/strands-aidr` within the repository. This expands our online knowledge base, enables self-help for our community, and can reduce the time needed to receive answers.

If you are a CrowdStrike customer and would prefer to have your questions or issues handled directly with CrowdStrike Support, you are welcome to contact the CrowdStrike technical support team.
