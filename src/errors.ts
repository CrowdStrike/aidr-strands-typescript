/**
 * Error thrown when CrowdStrike AIDR blocks an input or output.
 *
 * @example
 * ```typescript
 * try {
 *   await agent('tell me how to hack')
 * } catch (err) {
 *   if (err instanceof AIGuardBlockedError) {
 *     console.log(err.eventType) // 'input' | 'output'
 *     console.log(err.summary)
 *   }
 * }
 * ```
 */
export class AIGuardBlockedError extends Error {
  readonly eventType: 'input' | 'output'
  readonly summary: string

  constructor(eventType: 'input' | 'output', summary: string) {
    super(`AIDR blocked ${eventType}: ${summary}`)
    this.name = 'AIGuardBlockedError'
    this.eventType = eventType
    this.summary = summary
  }
}

/**
 * Error thrown when AIDR is unreachable or returns no result and failOpen is false.
 */
export class AIGuardUnavailableError extends Error {
  readonly eventType: 'input' | 'output'

  constructor(eventType: 'input' | 'output') {
    super(`AIDR ${eventType} guard unavailable and failOpen is disabled`)
    this.name = 'AIGuardUnavailableError'
    this.eventType = eventType
  }
}

/**
 * Error thrown when AIDR signals transformed: true but provides no usable guard_output.
 */
export class AIGuardTransformFailedError extends Error {
  readonly eventType: 'input' | 'output'

  constructor(eventType: 'input' | 'output') {
    super(`AIDR ${eventType} guard signalled transformation but provided no guard_output`)
    this.name = 'AIGuardTransformFailedError'
    this.eventType = eventType
  }
}
