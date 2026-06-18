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
