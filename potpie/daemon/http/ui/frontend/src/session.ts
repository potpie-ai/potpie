/**
 * Browser-session state for the explorer.
 *
 * The daemon returns 401 when the HttpOnly session cookie is missing, expired
 * or from a previous daemon process. Only `potpie ui` can read the daemon
 * token and create a new session. On the first 401, the app asks the user to
 * run that command again.
 */

export const SESSION_REQUIRED_MESSAGE =
  "This explorer session has ended. Run `potpie ui` in your terminal to open it again.";

export class SessionRequiredError extends Error {
  constructor(message: string = SESSION_REQUIRED_MESSAGE) {
    super(message);
    this.name = "SessionRequiredError";
  }
}

type Listener = () => void;

const listeners = new Set<Listener>();
let required = false;

export const session = {
  /** Snapshot for `useSyncExternalStore`. */
  isRequired: (): boolean => required,

  subscribe: (listener: Listener): (() => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },

  /** Record that the daemon refused this page's credential. */
  markRequired: (): void => {
    if (required) return;
    required = true;
    for (const listener of [...listeners]) listener();
  },

  /** Test hook: forget a previous refusal. */
  reset: (): void => {
    required = false;
  },
};
