/**
 * Browser-session state for the explorer.
 *
 * The daemon's API answers 401 once the HttpOnly session cookie is missing,
 * expired, or was issued by a daemon that has since restarted. Nothing in the
 * page can mint a new one -- only `potpie ui`, which can read the daemon token,
 * can -- so the first 401 flips the whole app into a "run `potpie ui` again"
 * state instead of surfacing as one error per request.
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
