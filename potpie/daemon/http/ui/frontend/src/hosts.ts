import type { Origin, PotRef, PotsResponse } from "./types.ts";

/** A pot is named by its host and its id together: ids are unique per host. */
export interface PotTarget {
  id: string;
  origin: Origin;
}

export const ORIGINS: readonly Origin[] = ["local", "managed"];

export function isOrigin(value: unknown): value is Origin {
  return value === "local" || value === "managed";
}

/** The selector's option value for one pot. */
export function potKey(target: PotTarget): string {
  return `${target.origin}:${target.id}`;
}

/** The inverse of `potKey`, or null for a value it never produced. */
export function parsePotKey(key: string): PotTarget | null {
  const at = key.indexOf(":");
  if (at < 0) return null;
  const origin = key.slice(0, at);
  const id = key.slice(at + 1);
  return isOrigin(origin) && id ? { origin, id } : null;
}

export function sameTarget(
  a: PotTarget | null | undefined,
  b: PotTarget | null | undefined,
): boolean {
  return Boolean(a && b && a.id === b.id && a.origin === b.origin);
}

/** The pot the page opens on.
 *
 * `potpie ui --pot managed:api` opens `?host=managed&pot=<id>`: the request
 * names its host, and a pot id is only looked up on that host. Without one,
 * the active host's active pot, then any active pot, then nothing.
 */
export function initialTarget(
  listing: Pick<PotsResponse, "pots" | "active" | "active_origin">,
  requestedPot: string | null,
  requestedHost: string | null,
): PotTarget | null {
  const host = isOrigin(requestedHost) ? requestedHost : listing.active_origin;
  if (requestedPot) {
    const match = listing.pots.find(
      (pot) =>
        pot.origin === host &&
        (pot.id === requestedPot || pot.name === requestedPot),
    );
    if (match) return { id: match.id, origin: match.origin };
  }
  if (listing.active) {
    return { id: listing.active.id, origin: listing.active.origin };
  }
  const active =
    listing.pots.find((pot) => pot.active && pot.origin === host) ??
    listing.pots.find((pot) => pot.active);
  return active ? { id: active.id, origin: active.origin } : null;
}

/** Option label: the host is named only when pots from both are listed. */
export function potLabel(pot: PotRef, mixed: boolean): string {
  const host = mixed ? ` (${pot.origin})` : "";
  // A pot listed without counts is not an empty pot: say nothing rather than 0.
  const claims = pot.counts ? ` - ${pot.counts.claims ?? 0} claims` : "";
  return `${pot.name}${host}${claims}`;
}

/** True when the listing holds pots from more than one host. */
export function isMixed(pots: PotRef[]): boolean {
  return new Set(pots.map((pot) => pot.origin)).size > 1;
}

/** The notice shown when a host predates commit history. */
export const HOST_OUTDATED_NOTICE =
  "This host runs an older Potpie without commit history. Saved plans show " +
  "mutation status and scope; commits, comparisons and rollback appear here " +
  "once the host is updated.";
