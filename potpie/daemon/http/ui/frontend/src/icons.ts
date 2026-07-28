// Per-type node glyphs for the canvas graph. Each icon is a 24x24 stroke
// path (lucide-style, round caps); GraphView scales it into the node circle.
// Path2D objects are parsed once and cached so the per-frame painter only
// strokes them. Tiny "h.01" segments render as dots via round line caps.
const ICON_PATHS: Record<string, string> = {
  // topology
  Repository:
    "M3.75 4a2.25 2.25 0 1 0 4.5 0a2.25 2.25 0 1 0-4.5 0 M6 6.25V15.5 M3.75 18a2.25 2.25 0 1 0 4.5 0a2.25 2.25 0 1 0-4.5 0 M15.75 6a2.25 2.25 0 1 0 4.5 0a2.25 2.25 0 1 0-4.5 0 M18 8.25a9.6 9.6 0 0 1-9.6 9.6",
  Service: "M2.5 4.5h19v5h-19z M2.5 14.5h19v5h-19z M6 7h.01 M6 17h.01",
  Environment:
    "M2.5 12a9.5 9.5 0 1 0 19 0a9.5 9.5 0 1 0-19 0 M2.5 12h19 M12 2.5c2.6 2.3 4 5.9 4 9.5s-1.4 7.2-4 9.5 M12 2.5c-2.6 2.3-4 5.9-4 9.5s1.4 7.2 4 9.5",
  DataStore:
    "M4 6a8 3 0 1 0 16 0a8 3 0 1 0-16 0 M4 6v12c0 1.66 3.58 3 8 3s8-1.34 8-3V6 M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3",
  Cluster: "M12 2.5l8.25 4.75v9.5L12 21.5l-8.25-4.75v-9.5z M12 12h.01",
  Dependency:
    "M12 2.5 3.5 7.25v9.5L12 21.5l8.5-4.75v-9.5z M3.5 7.25 12 12l8.5-4.75 M12 12v9.5",
  APIContract: "M8 6l-5.5 6L8 18 M16 6l5.5 6L16 18",
  Feature: "M12 3l2 5.8L20 11l-6 2.2L12 19l-2-5.8L4 11l6-2.2z",
  // people
  Person:
    "M8.5 8a3.5 3.5 0 1 0 7 0a3.5 3.5 0 1 0-7 0 M5 20a7 7 0 0 1 14 0",
  Team:
    "M6.25 8.25a2.75 2.75 0 1 0 5.5 0a2.75 2.75 0 1 0-5.5 0 M3.5 19a5.5 5.5 0 0 1 11 0 M13.6 8.9a2.4 2.4 0 1 0 4.8 0a2.4 2.4 0 1 0-4.8 0 M17 13.8a5 5 0 0 1 3.5 4.7",
  // timeline
  Activity: "M22 12h-4l-3 8L9 4l-3 8H2",
  Period: "M2.5 12a9.5 9.5 0 1 0 19 0a9.5 9.5 0 1 0-19 0 M12 7v5.5l3.5 2",
  // memory
  Preference:
    "M3 7.5h9 M16.5 7.5H21 M14 5v5 M3 16.5h5 M12.5 16.5H21 M10 14v5",
  Policy:
    "M12 2.5l7.5 3.25V12c0 4.8-3.2 8.2-7.5 9.5C7.7 20.2 4.5 16.8 4.5 12V5.75z",
  BugPattern:
    "M9 9a3 3 0 0 1 6 0v5a3 3 0 0 1-6 0z M12 9v8 M9 7.5 7 5.5 M15 7.5l2-2 M9 12H5 M15 12h4 M9.5 15.5 7 18 M14.5 15.5 17 18",
  Fix: "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z",
  Decision: "M12 3l9 9-9 9-9-9z",
  Document:
    "M14 2.5H6.5A1.5 1.5 0 0 0 5 4v16a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 20V7.5z M14 2.5V7.5h5 M9 13h6 M9 17h6",
  // soft-fail fallbacks
  Observation:
    "M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12z M9.5 12a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0",
  QualityIssue:
    "M10.7 4.3 2.7 18a1.5 1.5 0 0 0 1.3 2.25h16a1.5 1.5 0 0 0 1.3-2.25l-8-13.7a1.5 1.5 0 0 0-2.6 0z M12 9.5v4.5 M12 17.5h.01",
  __default: "M8.5 12a3.5 3.5 0 1 0 7 0a3.5 3.5 0 1 0-7 0",
};

export const ICON_BOX = 24;

// Raw path data for SVG rendering (sidebar legend/search chips).
export function iconPathD(type: string): string {
  return ICON_PATHS[type] || ICON_PATHS.__default;
}

const cache = new Map<string, Path2D>();

export function nodeIcon(type: string): Path2D {
  let p = cache.get(type);
  if (!p) {
    p = new Path2D(ICON_PATHS[type] || ICON_PATHS.__default);
    cache.set(type, p);
  }
  return p;
}
