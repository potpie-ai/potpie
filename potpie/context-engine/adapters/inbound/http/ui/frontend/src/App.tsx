import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { api } from "./api";
import GraphView from "./GraphView";
import Timeline from "./Timeline";
import {
  CATEGORY_ORDER,
  KIND_ORDER,
  kindColor,
  typeCategory,
  typeColor,
  typesInCategory,
  UI,
} from "./theme";
import { iconPathD } from "./icons";
import logoUrl from "./logo_dark.svg";
import type {
  GraphData,
  GraphEdge,
  GraphNode,
  PotRef,
  SearchEntity,
  StatusResponse,
} from "./types";

const EMPTY: GraphData = { nodes: [], edges: [] };
const TIMELINE_TYPES = typesInCategory("timeline");

const SIDEBAR_W_KEY = "potpie-ui:sidebar-w";
const SIDEBAR_W_DEFAULT = 320;
const clampSidebarW = (w: number) => Math.min(640, Math.max(240, w));
const requestedPot = () =>
  typeof location === "undefined"
    ? null
    : new URLSearchParams(location.search).get("pot");

function endId(v: string | GraphNode): string {
  return typeof v === "string" ? v : v.id;
}

function edgeId(e: GraphEdge): string {
  return `${endId(e.source)}|${e.predicate}|${endId(e.target)}`;
}

function mergeGraph(a: GraphData, b: GraphData): GraphData {
  const nodes = new Map<string, GraphNode>();
  for (const n of a.nodes) nodes.set(n.id, n);
  for (const n of b.nodes) if (!nodes.has(n.id)) nodes.set(n.id, n);
  const edges = new Map<string, GraphEdge>();
  for (const e of [...a.edges, ...b.edges]) {
    edges.set(edgeId(e), {
      id: edgeId(e),
      source: endId(e.source),
      target: endId(e.target),
      predicate: e.predicate,
    });
  }
  return { nodes: [...nodes.values()], edges: [...edges.values()] };
}

export default function App() {
  const [pots, setPots] = useState<PotRef[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [data, setData] = useState<GraphData>(EMPTY);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchEntity[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadRequestRef = useRef(0);

  const [view, setViewState] = useState<"graph" | "timeline">(
    typeof location !== "undefined" && location.hash === "#timeline"
      ? "timeline"
      : "graph",
  );
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [showHint, setShowHint] = useState(true);

  const revealTimelineTypes = () =>
    setHidden((prev) => {
      const next = new Set(prev);
      let changed = false;
      for (const t of TIMELINE_TYPES) {
        if (next.delete(t)) changed = true;
      }
      return changed ? next : prev;
    });

  const setView = (v: "graph" | "timeline") => {
    setViewState(v);
    if (typeof location !== "undefined") location.hash = v === "timeline" ? "timeline" : "";
    if (v === "timeline") revealTimelineTypes();
  };

  const [sidebarW, setSidebarW] = useState(() => {
    if (typeof localStorage === "undefined") return SIDEBAR_W_DEFAULT;
    const v = Number(localStorage.getItem(SIDEBAR_W_KEY));
    return Number.isFinite(v) && v > 0 ? clampSidebarW(v) : SIDEBAR_W_DEFAULT;
  });

  const startSidebarResize = (e: ReactPointerEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarW;
    const widthAt = (ev: PointerEvent) =>
      clampSidebarW(startW + ev.clientX - startX);
    const move = (ev: PointerEvent) => setSidebarW(widthAt(ev));
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      localStorage.setItem(SIDEBAR_W_KEY, String(widthAt(ev)));
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const resetSidebarW = () => {
    setSidebarW(SIDEBAR_W_DEFAULT);
    localStorage.setItem(SIDEBAR_W_KEY, String(SIDEBAR_W_DEFAULT));
  };

  const loadPot = useCallback(async (potId: string) => {
    const requestId = ++loadRequestRef.current;
    const isCurrent = () => requestId === loadRequestRef.current;
    setBusy(true);
    setError(null);
    setSelected(null);
    setResults([]);
    try {
      const [st, g] = await Promise.all([api.status(potId), api.graph(potId)]);
      if (!isCurrent()) return;
      setStatus(st);
      setData({ nodes: g.nodes, edges: g.edges });
    } catch (e: any) {
      if (!isCurrent()) return;
      setError(e?.message || String(e));
      setData(EMPTY);
      setStatus(null);
    } finally {
      if (isCurrent()) setBusy(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const p = await api.pots();
        setPots(p.pots);
        const requested = requestedPot();
        const requestedRef = requested
          ? p.pots.find((x) => x.id === requested || x.name === requested)
          : null;
        const active =
          requestedRef?.id || p.active?.id || p.pots.find((x) => x.active)?.id || null;
        setActiveId(active);
        if (active) await loadPot(active);
      } catch (e: any) {
        setError(e?.message || String(e));
      }
    })();
  }, [loadPot]);

  const switchPot = async (potId: string) => {
    const ref = pots.find((p) => p.id === potId);
    if (!ref) return;
    setBusy(true);
    try {
      await api.usePot(ref.id);
      setActiveId(potId);
      setPots((ps) => ps.map((p) => ({ ...p, active: p.id === potId })));
      await loadPot(potId);
    } catch (e: any) {
      setError(e?.message || String(e));
      setBusy(false);
    }
  };

  const expand = useCallback(
    async (node: GraphNode) => {
      if (!activeId) return;
      try {
        const nb = await api.neighborhood(node.key, 1, activeId);
        setData((cur) => mergeGraph(cur, { nodes: nb.nodes, edges: nb.edges }));
      } catch (e: any) {
        setError(e?.message || String(e));
      }
    },
    [activeId],
  );

  const runSearch = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !activeId) return;
    try {
      const r = await api.search(query.trim(), activeId);
      setResults(r.entities || []);
    } catch (err: any) {
      setError(err?.message || String(err));
    }
  };

  const pickResult = (key: string) => {
    const inGraph = data.nodes.find((n) => n.id === key);
    if (inGraph) {
      setSelected(inGraph);
    } else {
      api.neighborhood(key, 1, activeId || undefined).then((nb) => {
        setData((cur) => mergeGraph(cur, { nodes: nb.nodes, edges: nb.edges }));
        const found = nb.nodes.find((n) => n.id === key);
        if (found) setSelected(found);
      }).catch((e: any) => setError(e?.message || String(e)));
    }
  };

  // Per-type show/hide filtering, applied to both views. `hidden` holds entity
  // types; a category checkbox cascades to (and reflects) all of its types.
  const visible = useMemo(() => {
    const ok = (n: GraphNode) => !hidden.has(n.type);
    const nodes = data.nodes.filter(ok);
    const ids = new Set(nodes.map((n) => n.id));
    const edges = data.edges.filter(
      (e) => ids.has(endId(e.source)) && ids.has(endId(e.target)),
    );
    return { nodes, edges };
  }, [data, hidden]);

  // Legend: types grouped by ontology category, from the full (unfiltered) graph.
  const legend = useMemo(() => {
    const byCat = new Map<string, Map<string, number>>();
    for (const n of data.nodes) {
      const cat = typeCategory(n.type);
      if (!byCat.has(cat)) byCat.set(cat, new Map());
      const m = byCat.get(cat)!;
      m.set(n.type, (m.get(n.type) || 0) + 1);
    }
    return CATEGORY_ORDER.filter((c) => byCat.has(c)).map((c) => ({
      category: c,
      total: [...byCat.get(c)!.values()].reduce((a, b) => a + b, 0),
      types: [...byCat.get(c)!.entries()].sort((a, b) => b[1] - a[1]),
    }));
  }, [data]);

  const toggleType = (t: string) => {
    if (view === "timeline" && TIMELINE_TYPES.includes(t)) return;
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(t) ? next.delete(t) : next.add(t);
      return next;
    });
  };

  // Cascade a category checkbox to all its types: hide them all, or reveal them.
  const setCategoryHidden = (types: string[], hide: boolean) => {
    if (
      view === "timeline" &&
      hide &&
      types.every((t) => TIMELINE_TYPES.includes(t))
    ) {
      return;
    }
    setHidden((prev) => {
      const next = new Set(prev);
      for (const t of types) hide ? next.add(t) : next.delete(t);
      return next;
    });
  };

  const counts = status?.counts || {};

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img className="logo-img" src={logoUrl} alt="Potpie" />
          <span className="title">Graph Explorer</span>
        </div>

        <div className="topbar-search">
          <form onSubmit={runSearch}>
            <input
              placeholder="Search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <svg className="search-ico" viewBox="0 0 24 24" aria-hidden>
              <circle
                cx="11"
                cy="11"
                r="7"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
              <path
                d="M20 20l-3.5-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </form>
          {results.length > 0 && (
            <div className="search-results">
              <div className="section-title">Matches</div>
              {results.map((r) => (
                <button
                  key={r.key}
                  className="result"
                  onClick={() => {
                    pickResult(r.key);
                    setResults([]);
                  }}
                  title={r.key}
                >
                  <TypeBadge
                    type={r.labels.find((l) => l !== "Entity") || "Entity"}
                  />
                  <span className="result-key">{r.key}</span>
                  <span className="score">{r.score.toFixed(2)}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="topbar-spacer" />

        <div className="viewtoggle">
          <button
            className={view === "graph" ? "on" : ""}
            onClick={() => setView("graph")}
          >
            Graph
          </button>
          <button
            className={view === "timeline" ? "on" : ""}
            onClick={() => setView("timeline")}
          >
            Timeline
          </button>
        </div>

        <div className="pot-select">
          <label>Pot</label>
          <select
            value={activeId || ""}
            onChange={(e) => switchPot(e.target.value)}
            disabled={busy}
          >
            {pots.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} - {p.counts?.claims ?? 0} claims
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="body">
        <aside className="sidebar" style={{ width: sidebarW }}>
          {selected ? (
            <NodePanel node={selected} />
          ) : (
            <div className="hint-panel">
              {view === "graph"
                ? "Select a node to inspect it. Right-click a node (or use Expand) to pull in its neighbors."
                : "A time-ordered spine of activities (newest first) — scroll down for older. Each activity branches to what it touched / who performed it, alternating sides. Click any node for details."}
            </div>
          )}

          {view === "timeline" && (
            <div className="legend">
              <div className="card-title">Kind</div>
              <div className="kind-list">
                {KIND_ORDER.map((k) => (
                  <div className="kind-row" key={k}>
                    <span
                      className="kind-swatch"
                      style={{ background: kindColor(k) }}
                    />
                    <span>{k}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {legend.length > 0 && (
            <div className="legend">
              <div className="card-title">Categories</div>
              <div className="cat-list">
                {legend.map(({ category, total, types }) => {
                  const typeKeys = types.map(([t]) => t);
                  const hiddenCount = typeKeys.filter((t) =>
                    hidden.has(t),
                  ).length;
                  const allHidden = hiddenCount === typeKeys.length;
                  const timelineLocked =
                    view === "timeline" && category === "timeline";
                  return (
                    <div className="cat-group" key={category}>
                      <div
                        className="cat-head"
                        onClick={() =>
                          !timelineLocked &&
                          setCategoryHidden(typeKeys, !allHidden)
                        }
                      >
                        <span className="cat-name">{category}</span>
                        <span className="count-badge">{pad(total)}</span>
                        <VisToggle
                          visible={!allHidden}
                          disabled={timelineLocked}
                          title={
                            timelineLocked
                              ? "Timeline types stay visible in Timeline view"
                              : `Toggle all ${category} types`
                          }
                          onToggle={() =>
                            setCategoryHidden(typeKeys, !allHidden)
                          }
                        />
                      </div>
                      <div className="cat-rows">
                        {types.map(([type, n]) => {
                          const isHidden = hidden.has(type);
                          return (
                            <div
                              className={`cat-row${isHidden ? " dim" : ""}`}
                              key={type}
                              onClick={() => !timelineLocked && toggleType(type)}
                            >
                              <TypeBadge type={type} size={15} />
                              <span className="cat-row-name">{type}</span>
                              <span className="count-badge">{pad(n)}</span>
                              <VisToggle
                                visible={!isHidden}
                                disabled={timelineLocked}
                                title={
                                  timelineLocked
                                    ? "Timeline types stay visible in Timeline view"
                                    : `Toggle ${type}`
                                }
                                onToggle={() => toggleType(type)}
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </aside>

        <div
          className="sidebar-resizer"
          title="Drag to resize · double-click to reset"
          onPointerDown={startSidebarResize}
          onDoubleClick={resetSidebarW}
        />

        <main className="main">
          {error && <div className="error">{error}</div>}

          <div className="canvas-stats">
            <div className="stat-pill">
              <span className="stat-seg">
                <b>{counts.entities ?? 0}</b> entities
              </span>
              <span className="stat-seg">
                <b>{counts.claims ?? 0}</b> claims
              </span>
            </div>
            <span className="stat-chip">{status?.backend_profile || "—"}</span>
            {busy && <span className="stat-chip spin">loading…</span>}
          </div>

          <div className="canvas-tools">
            <button
              className={`info-btn${showHint ? " on" : ""}`}
              title="Toggle help"
              onClick={() => setShowHint((s) => !s)}
            >
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden>
                <circle
                  cx="12"
                  cy="12"
                  r="9.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                />
                <path
                  d="M12 11v5"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
                <circle cx="12" cy="7.8" r="1.1" fill="currentColor" />
              </svg>
            </button>
            {showHint && (
              <div className="hint-card">
                {view === "graph"
                  ? "click a node for details · right-click to expand · scroll to zoom · drag to pin"
                  : "newest activities first · scroll down for older · click any node for details"}
              </div>
            )}
          </div>

          {view === "graph" ? (
            <GraphView
              data={visible}
              selectedId={selected?.id || null}
              onSelect={setSelected}
              onExpand={expand}
            />
          ) : (
            <Timeline
              data={visible}
              selectedId={selected?.id || null}
              onSelect={setSelected}
            />
          )}
        </main>
      </div>
    </div>
  );
}

// Counts read as zero-padded two-digit numbers in the design (01, 07, 16…).
function pad(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

// Per-row visibility toggle: a square button showing − when the type is shown
// (lime tint) and + when hidden (neutral), replacing the legend checkboxes.
function VisToggle({
  visible,
  disabled,
  title,
  onToggle,
}: {
  visible: boolean;
  disabled?: boolean;
  title?: string;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className={`vis-toggle ${visible ? "on" : "off"}`}
      disabled={disabled}
      title={title}
      aria-pressed={visible}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) onToggle();
      }}
    >
      {visible ? "−" : "+"}
    </button>
  );
}

// Mini version of the canvas node rendering (colored disc + dark type glyph),
// so the sidebar teaches the icon → type mapping used on the graph.
function TypeBadge({ type, size = 14 }: { type: string; size?: number }) {
  return (
    <svg
      className="type-badge"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      aria-hidden
    >
      <circle cx="12" cy="12" r="12" fill={typeColor(type)} />
      <path
        d={iconPathD(type)}
        fill="none"
        stroke={UI.iconStroke}
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        transform="translate(4.8 4.8) scale(0.6)"
      />
    </svg>
  );
}

const isDateKey = (k: string) =>
  /(_at|_ts|timestamp|^created$|^updated$|^occurred)$/i.test(k);

// Property display order: date first, then name / summary / description, any
// other fields next, and provenance last.
function propRank(k: string): number {
  if (isDateKey(k)) return 0;
  if (k === "name") return 1;
  if (k === "summary") return 2;
  if (k === "description") return 3;
  if (/provenance/i.test(k)) return 100;
  return 50;
}

// Render epoch values on *_at / timestamp keys as a readable local timestamp;
// everything else (already-formatted strings, ISO dates) passes through.
function formatValue(key: string, value: unknown): string {
  const s = String(value);
  if (isDateKey(key)) {
    const num = typeof value === "number" ? value : Number(s);
    if (Number.isFinite(num) && num > 1e9) {
      const ms = num < 1e12 ? num * 1000 : num; // seconds → ms
      const d = new Date(ms);
      if (!Number.isNaN(d.getTime())) {
        return d.toLocaleString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
      }
    }
  }
  return s;
}

function NodePanel({ node }: { node: GraphNode }) {
  const props = Object.entries(node.properties)
    .filter(([k]) => !k.startsWith("prov_") && k !== "uuid" && k !== "entity_key")
    .sort((a, b) => propRank(a[0]) - propRank(b[0]));
  return (
    <div className="node-panel">
      <div className="np-head">
        <div className="np-type">
          <TypeBadge type={node.type} size={16} />
          {node.type}
        </div>
        <div className="node-key">{node.key}</div>
      </div>
      <div className="props">
        {props.map(([k, v]) => (
          <div className="prop" key={k}>
            <div className="pk">{k}</div>
            <div className="pv">{formatValue(k, v)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
