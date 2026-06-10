import {
  useCallback,
  useEffect,
  useMemo,
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

const SIDEBAR_W_KEY = "potpie-ui:sidebar-w";
const SIDEBAR_W_DEFAULT = 320;
const clampSidebarW = (w: number) => Math.min(640, Math.max(240, w));

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

  const [view, setViewState] = useState<"graph" | "timeline">(
    typeof location !== "undefined" && location.hash === "#timeline"
      ? "timeline"
      : "graph",
  );
  const setView = (v: "graph" | "timeline") => {
    setViewState(v);
    if (typeof location !== "undefined") location.hash = v === "timeline" ? "timeline" : "";
  };
  const [hidden, setHidden] = useState<Set<string>>(new Set());

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
    setBusy(true);
    setError(null);
    setSelected(null);
    setResults([]);
    try {
      const [st, g] = await Promise.all([api.status(potId), api.graph(potId)]);
      setStatus(st);
      setData({ nodes: g.nodes, edges: g.edges });
    } catch (e: any) {
      setError(e?.message || String(e));
      setData(EMPTY);
      setStatus(null);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const p = await api.pots();
        setPots(p.pots);
        const active = p.active?.id || p.pots.find((x) => x.active)?.id || null;
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
      });
    }
  };

  // Category show/hide filtering, applied to both views.
  const visible = useMemo(() => {
    const ok = (n: GraphNode) => !hidden.has(typeCategory(n.type));
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

  const toggleCat = (c: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(c) ? next.delete(c) : next.add(c);
      return next;
    });

  const counts = status?.counts || {};

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img className="logo-img" src={logoUrl} alt="Potpie" />
          <em>Graph Explorer</em>
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
                {p.name}
              </option>
            ))}
          </select>
        </div>
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
        <div className="stats">
          <span className="chip">{counts.entities ?? 0} entities</span>
          <span className="chip">{counts.claims ?? 0} claims</span>
          <span className="chip muted">{status?.backend_profile || "—"}</span>
          {busy && <span className="chip spin">loading…</span>}
        </div>
      </header>

      <div className="body">
        <aside className="sidebar" style={{ width: sidebarW }}>
          <form className="search" onSubmit={runSearch}>
            <input
              placeholder="Search entities…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit">Go</button>
          </form>

          {results.length > 0 && (
            <div className="results">
              <div className="section-title">Matches</div>
              {results.map((r) => (
                <button
                  key={r.key}
                  className="result"
                  onClick={() => pickResult(r.key)}
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

          {selected ? (
            <NodePanel node={selected} onExpand={() => expand(selected)} />
          ) : (
            <div className="hint-panel">
              {view === "graph"
                ? "Select a node to inspect it. Right-click a node (or use Expand) to pull in its neighbors."
                : "A time-ordered spine of activities (newest first) — scroll down for older. Each activity branches to what it touched / who performed it, alternating sides. Click any node for details."}
            </div>
          )}

          {view === "timeline" && (
            <div className="legend">
              <div className="section-title">Kind</div>
              {KIND_ORDER.map((k) => (
                <div className="legend-row" key={k}>
                  <span className="dot" style={{ background: kindColor(k) }} />
                  <span>{k}</span>
                </div>
              ))}
            </div>
          )}

          {legend.length > 0 && (
            <div className="legend">
              <div className="section-title">Categories</div>
              {legend.map(({ category, total, types }) => (
                <div className="cat-group" key={category}>
                  <label className="cat-head">
                    <input
                      type="checkbox"
                      checked={!hidden.has(category)}
                      onChange={() => toggleCat(category)}
                    />
                    <span className="cat-name">{category}</span>
                    <span className="legend-count">{total}</span>
                  </label>
                  {types.map(([type, n]) => (
                    <div className="legend-row indent" key={type}>
                      <TypeBadge type={type} />
                      <span>{type}</span>
                      <span className="legend-count">{n}</span>
                    </div>
                  ))}
                </div>
              ))}
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

function NodePanel({ node, onExpand }: { node: GraphNode; onExpand: () => void }) {
  const props = Object.entries(node.properties).filter(
    ([k]) => !k.startsWith("prov_") && k !== "uuid" && k !== "entity_key",
  );
  return (
    <div className="node-panel">
      <div className="section-title with-badge">
        <TypeBadge type={node.type} size={16} />
        {node.type}
      </div>
      <div className="node-key">{node.key}</div>
      <button className="expand-btn" onClick={onExpand}>
        Expand neighborhood
      </button>
      <div className="props">
        {props.map(([k, v]) => (
          <div className="prop" key={k}>
            <div className="pk">{k}</div>
            <div className="pv">{String(v)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
