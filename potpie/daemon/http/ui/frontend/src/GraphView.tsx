import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, GraphEdge, GraphNode } from "./types";
import { typeColor, UI } from "./theme";
import { ICON_BOX, nodeIcon } from "./icons";

interface Props {
  data: GraphData;
  selectedId: string | null;
  onSelect: (node: GraphNode | null) => void;
  onExpand: (node: GraphNode) => void;
  fitOnLoad?: boolean;
  onSelectEdge?: (edge: GraphEdge) => void;
  focusOnSelect?: boolean;
}

const diffColors: Record<string, string> = {
  added: UI.accent, modified: "#FFD86E", invalidated: "#F15B5B", retired: "#F15B5B", reactivated: "#45C7A8", before: "#a6a6af", previous: "#F15B5B", "recorded context": "#727a74", context: "#6c716e",
};

const radius = (n: GraphNode) => (4 + Math.min(7, Math.sqrt(n.degree || 0) * 2.2)) * (n.diff_status === "context" ? 0.78 : 1);

// Labels declutter by progressive disclosure: at far zoom only hubs are
// captioned; zooming in lowers the degree bar until everything is labeled.
// Hovered/selected nodes are always labeled regardless of zoom.
const labelMinDegree = (scale: number) =>
  scale >= 2.8 ? 0 : scale >= 1.8 ? 2 : scale >= 1.1 ? 4 : scale >= 0.7 ? 8 : 12;

// react-force-graph wants {nodes, links}; links reference node ids via
// source/target (the library rewrites those to node refs in place).
export default function GraphView({
  data,
  selectedId,
  onSelect,
  onExpand,
  fitOnLoad = false,
  onSelectEdge,
  focusOnSelect = false,
}: Props) {
  const fgRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<GraphNode | null>(null);
  const nodeRefs = useRef(new Map<string, GraphNode>());
  const [dims, setDims] = useState({ w: 800, h: 600 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // bail when unchanged: RO can fire for subpixel/zoom-rounding reasons, and
    // an always-new dims object would re-render (and re-size the canvas) each
    // time — at some browser-zoom roundings that fed back into a visible
    // resize/shake loop.
    const measure = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      setDims((d) => (d.w === w && d.h === h ? d : { w, h }));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    const degree: Record<string, number> = {};
    for (const e of data.edges) {
      const s = typeof e.source === "string" ? e.source : e.source.id;
      const t = typeof e.target === "string" ? e.target : e.target.id;
      degree[s] = (degree[s] || 0) + 1;
      degree[t] = (degree[t] || 0) + 1;
    }
    const currentIds = new Set(data.nodes.map((node) => node.id));
    for (const id of nodeRefs.current.keys()) {
      if (!fitOnLoad && !currentIds.has(id)) nodeRefs.current.delete(id);
    }
    const nodes = data.nodes.map((node) => {
      const existing = nodeRefs.current.get(node.id);
      if (existing) {
        Object.assign(existing, node, { degree: degree[node.id] || 0 });
        return existing;
      }
      const created = { ...node, degree: degree[node.id] || 0 };
      nodeRefs.current.set(node.id, created);
      return created;
    });
    const links = data.edges.map(edge => ({ ...edge, curvature: 0 }));
    const pairs = new Map<string, typeof links>();
    for (const edge of links) {
      const source = typeof edge.source === "string" ? edge.source : edge.source.id;
      const target = typeof edge.target === "string" ? edge.target : edge.target.id;
      const key = JSON.stringify([source, target].sort());
      const group = pairs.get(key) || [];
      group.push(edge); pairs.set(key, group);
    }
    for (const group of pairs.values()) {
      group.sort((left, right) => left.id.localeCompare(right.id));
      group.forEach((edge, index) => {
        const source = typeof edge.source === "string" ? edge.source : edge.source.id;
        const target = typeof edge.target === "string" ? edge.target : edge.target.id;
        edge.curvature = source === target ? 0.45 + index * 0.2
          : (index - (group.length - 1) / 2) * 0.35 * (source < target ? 1 : -1);
      });
    }
    return { nodes, links };
  }, [data, fitOnLoad]);

  // Stable camera: force-graph re-zooms to 4/cbrt(nodeCount) on EVERY data
  // update while the zoom level still equals its internal "default"
  // (lastSetZoom) — with load/expand/merge each changing the count, the graph
  // looked like it kept resizing. Starting an epsilon off 1 defeats that
  // equality check for good: the camera sits at (an effective) 100% and only
  // moves when the user pans/zooms or uses the controls below.
  const Z100 = 1.000001;
  const [zoomPct, setZoomPct] = useState(100);
  const zoomFrame = useRef<number | null>(null);
  const pendingZoom = useRef(100);
  useEffect(() => {
    fgRef.current?.zoom(Z100, 0);
    if (fitOnLoad) {
      fgRef.current?.d3Force("link")?.distance(60);
      fgRef.current?.d3Force("charge")?.strength(-90);
      fgRef.current?.d3ReheatSimulation();
    }
    return () => {
      if (zoomFrame.current !== null) cancelAnimationFrame(zoomFrame.current);
    };
  }, [fitOnLoad]);
  const resetZoom = () => {
    fgRef.current?.centerAt(0, 0, 300);
    fgRef.current?.zoom(Z100, 300);
  };
  const fitted = useRef(false);
  const [initialFitPainted, setInitialFitPainted] = useState(false);
  const graphReady = !fitOnLoad || !graphData.nodes.length || initialFitPainted;
  useEffect(() => {
    if (!focusOnSelect || !selectedId) return;
    const node = graphData.nodes.find(item => item.id === selectedId);
    const edge = graphData.links.find(item => item.id === selectedId);
    const endpoints = edge ? [edge.source, edge.target].map(endpoint => nodeRefs.current.get(typeof endpoint === "string" ? endpoint : endpoint.id)) : [];
    const points = (node ? [node] : endpoints).filter((point): point is GraphNode => Boolean(point && point.x !== undefined && point.y !== undefined));
    if (!points.length) return;
    const xs = points.map(point => point.x!);
    const ys = points.map(point => point.y!);
    const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const fg = fgRef.current;
    fg?.centerAt((left + right) / 2, (top + bottom) / 2, 250);
    if (fg && fg.zoom() < 1) fg.zoom(Math.min(2, (dims.w - 100) / (right - left + 80), (dims.h - 100) / (bottom - top + 80)), 250);
  }, [selectedId, focusOnSelect, graphData, dims.w, dims.h]);
  useEffect(() => {
    if (fitOnLoad && fitted.current) {
      fgRef.current?.zoomToFit?.(0, 60);
      if (fgRef.current?.zoom() > 3) fgRef.current.zoom(3, 0);
    }
  }, [dims.w, dims.h, fitOnLoad]);
  const fitZoom = () => {
    const fg = fgRef.current;
    fg?.zoomToFit?.(fitOnLoad ? 0 : 300, 60);
    if (fitOnLoad && fg?.zoom() > 3) fg.zoom(3, 0);
  };
  const zoomBy = (factor: number) => {
    const fg = fgRef.current;
    if (!fg?.zoom) return;
    fg.zoom(Math.max(0.05, Math.min(12, fg.zoom() * factor)), 200);
  };

  // The selected node is skipped in the normal per-node pass and repainted in
  // onRenderFramePost (emphasized), so it and its label sit above everything.
  const paintNode = (
    node: GraphNode,
    ctx: CanvasRenderingContext2D,
    scale: number,
    emphasized = false,
  ) => {
    if (!emphasized && node.id === selectedId) return;
    ctx.save();
    const context = node.diff_status === "context";
    const changed = Boolean(node.diff_status && !context && node.diff_status !== "recorded context");
    ctx.globalAlpha = context && !emphasized ? 0.6 : node.ghost ? 0.7 : 1;
    const r = radius(node);
    const hovered = node.id === hoverRef.current?.id;
    ctx.beginPath();
    ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
    ctx.fillStyle = diffColors[node.diff_status || ""] || typeColor(node.type);
    if (emphasized) {
      // glow renders in device space (unaffected by zoom) — a steady halo
      ctx.save();
      ctx.shadowColor = context ? "#a6a6af" : UI.glow;
      ctx.shadowBlur = 24;
      ctx.fill();
      ctx.restore();
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = UI.ring;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r + 3 / scale, 0, 2 * Math.PI);
      ctx.strokeStyle = UI.ringSoft;
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
    } else {
      ctx.fill();
      if (changed) {
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r + 3 / scale, 0, 2 * Math.PI);
        ctx.strokeStyle = diffColors[node.diff_status || ""];
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        ctx.globalAlpha = node.ghost ? 0.7 : 1;
      }
    }

    // Type glyph inside the circle — skipped while the node is too small on
    // screen to read, so the far view stays plain dots.
    if (r * scale >= 5) {
      const s = (r * 1.0) / ICON_BOX;
      ctx.save();
      ctx.translate(node.x! - (ICON_BOX / 2) * s, node.y! - (ICON_BOX / 2) * s);
      ctx.scale(s, s);
      ctx.strokeStyle = UI.iconStroke;
      ctx.lineWidth = 2.4;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke(nodeIcon(node.type));
      ctx.restore();
    }

    if (emphasized || hovered || (changed && graphData.nodes.length < 60) || (node.degree || 0) >= labelMinDegree(scale)) {
      const fontSize = Math.max(2.5, 11 / scale);
      const weight = emphasized || hovered ? "600 " : "";
      ctx.font = `${weight}${fontSize}px ${UI.font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const label = node.caption || node.key;
      const max = emphasized || hovered ? 42 : 24;
      const text = label.length > max ? label.slice(0, max - 1) + "…" : label;
      const y = node.y! + r + 2 / scale;
      // dark halo keeps labels readable over edges and other nodes
      ctx.lineWidth = (emphasized ? 4 : 3) / scale;
      ctx.lineJoin = "round";
      ctx.strokeStyle = emphasized ? UI.haloStrong : UI.halo;
      ctx.strokeText(text, node.x!, y);
      ctx.fillStyle = emphasized || hovered ? UI.labelBright : context ? "#979c98" : UI.label;
      ctx.fillText(text, node.x!, y);
    }
    ctx.restore();
  };

  // Faint blueprint dot-grid, fixed in graph space so it pans/zooms with the
  // nodes. Spacing doubles/halves with zoom to keep a steady screen density.
  const paintGrid = (ctx: CanvasRenderingContext2D, scale: number) => {
    const fg = fgRef.current;
    if (!fg?.screen2GraphCoords) return;
    const tl = fg.screen2GraphCoords(0, 0);
    const br = fg.screen2GraphCoords(dims.w, dims.h);
    let step = 32;
    while (step * scale < 26) step *= 2;
    while (step * scale > 52) step /= 2;
    const d = 1.2 / scale;
    const x0 = Math.floor(tl.x / step) * step;
    const y0 = Math.floor(tl.y / step) * step;
    ctx.fillStyle = UI.gridDot;
    for (let x = x0; x <= br.x; x += step)
      for (let y = y0; y <= br.y; y += step)
        ctx.fillRect(x - d / 2, y - d / 2, d, d);
  };

  const pointerArea = (node: GraphNode, color: string, ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x!, node.y!, radius(node) + 2, 0, 2 * Math.PI);
    ctx.fill();
  };

  // Geometry hit-test, used as a fallback for browsers that defeat the
  // library's canvas-readback picking. Brave's fingerprinting protection
  // ("farbling") randomizes getImageData(), so the colored shadow-canvas the
  // library reads to map cursor→node never matches — every click registers as
  // empty background. We recover the node from the click position + node
  // radius instead, which needs no pixel readback and works everywhere.
  const nodeAtEvent = (e: MouseEvent): GraphNode | null => {
    const fg = fgRef.current;
    const canvas = wrapRef.current?.querySelector("canvas");
    if (!fg?.screen2GraphCoords || !canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const p = fg.screen2GraphCoords(e.clientX - rect.left, e.clientY - rect.top);
    let best: GraphNode | null = null;
    let bestD = Infinity;
    for (const n of graphData.nodes as GraphNode[]) {
      if (n.x == null || n.y == null) continue;
      const rr = radius(n) + 4;
      const dx = n.x - p.x;
      const dy = n.y - p.y;
      const d = dx * dx + dy * dy;
      if (d <= rr * rr && d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  };

  return (
    <div ref={wrapRef} className="graph-canvas" style={{ visibility: graphReady ? "visible" : "hidden" }}>
      <ForceGraph2D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        graphData={graphData as any}
        linkCurvature="curvature"
        backgroundColor={UI.bg}
        nodeId="id"
        nodeRelSize={5}
        nodeCanvasObject={(n: any, ctx: any, s: any) => paintNode(n, ctx, s)}
        nodePointerAreaPaint={(n: any, c: any, ctx: any) => pointerArea(n, c, ctx)}
        onRenderFramePre={(ctx: CanvasRenderingContext2D, scale: number) =>
          paintGrid(ctx, scale)
        }
        onRenderFramePost={(ctx: CanvasRenderingContext2D, scale: number) => {
          // The first stop fits the camera during a frame that began at the
          // old scale. Reveal only after a complete frame at the fitted scale.
          if (fitOnLoad && fitted.current && !initialFitPainted && scale === fgRef.current?.zoom()) {
            setInitialFitPainted(true);
          }
          if (!selectedId) return;
          const n = graphData.nodes.find((x) => x.id === selectedId) as
            | GraphNode
            | undefined;
          if (n && n.x != null && n.y != null) paintNode(n, ctx, scale, true);
        }}
        linkColor={(link: GraphEdge) => (link.record_id || link.id) === selectedId ? (link.diff_status === "context" ? "#a6a6af" : UI.accent) : link.diff_status === "context" ? "rgba(150,158,153,0.22)" : diffColors[link.diff_status || ""] || UI.link}
        linkLineDash={(link: { diff_status?: string }) => ["retired", "invalidated", "previous"].includes(link.diff_status || "") ? [4, 3] : null}
        linkWidth={(link: GraphEdge) => (link.record_id || link.id) === selectedId ? 3 : link.diff_status === "context" ? 0.7 : link.diff_status ? 2 : 1}
        linkHoverPrecision={8}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={0.92}
        linkLabel={(link: GraphEdge) => {
          const label = document.createElement("span");
          label.textContent = `${link.diff_status ? `${link.diff_status}: ` : ""}${link.predicate}`;
          return label.innerHTML;
        }}
        nodeLabel={(node: GraphNode) => {
          const label = document.createElement("span");
          label.textContent = node.caption;
          return label.innerHTML;
        }}
        warmupTicks={fitOnLoad ? 60 : 0}
        cooldownTicks={fitOnLoad ? 0 : 120}
        onEngineStop={() => {
          // The engine can stop its previous (empty) data before the pending
          // graph update has initialized the new nodes.
          if (!fitOnLoad || !graphData.nodes.length || graphData.nodes.some(node => !Number.isFinite(node.x) || !Number.isFinite(node.y))) return;
          // Preserve the mental map when toggling before/after or inspecting
          // a record. Newly loaded nodes can settle around the fixed nodes.
          for (const node of graphData.nodes) Object.assign(node, { fx: node.x, fy: node.y });
          if (!fitted.current) {
            fitted.current = true;
            fgRef.current?.zoomToFit?.(0, 60);
            if (fgRef.current?.zoom() > 3) fgRef.current.zoom(3, 0);
          }
        }}
        onZoom={({ k }: { k: number }) => {
          // The library can emit zoom synchronously while applying new props.
          // Defer the React update and coalesce wheel events into one frame.
          pendingZoom.current = Math.round(k * 100);
          if (zoomFrame.current !== null) return;
          zoomFrame.current = requestAnimationFrame(() => {
            zoomFrame.current = null;
            setZoomPct(pendingZoom.current);
          });
        }}
        onNodeHover={(n: any) => {
          hoverRef.current = (n as GraphNode) || null;
        }}
        onNodeClick={(n: any) => onSelect(n as GraphNode)}
        onLinkClick={onSelectEdge ? (edge: GraphEdge) => onSelectEdge(edge) : undefined}
        onNodeRightClick={(n: any) => onExpand(n as GraphNode)}
        onBackgroundClick={(e: MouseEvent) => {
          // Fires for genuine empty clicks everywhere, and for *every* click in
          // browsers whose canvas readback is farbled (Brave). Recover the node
          // geometrically; fall back to deselect when the click was truly empty.
          const n = nodeAtEvent(e);
          onSelect(n || null);
        }}
        onBackgroundRightClick={(e: MouseEvent) => {
          const n = nodeAtEvent(e);
          if (n) onExpand(n);
        }}
        onNodeDragEnd={(n: any) => {
          n.fx = n.x;
          n.fy = n.y;
        }}
      />
      <div className="zoom-ctl">
        <button onClick={() => zoomBy(1.25)} title="Zoom in">
          +
        </button>
        <button
          className="zoom-val"
          onClick={resetZoom}
          title="Reset to 100%"
        >
          {zoomPct}%
        </button>
        <button onClick={() => zoomBy(0.8)} title="Zoom out">
          −
        </button>
        <button onClick={fitZoom} title="Fit graph to view">
          fit
        </button>
      </div>
    </div>
  );
}
