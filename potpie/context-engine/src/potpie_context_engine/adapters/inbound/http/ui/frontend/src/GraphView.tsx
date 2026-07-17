import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, GraphNode } from "./types";
import { typeColor, UI } from "./theme";
import { ICON_BOX, nodeIcon } from "./icons";

interface Props {
  data: GraphData;
  selectedId: string | null;
  onSelect: (node: GraphNode | null) => void;
  onExpand: (node: GraphNode) => void;
}

const radius = (n: GraphNode) => 4 + Math.min(7, Math.sqrt(n.degree || 0) * 2.2);

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
}: Props) {
  const fgRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const hoverRef = useRef<GraphNode | null>(null);
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
    return {
      nodes: data.nodes.map((n) => ({ ...n, degree: degree[n.id] || 0 })),
      links: data.edges.map((e) => ({ ...e })),
    };
  }, [data]);

  // Stable camera: force-graph re-zooms to 4/cbrt(nodeCount) on EVERY data
  // update while the zoom level still equals its internal "default"
  // (lastSetZoom) — with load/expand/merge each changing the count, the graph
  // looked like it kept resizing. Starting an epsilon off 1 defeats that
  // equality check for good: the camera sits at (an effective) 100% and only
  // moves when the user pans/zooms or uses the controls below.
  const Z100 = 1.000001;
  const [zoomPct, setZoomPct] = useState(100);
  useEffect(() => {
    fgRef.current?.zoom(Z100, 0);
  }, []);
  const resetZoom = () => {
    fgRef.current?.centerAt(0, 0, 300);
    fgRef.current?.zoom(Z100, 300);
  };
  const fitZoom = () => fgRef.current?.zoomToFit?.(300, 60);
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
    const r = radius(node);
    const hovered = node.id === hoverRef.current?.id;
    ctx.beginPath();
    ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
    ctx.fillStyle = typeColor(node.type);
    if (emphasized) {
      // glow renders in device space (unaffected by zoom) — a steady halo
      ctx.save();
      ctx.shadowColor = UI.glow;
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

    if (emphasized || hovered || (node.degree || 0) >= labelMinDegree(scale)) {
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
      ctx.fillStyle = emphasized || hovered ? UI.labelBright : UI.label;
      ctx.fillText(text, node.x!, y);
    }
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
    <div ref={wrapRef} className="graph-canvas">
      <ForceGraph2D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        graphData={graphData as any}
        backgroundColor={UI.bg}
        nodeId="id"
        nodeRelSize={5}
        nodeCanvasObject={(n: any, ctx: any, s: any) => paintNode(n, ctx, s)}
        nodePointerAreaPaint={(n: any, c: any, ctx: any) => pointerArea(n, c, ctx)}
        onRenderFramePre={(ctx: CanvasRenderingContext2D, scale: number) =>
          paintGrid(ctx, scale)
        }
        onRenderFramePost={(ctx: CanvasRenderingContext2D, scale: number) => {
          if (!selectedId) return;
          const n = graphData.nodes.find((x) => x.id === selectedId) as
            | GraphNode
            | undefined;
          if (n && n.x != null && n.y != null) paintNode(n, ctx, scale, true);
        }}
        linkColor={() => UI.link}
        linkWidth={1}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={0.92}
        linkLabel={(l: any) => l.predicate}
        cooldownTicks={120}
        onZoom={({ k }: { k: number }) => {
          const pct = Math.round(k * 100);
          setZoomPct((p) => (p === pct ? p : pct));
        }}
        onNodeHover={(n: any) => {
          hoverRef.current = (n as GraphNode) || null;
        }}
        onNodeClick={(n: any) => onSelect(n as GraphNode)}
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
