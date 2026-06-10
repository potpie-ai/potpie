import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, GraphNode } from "./types";
import { typeColor } from "./theme";

interface Props {
  data: GraphData;
  selectedId: string | null;
  onSelect: (node: GraphNode | null) => void;
  onExpand: (node: GraphNode) => void;
}

const radius = (n: GraphNode) => 4 + Math.min(7, Math.sqrt(n.degree || 0) * 2.2);

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
  const [dims, setDims] = useState({ w: 800, h: 600 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setDims({ w: el.clientWidth, h: el.clientHeight });
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

  useEffect(() => {
    const t = setTimeout(() => fgRef.current?.zoomToFit?.(420, 60), 350);
    return () => clearTimeout(t);
  }, [graphData]);

  const paintNode = (node: GraphNode, ctx: CanvasRenderingContext2D, scale: number) => {
    const r = radius(node);
    const selected = node.id === selectedId;
    ctx.beginPath();
    ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
    ctx.fillStyle = typeColor(node.type);
    ctx.fill();
    if (selected) {
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, r + 3 / scale, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
    }
    if (scale > 1.1 || selected) {
      const fontSize = Math.max(2.5, 11 / scale);
      ctx.font = `${fontSize}px -apple-system, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "rgba(232,234,240,0.92)";
      const label = node.caption || node.key;
      const text = label.length > 26 ? label.slice(0, 25) + "…" : label;
      ctx.fillText(text, node.x!, node.y! + r + 1.5 / scale);
    }
  };

  const pointerArea = (node: GraphNode, color: string, ctx: CanvasRenderingContext2D) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x!, node.y!, radius(node) + 2, 0, 2 * Math.PI);
    ctx.fill();
  };

  return (
    <div ref={wrapRef} className="graph-canvas">
      <ForceGraph2D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        graphData={graphData as any}
        backgroundColor="#11131a"
        nodeId="id"
        nodeRelSize={5}
        nodeCanvasObject={(n: any, ctx: any, s: any) => paintNode(n, ctx, s)}
        nodePointerAreaPaint={(n: any, c: any, ctx: any) => pointerArea(n, c, ctx)}
        linkColor={() => "rgba(150,160,180,0.35)"}
        linkWidth={1}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={0.92}
        linkLabel={(l: any) => l.predicate}
        cooldownTicks={120}
        onNodeClick={(n: any) => onSelect(n as GraphNode)}
        onNodeRightClick={(n: any) => onExpand(n as GraphNode)}
        onBackgroundClick={() => onSelect(null)}
        onNodeDragEnd={(n: any) => {
          n.fx = n.x;
          n.fy = n.y;
        }}
      />
      <div className="graph-hint">
        click a node for details · right-click to expand · scroll to zoom · drag
        to pin
      </div>
    </div>
  );
}
