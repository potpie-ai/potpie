import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphData, GraphNode } from "./types";
import { kindColor, typeColor, UI } from "./theme";

interface Props {
  data: GraphData;
  selectedId: string | null;
  onSelect: (n: GraphNode) => void;
}

const ROW_H = 116; // vertical space per activity
const DIVIDER_H = 40; // day-change divider band
const NB_VGAP = 30; // vertical gap between stacked branch neighbors
const TOP = 24;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function endId(v: string | GraphNode): string {
  return typeof v === "string" ? v : v.id;
}
function dayKey(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}
function dayLabel(ms: number): string {
  const d = new Date(ms);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}
function trim(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

type Neighbor = { node: GraphNode; predicate: string };
type Divider = { kind: "divider"; y: number; label: string };
type Act = {
  kind: "act";
  y: number;
  side: 1 | -1;
  node: GraphNode;
  t: number;
  akind: string;
  neighbors: Neighbor[];
};

export default function Timeline({ data, selectedId, onSelect }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      const next = el.clientWidth;
      setWidth((current) => (current === next ? current : next));
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const branch = Math.min(260, Math.max(150, width * 0.26));
  const svgW = Math.max(width, 2 * (branch + 220));
  const cx = svgW / 2;

  const { rows, height } = useMemo(() => {
    const byId = new Map(data.nodes.map((n) => [n.id, n]));
    const neighborsOf = (id: string): Neighbor[] => {
      const out: Neighbor[] = [];
      for (const e of data.edges) {
        const s = endId(e.source);
        const t = endId(e.target);
        let otherId: string | null = null;
        if (s === id && t !== id) otherId = t;
        else if (t === id && s !== id) otherId = s;
        if (otherId) {
          const nb = byId.get(otherId);
          if (nb) out.push({ node: nb, predicate: e.predicate });
        }
      }
      return out;
    };

    const acts = data.nodes
      .filter((n) => n.type === "Activity" && n.properties?.occurred_at)
      .map((n) => ({ n, t: Date.parse(String(n.properties.occurred_at)) }))
      .filter((a) => !Number.isNaN(a.t))
      .sort((a, b) => b.t - a.t); // newest first; scroll down for older

    const rows: (Divider | Act)[] = [];
    let y = TOP;
    let prevDay: string | null = null;
    acts.forEach(({ n, t }, i) => {
      const dk = dayKey(t);
      if (dk !== prevDay) {
        rows.push({ kind: "divider", y, label: dayLabel(t) });
        y += DIVIDER_H;
        prevDay = dk;
      }
      rows.push({
        kind: "act",
        y: y + ROW_H / 2,
        side: i % 2 === 0 ? 1 : -1,
        node: n,
        t,
        akind: String(n.properties.kind || n.properties.verb_class || "change"),
        neighbors: neighborsOf(n.id),
      });
      y += ROW_H;
    });
    return { rows, height: y + 24 };
  }, [data]);

  const acts = rows.filter((r): r is Act => r.kind === "act");
  if (!acts.length) {
    return (
      <div className="timeline-empty">
        No timeline activities in view. Activities need an <code>occurred_at</code>
        — pick a pot with PR / ticket history, or use the other category filters
        to narrow what branches off each activity.
      </div>
    );
  }
  const spineTop = acts[0].y;
  const spineBot = acts[acts.length - 1].y;

  return (
    <div ref={wrapRef} className="timeline">
      <svg width={svgW} height={height}>
        {/* central time spine */}
        <line x1={cx} x2={cx} y1={spineTop} y2={spineBot} stroke={UI.link} strokeWidth={2} />

        {/* day dividers (the timeframe markers) */}
        {rows
          .filter((r): r is Divider => r.kind === "divider")
          .map((d, i) => (
            <g key={`d${i}`}>
              <line x1={0} x2={svgW} y1={d.y + DIVIDER_H / 2} y2={d.y + DIVIDER_H / 2} stroke="rgba(138,167,155,0.12)" strokeDasharray="3 4" />
              <rect x={cx - 58} y={d.y + 4} width={116} height={20} rx={10} fill={UI.panel2} stroke="rgba(138,167,155,0.25)" />
              <text x={cx} y={d.y + 18} fill="#cfded1" fontSize={11} textAnchor="middle">{d.label}</text>
            </g>
          ))}

        {/* activities + perpendicular alternating branches */}
        {acts.map((a) => {
          const sel = a.node.id === selectedId;
          const bx = cx + a.side * branch; // branch terminal x
          const k = a.neighbors.length;
          return (
            <g key={a.node.id}>
              {/* branch edges out one side, perpendicular to the spine */}
              {a.neighbors.map((nb, j) => {
                const ny = a.y + (j - (k - 1) / 2) * NB_VGAP;
                const nsel = nb.node.id === selectedId;
                return (
                  <g key={nb.node.id + j}>
                    <path
                      d={`M ${cx} ${a.y} H ${cx + a.side * 28} Q ${bx} ${a.y} ${bx} ${ny} `}
                      fill="none"
                      stroke={UI.link}
                      strokeWidth={1}
                    />
                    <text
                      x={cx + a.side * 34}
                      y={a.y + (a.side > 0 ? -5 : -5) + (j - (k - 1) / 2) * (NB_VGAP / 2)}
                      fill="#7d9a8d"
                      fontSize={9}
                      textAnchor={a.side > 0 ? "start" : "end"}
                    >
                      {nb.predicate}
                    </text>
                    <circle
                      cx={bx}
                      cy={ny}
                      r={6}
                      fill={typeColor(nb.node.type)}
                      stroke={nsel ? UI.ring : "rgba(2,26,24,0.6)"}
                      strokeWidth={nsel ? 2 : 1}
                      style={{ cursor: "pointer" }}
                      onClick={() => onSelect(nb.node)}
                    >
                      <title>{`${nb.node.type}: ${nb.node.caption}`}</title>
                    </circle>
                    <text
                      x={bx + a.side * 11}
                      y={ny + 3.5}
                      fill="#b9ccc0"
                      fontSize={11}
                      textAnchor={a.side > 0 ? "start" : "end"}
                      style={{ cursor: "pointer" }}
                      onClick={() => onSelect(nb.node)}
                    >
                      {trim(nb.node.caption, 22)}
                    </text>
                  </g>
                );
              })}

              {/* activity node on the spine */}
              <circle
                cx={cx}
                cy={a.y}
                r={sel ? 10 : 8}
                fill={kindColor(a.akind)}
                stroke={sel ? UI.ring : UI.bg}
                strokeWidth={sel ? 2.5 : 2}
                style={{ cursor: "pointer" }}
                onClick={() => onSelect(a.node)}
              >
                <title>{`${a.node.caption}\n${a.akind}`}</title>
              </circle>

              {/* activity caption + time on the opposite side of the branch */}
              <text
                x={cx - a.side * 18}
                y={a.y - 4}
                fill={sel ? UI.labelBright : "#efe6df"}
                fontSize={12.5}
                fontWeight={600}
                textAnchor={a.side > 0 ? "end" : "start"}
                style={{ cursor: "pointer" }}
                onClick={() => onSelect(a.node)}
              >
                {trim(a.node.caption, 34)}
              </text>
              <text
                x={cx - a.side * 18}
                y={a.y + 12}
                fill={UI.textMuted}
                fontSize={10.5}
                textAnchor={a.side > 0 ? "end" : "start"}
              >
                {a.akind}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
