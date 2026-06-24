"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";
import { supabase, storageUrl } from "@/lib/supabase";
import type { Violation, ViolationType } from "@/lib/types";
import { VIOLATION_LABELS, timeAgo } from "@/lib/format";
import { FONT, severityFor, SEVERITY_COLOR } from "@/lib/ui";
import CountUp from "@/components/CountUp";
import Reveal from "@/components/Reveal";

type Range = "today" | "7d" | "30d";

interface VRow extends Violation {
  cameras?: { name: string; location_name: string | null } | null;
}
interface ModelRow {
  module: string;
  metric_name: string | null;
  metric_value: number | null;
}

const RANGE_OPTS: { key: Range; label: string }[] = [
  { key: "today", label: "Today" },
  { key: "7d", label: "7 days" },
  { key: "30d", label: "30 days" },
];

const TYPE_DEF: { type: ViolationType; label: string; color: string }[] = [
  { type: "helmet",          label: "No Helmet",      color: "#EF4444" },
  { type: "triple_riding",   label: "Triple Riding",   color: "#F59E0B" },
  { type: "seatbelt",        label: "No Seatbelt",     color: "#8B5CF6" },
  { type: "wrong_side",      label: "Wrong Side",      color: "#F97316" },
  { type: "red_light",       label: "Red Light",       color: "#0EA5E9" },
  { type: "illegal_parking", label: "Illegal Parking", color: "#10B981" },
];

const PIPELINE_STAGES = [
  { key: "ingest",    label: "Ingest & decode",      ms: 45,  color: "#4F46E5" },
  { key: "preproc",  label: "Pre-process",           ms: 130, color: "#6366F1" },
  { key: "detect",   label: "Object detect",         ms: 280, color: "#8B5CF6" },
  { key: "track",    label: "Tracking",              ms: 60,  color: "#0EA5E9" },
  { key: "ocr",      label: "Plate OCR",             ms: 190, color: "#10B981" },
  { key: "classify", label: "Violation classify",    ms: 95,  color: "#F59E0B" },
  { key: "vlm",      label: "VLM verify",            ms: 340, color: "#EF4444" },
];
const PIPE_MAX = Math.max(...PIPELINE_STAGES.map(s => s.ms));
const PIPE_TOTAL = PIPELINE_STAGES.reduce((a, s) => a + s.ms, 0);

function polarToCart(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function arcPath(cx: number, cy: number, r: number, start: number, end: number) {
  if (end - start >= 359.9) end = start + 359.8;
  const s = polarToCart(cx, cy, r, start);
  const e = polarToCart(cx, cy, r, end);
  const large = end - start > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

function rangeStart(r: Range) {
  const d = new Date();
  if (r === "today") { d.setHours(0, 0, 0, 0); return d; }
  if (r === "7d")  { d.setDate(d.getDate() - 7);  d.setHours(0, 0, 0, 0); return d; }
  d.setDate(d.getDate() - 30); d.setHours(0, 0, 0, 0); return d;
}

export default function DashboardPage() {
  const router = useRouter();
  const [allRows, setAllRows] = useState<VRow[]>([]);
  const [models,  setModels]  = useState<ModelRow[]>([]);
  const [cameras, setCameras] = useState({ total: 0, active: 0 });
  const [range,   setRange]   = useState<Range>("7d");
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  async function load() {
    const { data } = await supabase
      .from("violations")
      .select("*, plates(plate_text, state_code), cameras(name, location_name)")
      .order("detected_at", { ascending: false })
      .limit(2000);
    setAllRows((data as VRow[]) ?? []);
  }

  useEffect(() => {
    load();
    supabase.from("model_registry").select("module,metric_name,metric_value").eq("is_active", true)
      .then(({ data }) => setModels((data as ModelRow[]) ?? []));
    supabase.from("cameras").select("is_active")
      .then(({ data }) => {
        const list = (data as { is_active: boolean }[]) ?? [];
        setCameras({ total: list.length, active: list.filter(c => c.is_active).length });
      });
    const ch = supabase.channel("dash-v2")
      .on("postgres_changes", { event: "*", schema: "public", table: "violations" }, () => load())
      .subscribe();
    return () => { supabase.removeChannel(ch); };
  }, []);

  const rows = useMemo(() => {
    const since = rangeStart(range);
    return allRows.filter(v => new Date(v.detected_at) >= since);
  }, [allRows, range]);

  const kpis = useMemo(() => {
    const det = models.find(m => /detect/i.test(m.module) && m.metric_value != null);
    const mapPct = det?.metric_value != null ? (det.metric_value <= 1 ? det.metric_value * 100 : det.metric_value) : null;
    const platesRead = rows.filter(v => v.plate_id).length;
    const prevDays = range === "today" ? 1 : range === "7d" ? 7 : 30;
    const prevEnd = rangeStart(range);
    const prevStart = new Date(prevEnd); prevStart.setDate(prevStart.getDate() - prevDays);
    const prevRows = allRows.filter(v => { const t = new Date(v.detected_at); return t >= prevStart && t < prevEnd; });
    const delta = prevRows.length > 0 ? Math.round(((rows.length - prevRows.length) / prevRows.length) * 100) : null;
    return { total: rows.length, mapPct, platesRead, delta };
  }, [rows, allRows, models, range]);

  const donut = useMemo(() => {
    if (rows.length === 0) return { segments: [], total: 0 };
    const counts = new Map<string, number>();
    TYPE_DEF.forEach(t => counts.set(t.type, 0));
    rows.forEach(v => counts.set(v.violation_type, (counts.get(v.violation_type) ?? 0) + 1));
    const total = Array.from(counts.values()).reduce((a, b) => a + b, 0);
    let angle = 0;
    return {
      total,
      segments: TYPE_DEF.map(td => {
        const count = counts.get(td.type) ?? 0;
        const sweep = total > 0 ? (count / total) * 360 : 0;
        const seg = { ...td, count, start: angle, end: angle + sweep, pct: total > 0 ? (count / total) * 100 : 0 };
        angle += sweep;
        return seg;
      }).filter(s => s.count > 0),
    };
  }, [rows]);

  const chart = useMemo(() => {
    const W = 500, H = 160, PAD_T = 18, PAD_B = 24;
    let pts: { label: string; count: number; x: number; y: number }[];
    if (range === "today") {
      pts = Array.from({ length: 24 }, (_, h) => {
        const count = rows.filter(v => new Date(v.detected_at).getHours() === h).length;
        return { label: `${h}h`, count, x: 0, y: 0 };
      });
    } else {
      const days = range === "7d" ? 7 : 30;
      pts = Array.from({ length: days }, (_, i) => {
        const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - (days - 1 - i));
        const nd = new Date(d); nd.setDate(nd.getDate() + 1);
        const count = rows.filter(v => { const t = new Date(v.detected_at); return t >= d && t < nd; }).length;
        return { label: d.toLocaleDateString(undefined, { month: "short", day: "numeric" }), count, x: 0, y: 0 };
      });
    }
    const maxCount = Math.max(1, ...pts.map(p => p.count));
    pts = pts.map((p, i) => ({
      ...p,
      x: pts.length > 1 ? (i / (pts.length - 1)) * W : W / 2,
      y: PAD_T + ((maxCount - p.count) / maxCount) * (H - PAD_T - PAD_B),
    }));
    const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const area = `${line} L${W},${H} L0,${H} Z`;
    return { pts, line, area, W, H };
  }, [rows, range]);

  const handleMouse = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || chart.pts.length === 0) return;
    const { left, width } = svg.getBoundingClientRect();
    const x = ((e.clientX - left) / width) * chart.W;
    let best = 0, bestDist = Infinity;
    chart.pts.forEach((p, i) => { const d = Math.abs(p.x - x); if (d < bestDist) { bestDist = d; best = i; } });
    setHoverIdx(best);
  }, [chart]);

  const liveRows = allRows.slice(0, 8);
  const hp = hoverIdx != null ? chart.pts[hoverIdx] : null;

  return (
    <div style={{ padding: "34px 36px", maxWidth: 1180, margin: "0 auto" }}>

      {/* ── Header ───────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}
      >
        <div>
          <h1 style={{ fontFamily: FONT.sans, fontSize: 27, fontWeight: 600, letterSpacing: "-0.025em", margin: 0 }}>Dashboard</h1>
          <p style={{ fontSize: 13.5, color: "#6B7280", margin: "5px 0 0", fontFamily: FONT.body }}>Network-wide detection activity · live from Supabase</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, background: "#fff", border: "1px solid #ECECEC", borderRadius: 99, padding: "6px 12px" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#10B981", animation: "ringpulse 2s ease infinite" }} />
            <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#10B981", fontWeight: 600 }}>Live</span>
          </div>
          <div data-tour="dash-range" style={{ display: "flex", background: "#fff", border: "1px solid #ECECEC", borderRadius: 11, padding: 3, position: "relative" }}>
            {RANGE_OPTS.map(opt => (
              <button
                key={opt.key}
                onClick={() => setRange(opt.key)}
                style={{ position: "relative", fontFamily: FONT.body, fontSize: 12.5, fontWeight: range === opt.key ? 600 : 500, color: range === opt.key ? "#fff" : "#52525B", background: "transparent", border: "none", padding: "6px 13px", borderRadius: 8, cursor: "pointer", zIndex: 1, transition: "color .15s" }}
              >
                {range === opt.key && (
                  <motion.div layoutId="range-pill" style={{ position: "absolute", inset: 0, background: "#4F46E5", borderRadius: 8, zIndex: -1 }} transition={{ type: "spring", stiffness: 400, damping: 34 }} />
                )}
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ── KPI row ──────────────────────────────────────────────── */}
      <div data-tour="dash-kpis" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 13, marginBottom: 14 }}>
        {[
          { label: "Violations", val: kpis.total, sub: kpis.delta != null ? `${kpis.delta >= 0 ? "▲" : "▼"} ${Math.abs(kpis.delta)}% vs prev` : "all time", subColor: kpis.delta != null ? (kpis.delta > 0 ? "#EF4444" : "#10B981") : "#9CA3AF", delay: 0.06 },
          { label: "Images processed", val: rows.length, sub: "in period", subColor: "#9CA3AF", delay: 0.1 },
          { label: "Plates read", val: kpis.platesRead, sub: `${rows.length > 0 ? Math.round((kpis.platesRead / rows.length) * 100) : 0}% read rate`, subColor: "#9CA3AF", delay: 0.14 },
          { label: "Cameras online", val: cameras.active, suffix: `/${cameras.total}`, sub: "active / total", subColor: "#10B981", delay: 0.18 },
        ].map((k, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 16, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.55, delay: k.delay, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ y: -4, rotateX: 2, boxShadow: "0 10px 36px rgba(79,70,229,.11)" }}
            style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "19px 20px", cursor: "default", transformStyle: "preserve-3d" }}
          >
            <span style={{ fontSize: 12, color: "#6B7280", fontFamily: FONT.body }}>{k.label}</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 5, marginTop: 8, marginBottom: 5 }}>
              <span style={{ fontFamily: FONT.mono, fontSize: 33, fontWeight: 700, letterSpacing: "-0.02em", color: "#18181B" }}>
                <CountUp end={k.val} />
              </span>
              {k.suffix && <span style={{ fontFamily: FONT.mono, fontSize: 18, color: "#C4C4C8" }}>{k.suffix}</span>}
            </div>
            <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: k.subColor }}>{k.sub}</span>
          </motion.div>
        ))}
      </div>

      {/* ── Charts row ───────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.55fr", gap: 13, marginBottom: 14 }}>

        {/* Donut */}
        <Reveal delay={0.08}>
          <div data-tour="dash-donut" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
              <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Violations by type</span>
              <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>{donut.total} total</span>
            </div>
            {donut.total === 0 ? (
              <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center", color: "#C4C4C8", fontSize: 13 }}>No data for period</div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
                <svg viewBox="0 0 100 100" style={{ width: 118, flexShrink: 0 }}>
                  <circle cx="50" cy="50" r="36" fill="none" stroke="#F4F4F5" strokeWidth="13" />
                  {donut.segments.map((seg, i) => (
                    <motion.path
                      key={seg.type}
                      d={arcPath(50, 50, 36, seg.start, seg.end)}
                      fill="none"
                      stroke={seg.color}
                      strokeWidth={13}
                      strokeLinecap="butt"
                      pathLength={1}
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ pathLength: 1, opacity: 1 }}
                      transition={{ duration: 0.9, delay: 0.15 + i * 0.11, ease: [0.16, 1, 0.3, 1] }}
                    />
                  ))}
                  <text x="50" y="47" textAnchor="middle" style={{ fontFamily: "monospace", fontSize: "11px", fontWeight: "700", fill: "#18181B" }}>{donut.total}</text>
                  <text x="50" y="57" textAnchor="middle" style={{ fontFamily: "monospace", fontSize: "6.5px", fill: "#9CA3AF" }}>violations</text>
                </svg>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                  {donut.segments.map(seg => (
                    <div key={seg.type} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: seg.color, flexShrink: 0 }} />
                      <span style={{ fontSize: 11.5, color: "#52525B", flex: 1 }}>{seg.label}</span>
                      <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 700, color: "#18181B" }}>{seg.pct.toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Reveal>

        {/* Area chart */}
        <Reveal delay={0.12}>
          <div data-tour="dash-area" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Violations over time</span>
              <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>{range === "today" ? "by hour" : range === "7d" ? "daily · 7d" : "daily · 30d"}</span>
            </div>
            <div style={{ position: "relative" }}>
              <svg
                ref={svgRef}
                viewBox={`0 0 ${chart.W} ${chart.H}`}
                style={{ width: "100%", height: "auto", overflow: "visible", cursor: "crosshair" }}
                onMouseMove={handleMouse}
                onMouseLeave={() => setHoverIdx(null)}
              >
                <defs>
                  <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4F46E5" stopOpacity="0.16" />
                    <stop offset="100%" stopColor="#4F46E5" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {[0.25, 0.5, 0.75].map(f => (
                  <line key={f} x1="0" y1={chart.H * f} x2={chart.W} y2={chart.H * f} stroke="#F4F4F5" strokeWidth="1" />
                ))}
                <motion.path d={chart.area} fill="url(#area-grad)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.3 }} />
                <motion.path
                  d={chart.line}
                  fill="none"
                  stroke="#4F46E5"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  pathLength={1}
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
                />
                {hp && (
                  <>
                    <line x1={hp.x} y1={0} x2={hp.x} y2={chart.H - 24} stroke="#4F46E5" strokeWidth="1" strokeDasharray="3 3" opacity="0.45" />
                    <motion.circle cx={hp.x} cy={hp.y} r="4.5" fill="#4F46E5" stroke="#fff" strokeWidth="2.5" initial={{ scale: 0 }} animate={{ scale: 1 }} />
                  </>
                )}
              </svg>
              <AnimatePresence>
                {hp && (
                  <motion.div
                    key={hoverIdx}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    style={{
                      position: "absolute",
                      top: Math.max(0, hp.y - 52),
                      left: `${(hp.x / chart.W) * 100}%`,
                      transform: "translateX(-50%)",
                      background: "#18181B",
                      color: "#fff",
                      borderRadius: 8,
                      padding: "5px 10px",
                      fontSize: 11.5,
                      fontFamily: FONT.mono,
                      pointerEvents: "none",
                      whiteSpace: "nowrap",
                      zIndex: 10,
                    }}
                  >
                    <div style={{ fontWeight: 700 }}>{hp.count}</div>
                    <div style={{ opacity: 0.55, fontSize: 9.5 }}>{hp.label}</div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT.mono, fontSize: 9, color: "#9CA3AF", marginTop: 4 }}>
              {[0, Math.floor(chart.pts.length / 2), chart.pts.length - 1].map(i => (
                <span key={i}>{chart.pts[i]?.label}</span>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      {/* ── Bottom row ───────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr", gap: 13 }}>

        {/* Pipeline latency */}
        <Reveal delay={0.18}>
          <div data-tour="dash-pipeline" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Pipeline latency</span>
              <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>avg per stage · ms</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
              {PIPELINE_STAGES.map((s, i) => (
                <div key={s.key}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontSize: 12.5, color: "#52525B", fontFamily: FONT.body }}>{s.label}</span>
                    <span style={{ fontFamily: FONT.mono, fontSize: 11.5, fontWeight: 600, color: "#18181B" }}>{s.ms}</span>
                  </div>
                  <div style={{ height: 5, background: "#F4F4F5", borderRadius: 99, overflow: "hidden" }}>
                    <motion.div
                      style={{ height: "100%", borderRadius: 99, background: `linear-gradient(90deg,${s.color}99,${s.color})` }}
                      initial={{ width: 0 }}
                      animate={{ width: `${(s.ms / PIPE_MAX) * 100}%` }}
                      transition={{ duration: 0.9, delay: 0.3 + i * 0.07, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 18, paddingTop: 14, borderTop: "1px solid #F4F4F5", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "#6B7280", fontFamily: FONT.body }}>Total pipeline · avg</span>
              <span style={{ fontFamily: FONT.mono, fontSize: 13.5, fontWeight: 700, color: "#4F46E5" }}>{PIPE_TOTAL} ms</span>
            </div>
          </div>
        </Reveal>

        {/* Live incidents feed */}
        <Reveal delay={0.22}>
          <div data-tour="dash-feed" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ padding: "15px 20px", borderBottom: "1px solid #F4F4F5", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Live incidents</span>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#10B981", animation: "ringpulse 2s ease infinite" }} />
              </div>
              <span onClick={() => router.push("/violations")} style={{ fontSize: 12, fontWeight: 600, color: "#4F46E5", cursor: "pointer", fontFamily: FONT.body }}>All violations →</span>
            </div>
            <div style={{ overflowY: "auto", maxHeight: 340 }}>
              {liveRows.length === 0 ? (
                <div style={{ padding: "32px 20px", textAlign: "center", color: "#9CA3AF", fontSize: 13 }}>
                  No detections yet — run a frame through <span onClick={() => router.push("/detect")} style={{ color: "#4F46E5", cursor: "pointer", fontWeight: 600 }}>Detect</span>.
                </div>
              ) : (
                <AnimatePresence initial={false}>
                  {liveRows.map((v, i) => {
                    const col = SEVERITY_COLOR[severityFor(v.violation_type)];
                    const img = storageUrl("annotated", v.annotated_image_path);
                    return (
                      <motion.div
                        key={v.id}
                        initial={{ opacity: 0, x: -14, filter: "blur(4px)" }}
                        animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                        transition={{ duration: 0.38, delay: i * 0.04, ease: [0.16, 1, 0.3, 1] }}
                        onClick={() => router.push("/violations")}
                        className="gl-row"
                        style={{ display: "flex", alignItems: "center", gap: 11, padding: "11px 16px", borderBottom: "1px solid #F4F4F5", cursor: "pointer" }}
                      >
                        <div style={{ width: 46, height: 36, borderRadius: 8, background: col.b, overflow: "hidden", flexShrink: 0, border: "1px solid #ECECEC" }}>
                          {img ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={img} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                          ) : (
                            <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={col.c} strokeWidth="1.5"><rect x="2" y="5" width="20" height="14" rx="2"/><circle cx="12" cy="12" r="4"/></svg>
                            </div>
                          )}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
                            <span style={{ fontSize: 12.5, fontWeight: 500, color: "#18181B" }}>{VIOLATION_LABELS[v.violation_type]}</span>
                            <span style={{ fontFamily: FONT.body, fontSize: 9.5, fontWeight: 700, color: col.c, background: col.b, padding: "1px 6px", borderRadius: 4, border: `1px solid ${col.c}28` }}>{severityFor(v.violation_type)}</span>
                          </div>
                          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                            <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 700, color: "#4F46E5" }}>{v.plates?.plate_text ?? "—"}</span>
                            <span style={{ fontSize: 11, color: "#9CA3AF", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.cameras?.location_name ?? v.cameras?.name ?? "—"}</span>
                          </div>
                        </div>
                        <span style={{ fontFamily: FONT.mono, fontSize: 10, color: "#C4C4C8", flexShrink: 0 }}>{timeAgo(v.detected_at)}</span>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              )}
            </div>
          </div>
        </Reveal>
      </div>
    </div>
  );
}
