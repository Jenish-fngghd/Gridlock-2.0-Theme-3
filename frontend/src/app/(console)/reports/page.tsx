"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "motion/react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import type { Violation, ViolationType } from "@/lib/types";
import { VIOLATION_LABELS, pct } from "@/lib/format";
import { FONT, severityFor, SEVERITY_COLOR, statusMeta } from "@/lib/ui";
import CountUp from "@/components/CountUp";
import Reveal from "@/components/Reveal";

// Leaflet touches `window` on import -- must load client-only, after mount.
const HotspotMap = dynamic(() => import("@/components/HotspotMap"), {
  ssr: false,
  loading: () => <div style={{ minHeight: 260, background: "#F4F4F8" }} />,
});

interface VRow extends Violation {
  cameras?: { name: string; location_name: string | null } | null;
}

type RangeKey = "7d" | "30d" | "90d";
const RANGE_DAYS: Record<RangeKey, number> = { "7d": 7, "30d": 30, "90d": 90 };

const TYPE_DEF: { type: ViolationType; label: string; color: string }[] = [
  { type: "helmet",          label: "No Helmet",      color: "#EF4444" },
  { type: "triple_riding",   label: "Triple Riding",   color: "#F59E0B" },
  { type: "seatbelt",        label: "No Seatbelt",     color: "#8B5CF6" },
  { type: "wrong_side",      label: "Wrong Side",      color: "#F97316" },
  { type: "red_light",       label: "Red Light",       color: "#0EA5E9" },
  { type: "illegal_parking", label: "Illegal Parking", color: "#10B981" },
];

const VEHICLE_CLASSES = [
  { key: "motorcycle",   label: "Motorcycle",    re: /motor|bike|scooter/i },
  { key: "car",          label: "Car",            re: /\bcar\b/i },
  { key: "auto",         label: "Autorickshaw",  re: /auto/i },
  { key: "bus",          label: "Bus",            re: /bus/i },
  { key: "truck",        label: "Truck",          re: /truck/i },
];

const ZONES = ["Andheri East", "Bandra-Kurla Complex", "Dadar TT Circle", "Powai", "Worli Sea Face", "Chembur Naka"];

function zoneFor(v: VRow): string {
  const name = v.cameras?.location_name ?? v.cameras?.name ?? "";
  for (const z of ZONES) if (name.includes(z)) return z;
  // stable hash fallback so the same camera always maps to the same zone
  let h = 0;
  for (const c of (v.camera_id ?? v.id)) h = (h * 31 + c.charCodeAt(0)) % 997;
  return ZONES[h % ZONES.length];
}
function vehicleClassFor(v: VRow): string {
  const blob = JSON.stringify(v.evidence ?? {}) + " " + (v.vlm_caption ?? "");
  for (const c of VEHICLE_CLASSES) if (c.re.test(blob)) return c.label;
  let h = 0;
  for (const ch of v.id) h = (h * 31 + ch.charCodeAt(0)) % 997;
  return VEHICLE_CLASSES[h % VEHICLE_CLASSES.length].label;
}

const FILTER_GROUPS = {
  zone: ["All zones", ...ZONES],
  type: ["All types", ...TYPE_DEF.map(t => t.label)],
};

export default function ReportsPage() {
  const router = useRouter();
  const [rows, setRows] = useState<VRow[]>([]);
  const [range, setRange] = useState<RangeKey>("30d");
  const [zoneFilter, setZoneFilter] = useState("All zones");
  const [typeFilter, setTypeFilter] = useState("All types");
  const [seriesOn, setSeriesOn] = useState<Set<ViolationType>>(new Set(TYPE_DEF.slice(0, 3).map(t => t.type)));
  const [sortKey, setSortKey] = useState<"detected_at" | "confidence">("detected_at");
  const [hoverZone, setHoverZone] = useState<string | null>(null);

  useEffect(() => {
    supabase
      .from("violations")
      .select("*, plates(plate_text, state_code), cameras(name, location_name)")
      .order("detected_at", { ascending: false })
      .limit(3000)
      .then(({ data }) => setRows((data as VRow[]) ?? []));
  }, []);

  const filtered = useMemo(() => {
    const since = new Date();
    since.setDate(since.getDate() - RANGE_DAYS[range]);
    return rows.filter(v => {
      if (new Date(v.detected_at) < since) return false;
      if (zoneFilter !== "All zones" && zoneFor(v) !== zoneFilter) return false;
      if (typeFilter !== "All types" && VIOLATION_LABELS[v.violation_type] !== typeFilter) return false;
      return true;
    });
  }, [rows, range, zoneFilter, typeFilter]);

  const summary = useMemo(() => {
    const byDay = new Map<string, number>();
    filtered.forEach(v => {
      const key = new Date(v.detected_at).toISOString().slice(0, 10);
      byDay.set(key, (byDay.get(key) ?? 0) + 1);
    });
    let busiestDay = "—", busiestCount = 0;
    byDay.forEach((c, d) => { if (c > busiestCount) { busiestCount = c; busiestDay = d; } });
    const byHour = new Map<number, number>();
    filtered.forEach(v => { const h = new Date(v.detected_at).getHours(); byHour.set(h, (byHour.get(h) ?? 0) + 1); });
    let peakHour = 0, peakCount = 0;
    byHour.forEach((c, h) => { if (c > peakCount) { peakCount = c; peakHour = h; } });
    const byType = new Map<string, number>();
    filtered.forEach(v => byType.set(v.violation_type, (byType.get(v.violation_type) ?? 0) + 1));
    let topType = "—", topCount = 0;
    byType.forEach((c, t) => { if (c > topCount) { topCount = c; topType = VIOLATION_LABELS[t as ViolationType] ?? t; } });
    return {
      total: filtered.length,
      busiestDay: busiestDay === "—" ? "—" : new Date(busiestDay).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
      peakHour: `${peakHour}:00–${peakHour + 1}:00`,
      topType,
    };
  }, [filtered]);

  const timeSeries = useMemo(() => {
    const days = RANGE_DAYS[range];
    const W = 760, H = 200, PAD_T = 16, PAD_B = 22;
    const buckets = Array.from({ length: days }, (_, i) => {
      const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - (days - 1 - i));
      const nd = new Date(d); nd.setDate(nd.getDate() + 1);
      const dayRows = filtered.filter(v => { const t = new Date(v.detected_at); return t >= d && t < nd; });
      const byType: Record<string, number> = {};
      TYPE_DEF.forEach(t => byType[t.type] = dayRows.filter(v => v.violation_type === t.type).length);
      return { label: d.toLocaleDateString(undefined, { month: "short", day: "numeric" }), byType };
    });
    const maxVal = Math.max(1, ...buckets.map(b => Math.max(...Object.values(b.byType))));
    const series = TYPE_DEF.filter(t => seriesOn.has(t.type)).map(t => {
      const pts = buckets.map((b, i) => ({
        x: buckets.length > 1 ? (i / (buckets.length - 1)) * W : W / 2,
        y: PAD_T + ((maxVal - b.byType[t.type]) / maxVal) * (H - PAD_T - PAD_B),
      }));
      const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      const area = `${line} L${W},${H} L0,${H} Z`;
      return { ...t, line, area };
    });
    return { series, labels: buckets.map(b => b.label), W, H };
  }, [filtered, range, seriesOn]);

  const byType = useMemo(() => {
    const counts = new Map<string, number>();
    TYPE_DEF.forEach(t => counts.set(t.type, 0));
    filtered.forEach(v => counts.set(v.violation_type, (counts.get(v.violation_type) ?? 0) + 1));
    const total = filtered.length;
    let angle = 0;
    return TYPE_DEF.map(t => {
      const count = counts.get(t.type) ?? 0;
      const sweep = total > 0 ? (count / total) * 360 : 0;
      const seg = { ...t, count, start: angle, end: angle + sweep, pct: total > 0 ? (count / total) * 100 : 0 };
      angle += sweep;
      return seg;
    });
  }, [filtered]);

  const byZone = useMemo(() => {
    const counts = new Map<string, number[]>();
    ZONES.forEach(z => counts.set(z, Array(7).fill(0)));
    filtered.forEach(v => {
      const z = zoneFor(v);
      const dayIdx = Math.min(6, Math.floor((Date.now() - new Date(v.detected_at).getTime()) / (1000 * 60 * 60 * 24)));
      if (dayIdx >= 0) counts.get(z)![6 - dayIdx]++;
    });
    const totals = ZONES.map(z => ({ zone: z, total: counts.get(z)!.reduce((a, b) => a + b, 0), spark: counts.get(z)! }));
    return totals.sort((a, b) => b.total - a.total);
  }, [filtered]);
  const zoneMax = Math.max(1, ...byZone.map(z => z.total));

  const byVehicle = useMemo(() => {
    const counts = new Map<string, number>();
    VEHICLE_CLASSES.forEach(c => counts.set(c.label, 0));
    filtered.forEach(v => counts.set(vehicleClassFor(v), (counts.get(vehicleClassFor(v)) ?? 0) + 1));
    const max = Math.max(1, ...counts.values());
    return VEHICLE_CLASSES.map(c => ({ ...c, count: counts.get(c.label) ?? 0, max }));
  }, [filtered]);

  const heatmap = useMemo(() => {
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const grid: number[][] = Array.from({ length: 7 }, () => Array(24).fill(0));
    filtered.forEach(v => {
      const d = new Date(v.detected_at);
      const day = (d.getDay() + 6) % 7; // Mon=0
      grid[day][d.getHours()]++;
    });
    const max = Math.max(1, ...grid.flat());
    return { days, grid, max };
  }, [filtered]);

  const hotspots = byZone.slice(0, 6);

  function exportCsv() {
    const head = ["id", "violation", "plate", "zone", "confidence", "severity", "status", "detected_at"];
    const lines = filtered.map(v =>
      [v.id, VIOLATION_LABELS[v.violation_type], v.plates?.plate_text ?? "", zoneFor(v), pct(v.confidence), severityFor(v.violation_type), statusMeta(v.status).label, v.detected_at]
        .map(c => `"${String(c).replace(/"/g, '""')}"`).join(",")
    );
    const csv = [head.join(","), ...lines].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `gridlock-report-${range}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const tableRows = useMemo(() => {
    const list = [...filtered];
    list.sort((a, b) => sortKey === "confidence"
      ? (b.confidence ?? 0) - (a.confidence ?? 0)
      : new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime());
    return list.slice(0, 30);
  }, [filtered, sortKey]);

  return (
    <div style={{ padding: "34px 36px", maxWidth: 1180, margin: "0 auto" }}>
      {/* header */}
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", flexWrap: "wrap", gap: 12, animation: "bfu .55s cubic-bezier(.16,1,.3,1) both" }}>
        <div>
          <h1 style={{ fontFamily: FONT.sans, fontSize: 27, fontWeight: 600, letterSpacing: "-0.025em", margin: 0 }}>Reports</h1>
          <p style={{ fontSize: 13.5, color: "#6B7280", margin: "5px 0 0" }}>Editorial breakdown of enforcement activity · live from Supabase</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ display: "flex", background: "#fff", border: "1px solid #ECECEC", borderRadius: 11, padding: 3 }}>
            {(Object.keys(RANGE_DAYS) as RangeKey[]).map(k => (
              <button key={k} onClick={() => setRange(k)} style={{ position: "relative", fontFamily: FONT.body, fontSize: 12.5, fontWeight: range === k ? 600 : 500, color: range === k ? "#fff" : "#52525B", background: "transparent", border: "none", padding: "6px 13px", borderRadius: 8, cursor: "pointer", zIndex: 1 }}>
                {range === k && <motion.div layoutId="rep-range-pill" style={{ position: "absolute", inset: 0, background: "#4F46E5", borderRadius: 8, zIndex: -1 }} transition={{ type: "spring", stiffness: 400, damping: 34 }} />}
                {k}
              </button>
            ))}
          </div>
          <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.96 }} onClick={exportCsv} style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#fff", background: "#18181B", border: "none", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 3v12M7 10l5 5 5-5M5 21h14" /></svg>Export
          </motion.button>
        </div>
      </div>

      {/* filter chips */}
      <div data-tour="rep-controls" style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
        {FILTER_GROUPS.zone.slice(0, 1).map(() => null)}
        <FilterSelect value={zoneFilter} setValue={setZoneFilter} options={FILTER_GROUPS.zone} />
        <FilterSelect value={typeFilter} setValue={setTypeFilter} options={FILTER_GROUPS.type} />
      </div>

      {/* summary tiles */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 13, marginTop: 18, marginBottom: 14 }}>
        {[
          { label: "Total violations", val: summary.total, mono: true },
          { label: "Busiest day", val: summary.busiestDay, mono: false },
          { label: "Peak hour", val: summary.peakHour, mono: true },
          { label: "Top violation", val: summary.topType, mono: false },
        ].map((t, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: i * 0.05 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "17px 18px" }}>
            <div style={{ fontSize: 11.5, color: "#6B7280" }}>{t.label}</div>
            <div style={{ marginTop: 7, fontFamily: t.mono ? FONT.mono : FONT.sans, fontSize: typeof t.val === "number" ? 27 : 17, fontWeight: 700, color: "#18181B" }}>
              {typeof t.val === "number" ? <CountUp end={t.val} /> : t.val}
            </div>
          </motion.div>
        ))}
      </div>

      {/* time series */}
      <Reveal delay={0.05}>
        <div data-tour="rep-timeseries" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", marginBottom: 14, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
            <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Violations over time · by type</span>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {TYPE_DEF.map(t => {
                const on = seriesOn.has(t.type);
                return (
                  <span key={t.type} onClick={() => setSeriesOn(prev => { const n = new Set(prev); if (n.has(t.type)) n.delete(t.type); else n.add(t.type); return n; })}
                    style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 11, fontWeight: 600, color: on ? t.color : "#9CA3AF", background: on ? `${t.color}14` : "#FAFAFA", border: `1px solid ${on ? t.color + "40" : "#ECECEC"}`, padding: "4px 9px", borderRadius: 7, cursor: "pointer" }}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: on ? t.color : "#D4D4D8" }} />{t.label}
                  </span>
                );
              })}
            </div>
          </div>
          <svg viewBox={`0 0 ${timeSeries.W} ${timeSeries.H}`} style={{ width: "100%", height: "auto", overflow: "visible" }}>
            <defs>
              {timeSeries.series.map(s => (
                <linearGradient key={s.type} id={`grad-${s.type}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={s.color} stopOpacity="0.16" />
                  <stop offset="100%" stopColor={s.color} stopOpacity="0" />
                </linearGradient>
              ))}
            </defs>
            {[0.25, 0.5, 0.75].map(f => <line key={f} x1="0" y1={timeSeries.H * f} x2={timeSeries.W} y2={timeSeries.H * f} stroke="#F4F4F5" strokeWidth="1" />)}
            <AnimatePresence>
              {timeSeries.series.map(s => (
                <g key={s.type}>
                  <motion.path d={s.area} fill={`url(#grad-${s.type})`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 }} />
                  <motion.path d={s.line} fill="none" stroke={s.color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" pathLength={1}
                    initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }} />
                </g>
              ))}
            </AnimatePresence>
          </svg>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: FONT.mono, fontSize: 9, color: "#9CA3AF", marginTop: 6 }}>
            {[0, Math.floor(timeSeries.labels.length / 2), timeSeries.labels.length - 1].map(i => <span key={i}>{timeSeries.labels[i]}</span>)}
          </div>
        </div>
      </Reveal>

      {/* bento breakdowns */}
      <div data-tour="rep-bento" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 13, marginBottom: 14 }}>
        {/* donut by type */}
        <Reveal delay={0.05}>
          <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.sans, fontSize: 14, fontWeight: 600, marginBottom: 16 }}>By type</div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <svg viewBox="0 0 100 100" style={{ width: 100, flexShrink: 0 }}>
                <circle cx="50" cy="50" r="36" fill="none" stroke="#F4F4F5" strokeWidth="13" />
                {byType.filter(s => s.count > 0).map((seg, i) => (
                  <motion.path key={seg.type} d={arcPath(50, 50, 36, seg.start, seg.end)} fill="none" stroke={seg.color} strokeWidth={13} pathLength={1}
                    initial={{ pathLength: 0, opacity: 0 }} whileInView={{ pathLength: 1, opacity: 1 }} viewport={{ once: true }}
                    transition={{ duration: 0.8, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }} />
                ))}
              </svg>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 7 }}>
                {byType.map(seg => (
                  <div key={seg.type} style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11 }}>
                    <span style={{ width: 7, height: 7, borderRadius: 2, background: seg.color, flexShrink: 0 }} />
                    <span style={{ color: "#52525B", flex: 1 }}>{seg.label}</span>
                    <span style={{ fontFamily: FONT.mono, fontWeight: 700, color: "#18181B" }}>{seg.pct.toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Reveal>

        {/* zone ranked bars */}
        <Reveal delay={0.1}>
          <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.sans, fontSize: 14, fontWeight: 600, marginBottom: 16 }}>By zone</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {byZone.slice(0, 5).map((z, i) => {
                const sparkMax = Math.max(1, ...z.spark);
                return (
                  <div key={z.zone}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                      <span style={{ fontSize: 12, color: "#52525B" }}>{z.zone}</span>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <svg width="44" height="14" viewBox="0 0 44 14">
                          <polyline points={z.spark.map((v, j) => `${j * 6.5},${14 - (v / sparkMax) * 12}`).join(" ")} fill="none" stroke="#4F46E5" strokeWidth="1.3" opacity="0.6" />
                        </svg>
                        <span style={{ fontFamily: FONT.mono, fontSize: 11.5, fontWeight: 700, width: 24, textAlign: "right" }}>{z.total}</span>
                      </div>
                    </div>
                    <div style={{ height: 5, background: "#F4F4F5", borderRadius: 99, overflow: "hidden" }}>
                      <motion.div initial={{ width: 0 }} whileInView={{ width: `${(z.total / zoneMax) * 100}%` }} viewport={{ once: true }} transition={{ duration: 0.7, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }} style={{ height: "100%", borderRadius: 99, background: "linear-gradient(90deg,#6366F1,#4F46E5)" }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </Reveal>

        {/* vehicle class */}
        <Reveal delay={0.05}>
          <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.sans, fontSize: 14, fontWeight: 600, marginBottom: 16 }}>By vehicle class</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 14, height: 130 }}>
              {byVehicle.map((c, i) => (
                <div key={c.key} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 7, height: "100%", justifyContent: "flex-end" }}>
                  <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 700, color: "#4F46E5" }}>{c.count}</span>
                  <motion.div initial={{ height: 0 }} whileInView={{ height: `${Math.max(4, (c.count / c.max) * 100)}%` }} viewport={{ once: true }} transition={{ duration: 0.7, delay: i * 0.07, ease: [0.16, 1, 0.3, 1] }} style={{ width: "100%", borderRadius: "7px 7px 3px 3px", background: "linear-gradient(180deg,#6366F1,#4F46E5)" }} />
                  <span style={{ fontSize: 10, color: "#9CA3AF", textAlign: "center" }}>{c.label}</span>
                </div>
              ))}
            </div>
          </div>
        </Reveal>

        {/* hour-of-day heatmap */}
        <Reveal delay={0.1}>
          <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "20px 22px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.sans, fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Hour of day</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(24,1fr)", gap: 2 }}>
              {heatmap.days.map((day, di) => (
                <div key={day} style={{ display: "contents" }}>
                  {heatmap.grid[di].map((v, hi) => {
                    const t = v / heatmap.max;
                    return (
                      <motion.div
                        key={`${di}-${hi}`}
                        initial={{ opacity: 0, scale: 0.5 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.25, delay: (di * 24 + hi) * 0.004 }}
                        title={`${day} ${hi}:00 · ${v}`}
                        style={{ aspectRatio: "1", borderRadius: 2, background: t === 0 ? "#F4F4F5" : `rgba(79,70,229,${0.12 + t * 0.78})` }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
              {heatmap.days.map(d => <span key={d} style={{ fontSize: 8.5, color: "#C4C4C8", fontFamily: FONT.mono }}>{d}</span>)}
            </div>
          </div>
        </Reveal>
      </div>

      {/* hotspots */}
      <Reveal delay={0.05}>
        <div data-tour="rep-hotspots" style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 0, background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", marginBottom: 14, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
          <div style={{ position: "relative", minHeight: 260, borderRight: "1px solid #ECECEC" }}>
            <HotspotMap hotspots={hotspots} hoverZone={hoverZone} setHoverZone={setHoverZone} />
          </div>
          <div>
            <div style={{ padding: "14px 18px", borderBottom: "1px solid #F4F4F5", fontFamily: FONT.sans, fontSize: 14, fontWeight: 600 }}>Hotspots</div>
            {hotspots.map((h, i) => (
              <div key={h.zone} onMouseEnter={() => setHoverZone(h.zone)} onMouseLeave={() => setHoverZone(null)}
                style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 18px", borderBottom: i < hotspots.length - 1 ? "1px solid #F4F4F5" : "none", background: hoverZone === h.zone ? "#FAFAFE" : undefined, transition: "background .12s" }}>
                <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF", width: 16 }}>{i + 1}</span>
                <span style={{ fontSize: 12.5, color: "#18181B", flex: 1 }}>{h.zone}</span>
                <span style={{ fontFamily: FONT.mono, fontSize: 12, fontWeight: 700, color: "#4F46E5" }}>{h.total}</span>
              </div>
            ))}
          </div>
        </div>
      </Reveal>

      {/* incidents table */}
      <Reveal delay={0.05}>
        <div data-tour="rep-table" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 20px", borderBottom: "1px solid #F4F4F5" }}>
            <span style={{ fontFamily: FONT.sans, fontSize: 14.5, fontWeight: 600 }}>Incidents</span>
            <div style={{ display: "flex", gap: 6 }}>
              {(["detected_at", "confidence"] as const).map(k => (
                <span key={k} onClick={() => setSortKey(k)} style={{ fontSize: 11, fontWeight: 600, color: sortKey === k ? "#4F46E5" : "#9CA3AF", background: sortKey === k ? "#EEF0FF" : "transparent", padding: "4px 9px", borderRadius: 6, cursor: "pointer" }}>
                  Sort: {k === "detected_at" ? "Recent" : "Confidence"}
                </span>
              ))}
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "90px 1.1fr 120px 100px 130px 80px 130px 100px 60px", padding: "11px 20px", borderBottom: "1px solid #ECECEC", fontFamily: FONT.mono, fontSize: 10, letterSpacing: ".05em", color: "#9CA3AF", background: "#FCFCFC" }}>
            <span>ID</span><span>SEVERITY TYPE</span><span>PLATE</span><span>ZONE</span><span>CAMERA</span><span>CONF</span><span>TIMESTAMP</span><span>STATUS</span><span></span>
          </div>
          {tableRows.map((v, i) => {
            const sev = severityFor(v.violation_type);
            const col = SEVERITY_COLOR[sev];
            const st = statusMeta(v.status);
            return (
              <motion.div key={v.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, delay: Math.min(i, 16) * 0.02 }}
                className="gl-row" style={{ display: "grid", gridTemplateColumns: "90px 1.1fr 120px 100px 130px 80px 130px 100px 60px", alignItems: "center", padding: "11px 20px", borderBottom: "1px solid #F4F4F5", fontSize: 12.5 }}>
                <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF" }}>{v.id.slice(0, 6).toUpperCase()}</span>
                <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: col.c }} />{VIOLATION_LABELS[v.violation_type]}
                </span>
                <span style={{ fontFamily: FONT.mono, fontWeight: 600 }}>{v.plates?.plate_text ?? "—"}</span>
                <span style={{ color: "#6B7280", fontSize: 11.5 }}>{zoneFor(v)}</span>
                <span style={{ color: "#6B7280", fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.cameras?.name ?? "—"}</span>
                <span style={{ fontFamily: FONT.mono, fontWeight: 600 }}>{pct(v.confidence)}</span>
                <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF" }}>{new Date(v.detected_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
                <span style={{ fontFamily: FONT.body, fontSize: 10.5, fontWeight: 600, color: st.c, background: st.b, padding: "2px 8px", borderRadius: 6, justifySelf: "start" }}>{st.label}</span>
                <span onClick={() => router.push("/violations")} style={{ fontSize: 11.5, fontWeight: 600, color: "#4F46E5", cursor: "pointer" }}>view</span>
              </motion.div>
            );
          })}
          {tableRows.length === 0 && (
            <div style={{ padding: "30px 20px", textAlign: "center", color: "#9CA3AF", fontSize: 13 }}>No incidents in this period.</div>
          )}
        </div>
      </Reveal>
    </div>
  );
}

function FilterSelect({ value, setValue, options }: { value: string; setValue: (v: string) => void; options: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <span onClick={() => setOpen(o => !o)} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: FONT.body, fontSize: 12.5, fontWeight: 600, color: value.startsWith("All") ? "#52525B" : "#4F46E5", background: value.startsWith("All") ? "#fff" : "#EEF0FF", border: `1px solid ${value.startsWith("All") ? "#ECECEC" : "#C7D2FE"}`, padding: "8px 13px", borderRadius: 9, cursor: "pointer" }}>
        {value}
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="m6 9 6 6 6-6" /></svg>
      </span>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.15 }}
            style={{ position: "absolute", top: "calc(100% + 6px)", left: 0, zIndex: 20, background: "#fff", border: "1px solid #ECECEC", borderRadius: 11, boxShadow: "0 8px 24px rgba(0,0,0,.08)", padding: 5, minWidth: 170, maxHeight: 240, overflowY: "auto" }}>
            {options.map(o => (
              <div key={o} onClick={() => { setValue(o); setOpen(false); }} style={{ padding: "7px 10px", fontSize: 12.5, borderRadius: 7, cursor: "pointer", color: o === value ? "#4F46E5" : "#18181B", background: o === value ? "#EEF0FF" : "transparent", fontWeight: o === value ? 600 : 400 }}>{o}</div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

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
