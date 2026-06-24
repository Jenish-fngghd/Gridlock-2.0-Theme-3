"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import { supabase, storageUrl } from "@/lib/supabase";
import type { Violation } from "@/lib/types";
import { VIOLATION_LABELS, pct, timeAgo } from "@/lib/format";
import { reviewViolation } from "@/lib/api";
import { FONT, severityFor, SEVERITY_COLOR, statusMeta, boxColor, prettyClass, type Severity } from "@/lib/ui";

interface VRow extends Violation {
  cameras?: { name: string; location_name: string | null } | null;
  model_module?: string | null;
}
interface DetRow {
  id: string;
  job_id: string;
  class_label: string;
  confidence: number | null;
  bbox: { x: number; y: number; w: number; h: number } | null;
}
interface AuditRow {
  id: string;
  violation_id: string;
  event_type: string;
  payload: Record<string, unknown>;
  sha256: string;
  created_at: string;
}

type Filter = "All" | "Critical" | "High" | "Pending";
const FILTERS: Filter[] = ["All", "Critical", "High", "Pending"];

function vioCode(id: string) {
  return "VIO-" + id.replace(/-/g, "").slice(0, 6).toUpperCase();
}

const SEVERITY_RANK: Record<Severity, number> = { Critical: 3, High: 2, Medium: 1, Low: 0 };

/** The most severe (then most confident) violation in a job's group — used as the card's
 *  headline chip and as the violation a click opens, while the body still lists every
 *  violation flagged on that same frame. */
function pickPrimary(group: VRow[]): VRow {
  return group.reduce((best, v) => {
    const a = SEVERITY_RANK[severityFor(v.violation_type)];
    const b = SEVERITY_RANK[severityFor(best.violation_type)];
    if (a !== b) return a > b ? v : best;
    return (v.confidence ?? 0) > (best.confidence ?? 0) ? v : best;
  });
}

export default function ViolationsPage() {
  return (
    <Suspense fallback={null}>
      <ViolationsInner />
    </Suspense>
  );
}

// Detail view is keyed off a `?id=` query param (rather than local state) so opening it pushes
// a real browser-history entry — that's what makes the back button / trackpad-back gesture land
// you on the violations list instead of skipping past it to whatever page you came from.
function ViolationsInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("id");

  const [rows, setRows] = useState<VRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<Filter>("All");
  const [q, setQ] = useState("");
  const selected = selectedId ? rows.find((r) => r.id === selectedId) ?? null : null;

  async function load() {
    const { data } = await supabase
      .from("violations")
      .select("*, plates(plate_text, plate_normalized, state_code), cameras(name, location_name)")
      .order("detected_at", { ascending: false })
      .limit(500);
    setRows((data as VRow[]) ?? []);
    setLoading(false);
  }

  useEffect(() => {
    load();
    const ch = supabase
      .channel("violations-list")
      .on("postgres_changes", { event: "*", schema: "public", table: "violations" }, () => load())
      .subscribe();
    return () => { supabase.removeChannel(ch); };
  }, []);

  const filtered = useMemo(() => {
    return rows.filter((v) => {
      const sev = severityFor(v.violation_type);
      if (filter === "Critical" && sev !== "Critical") return false;
      if (filter === "High" && sev !== "High") return false;
      if (filter === "Pending" && v.status !== "pending") return false;
      if (q) {
        const hay = `${v.plates?.plate_normalized ?? ""} ${v.id} ${v.cameras?.name ?? ""}`.toLowerCase();
        if (!hay.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [rows, filter, q]);

  // One ingestion job can carry several violations on the same frame (e.g. no-helmet + triple-
  // riding on one rider) — they share a single annotated_image_path (every box already burned
  // in server-side), so group by job_id and show one card per frame listing all of them, rather
  // than repeating the same image once per violation.
  const grouped = useMemo(() => {
    const map = new Map<string, VRow[]>();
    for (const v of filtered) {
      const arr = map.get(v.job_id);
      if (arr) arr.push(v);
      else map.set(v.job_id, [v]);
    }
    return Array.from(map.values());
  }, [filtered]);

  function exportCsv() {
    const head = ["id", "violation", "plate", "location", "confidence", "severity", "status", "detected_at"];
    const lines = filtered.map((v) =>
      [
        v.id,
        VIOLATION_LABELS[v.violation_type],
        v.plates?.plate_text ?? "",
        v.cameras?.location_name ?? v.cameras?.name ?? "",
        pct(v.confidence),
        severityFor(v.violation_type),
        statusMeta(v.status).label,
        v.detected_at,
      ].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")
    );
    const csv = [head.join(","), ...lines].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `gridlock-violations-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (selected) {
    return (
      <DetailScreen
        v={selected}
        siblings={rows.filter((r) => r.job_id === selected.job_id)}
        onBack={() => router.back()}
        onChanged={load}
      />
    );
  }

  const pending = rows.filter((v) => v.status === "pending").length;

  return (
    <div style={{ padding: "34px 36px", maxWidth: 1180, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", animation: "bfu .55s cubic-bezier(.16,1,.3,1) both" }}>
        <div>
          <h1 style={{ fontFamily: FONT.sans, fontSize: 27, fontWeight: 600, letterSpacing: "-0.025em", margin: 0 }}>Violations</h1>
          <p style={{ fontSize: 13.5, color: "#6B7280", margin: "5px 0 0" }}>
            <span style={{ fontFamily: FONT.mono, color: "#18181B", fontWeight: 600 }}>{rows.length}</span> records · {pending} pending review
          </p>
        </div>
        <button data-tour="viol-export" onClick={exportCsv} className="gl-press" style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: FONT.body, fontSize: 13.5, fontWeight: 600, color: "#fff", background: "#18181B", border: "none", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 3v12M7 10l5 5 5-5M5 21h14" /></svg>Export CSV
        </button>
      </div>

      {/* toolbar */}
      <div data-tour="viol-toolbar" style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 22, animation: "bfu .55s cubic-bezier(.16,1,.3,1) both", animationDelay: ".06s" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, background: "#fff", border: "1px solid #ECECEC", borderRadius: 11, padding: "9px 13px", flex: 1 }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="m21 21-4-4" /></svg>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by plate, ID, camera…"
            style={{ border: "none", outline: "none", fontSize: 13.5, flex: 1, fontFamily: FONT.body, color: "#18181B", background: "transparent" }}
          />
        </div>
        {FILTERS.map((f) => {
          const active = filter === f;
          return (
            <span
              key={f}
              onClick={() => setFilter(f)}
              className={active ? undefined : "gl-btn-ghost"}
              style={{
                fontFamily: FONT.body, fontSize: 12.5, fontWeight: active ? 600 : 500,
                color: active ? "#fff" : "#52525B", background: active ? "#4F46E5" : "#fff",
                border: active ? "none" : "1px solid #ECECEC", padding: "8px 14px", borderRadius: 9, cursor: "pointer",
              }}
            >
              {f}
            </span>
          );
        })}
      </div>

      {/* card grid */}
      {loading ? (
        <div data-tour="viol-table" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16, marginTop: 18 }}>
          {[0, 1, 2, 3, 4, 5].map((i) => <div key={i} className="shimmer" style={{ height: 312, borderRadius: 16 }} />)}
        </div>
      ) : filtered.length === 0 ? (
        <div style={{ marginTop: 18, padding: "48px 20px", textAlign: "center", color: "#6B7280", fontSize: 14, background: "#fff", border: "1px solid #ECECEC", borderRadius: 16 }}>No matching violations.</div>
      ) : (
        <div data-tour="viol-table" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16, marginTop: 18 }}>
          {grouped.map((group, i) => {
            const primary = pickPrimary(group);
            return <ViolationCard key={primary.job_id} group={group} primary={primary} index={i} onOpen={() => router.push(`/violations?id=${primary.id}`)} />;
          })}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "20px 4px 4px", fontSize: 12.5, color: "#9CA3AF" }}>
        <span style={{ fontFamily: FONT.mono }}>Showing {grouped.length} frame{grouped.length === 1 ? "" : "s"} · {filtered.length} of {rows.length} violations</span>
      </div>
    </div>
  );
}

function ViolationCard({ group, primary, index, onOpen }: { group: VRow[]; primary: VRow; index: number; onOpen: () => void }) {
  const col = SEVERITY_COLOR[severityFor(primary.violation_type)];
  const img = storageUrl("annotated", primary.annotated_image_path);
  const plate = group.find((v) => v.plates?.plate_text)?.plates?.plate_text ?? null;
  const location = primary.cameras?.location_name ?? primary.cameras?.name ?? "Unknown location";
  const lighting = (() => {
    const h = new Date(primary.detected_at).getHours();
    return h >= 6 && h < 18 ? "Daylight" : h >= 18 && h < 20 ? "Dusk" : "Low-light";
  })();

  const pendingCount = group.filter((v) => v.status === "pending").length;
  const statusChip = pendingCount > 0
    ? { label: pendingCount > 1 ? `${pendingCount} pending` : "Pending", c: "#F59E0B", b: "#FFFBEB" }
    : statusMeta(primary.status);

  const visible = group.slice(0, 4);
  const overflow = group.length - visible.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18, filter: "blur(6px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ duration: 0.45, delay: (index % 8) * 0.045, ease: [0.16, 1, 0.3, 1] }}
      onClick={onOpen}
      className="gl-vcard"
      style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", boxShadow: "0 1px 2px rgba(24,24,27,.04)", cursor: "pointer" }}
    >
      {/* image */}
      <div style={{ position: "relative", aspectRatio: "16/10", overflow: "hidden", background: "#0d0d12" }}>
        {img ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={img} alt="evidence" loading="lazy" className="gl-vcard-img" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#52525B", fontFamily: FONT.mono, fontSize: 11 }}>no frame stored</div>
        )}

        {/* top chips */}
        <div style={{ position: "absolute", top: 10, left: 10, display: "flex", alignItems: "center", gap: 6, fontFamily: FONT.body, fontSize: 10.5, fontWeight: 600, color: "#fff", background: "rgba(10,10,14,.55)", backdropFilter: "blur(6px)", padding: "5px 10px", borderRadius: 99 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: col.c, flex: "none" }} />
          {VIOLATION_LABELS[primary.violation_type]}
          {group.length > 1 && (
            <span style={{ marginLeft: 2, fontFamily: FONT.mono, fontSize: 9.5, color: "rgba(255,255,255,.75)" }}>+{group.length - 1}</span>
          )}
        </div>
        <div style={{ position: "absolute", top: 10, right: 10, fontFamily: FONT.body, fontSize: 10, fontWeight: 700, color: statusChip.c, background: statusChip.b, padding: "4px 9px", borderRadius: 99, border: `1px solid ${statusChip.c}28` }}>
          {statusChip.label}
        </div>

        {/* bottom scrim + caption */}
        {img && (
          <div className="gl-vcard-scrim" style={{ position: "absolute", inset: 0, top: "45%", background: "linear-gradient(to bottom, transparent, rgba(8,8,12,.82))", pointerEvents: "none" }} />
        )}
        <div style={{ position: "absolute", bottom: 9, left: 11, right: 11, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
          <span style={{ fontFamily: FONT.mono, fontSize: 13, fontWeight: 700, letterSpacing: ".03em", color: "#fff" }}>{plate ?? "no plate"}</span>
          <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,.85)" }}>{pct(primary.confidence)}</span>
        </div>

        {/* hover CTA */}
        <div className="gl-vcard-cta" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(10,10,14,.32)" }}>
          <span style={{ fontFamily: FONT.body, fontSize: 12.5, fontWeight: 600, color: "#18181B", background: "#fff", padding: "9px 16px", borderRadius: 99, display: "inline-flex", alignItems: "center", gap: 6, boxShadow: "0 8px 20px -8px rgba(0,0,0,.4)" }}>
            View evidence
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#18181B" strokeWidth="2.4"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </span>
        </div>
      </div>

      {/* body */}
      <div style={{ padding: "12px 14px 13px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>{vioCode(primary.id)}</span>
          <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>{timeAgo(primary.detected_at)}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 7, fontSize: 12.5, color: "#52525B", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2" style={{ flex: "none" }}><path d="M12 22s7-7.58 7-12.5A7 7 0 0 0 5 9.5C5 14.42 12 22 12 22z" /><circle cx="12" cy="9.5" r="2.5" /></svg>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{location}</span>
          <span style={{ color: "#D4D4D8", flex: "none" }}>·</span>
          <span style={{ flex: "none" }}>{lighting}</span>
        </div>

        {/* every violation flagged on this frame */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {visible.map((v) => {
            const vcol = SEVERITY_COLOR[severityFor(v.violation_type)];
            return (
              <span key={v.id} style={{ display: "inline-flex", alignItems: "center", gap: 5, fontFamily: FONT.body, fontSize: 11, fontWeight: 500, color: "#3F3F46", background: "#FAFAFA", border: "1px solid #F0F0F0", padding: "3px 8px 3px 6px", borderRadius: 7 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: vcol.c, flex: "none" }} />
                {VIOLATION_LABELS[v.violation_type]}
                <span style={{ fontFamily: FONT.mono, color: "#9CA3AF" }}>{pct(v.confidence)}</span>
              </span>
            );
          })}
          {overflow > 0 && (
            <span style={{ display: "inline-flex", alignItems: "center", fontFamily: FONT.mono, fontSize: 11, fontWeight: 600, color: "#9CA3AF", padding: "3px 7px" }}>+{overflow} more</span>
          )}
        </div>

        <div style={{ marginTop: 10, height: 4, background: "#F4F4F5", borderRadius: 99, overflow: "hidden" }}>
          <motion.div
            initial={{ width: 0 }}
            whileInView={{ width: `${(primary.confidence ?? 0) * 100}%` }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: (index % 8) * 0.045 + 0.15, ease: [0.16, 1, 0.3, 1] }}
            style={{ height: "100%", borderRadius: 99, background: col.c }}
          />
        </div>
      </div>
    </motion.div>
  );
}

/* ───────────────────────── Detail / evidence screen ───────────────────────── */

// Renders a bbox crop of `src` (a normalized 0-1 fraction of the image) using an SVG viewBox
// instead of CSS background-position math, so it never stretches/distorts: "meet" (contain)
// shows the full crop letterboxed within the frame with even margins regardless of the bbox's
// own aspect ratio; "slice" (cover) fills the frame completely, lightly cropping overflow.
function CropSVG({ src, bbox, natural, fit }: {
  src: string; bbox: { x: number; y: number; w: number; h: number };
  natural: { w: number; h: number }; fit: "meet" | "slice";
}) {
  const vx = bbox.x * natural.w;
  const vy = bbox.y * natural.h;
  const vw = Math.max(1, bbox.w * natural.w);
  const vh = Math.max(1, bbox.h * natural.h);
  return (
    <svg width="100%" height="100%" viewBox={`${vx} ${vy} ${vw} ${vh}`} preserveAspectRatio={`xMidYMid ${fit}`} style={{ display: "block" }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <image href={src} x={0} y={0} width={natural.w} height={natural.h} />
    </svg>
  );
}

function PlateScramble({ text }: { text: string }) {
  const [display, setDisplay] = useState(text);
  useEffect(() => {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let frame = 0;
    const total = 11;
    const iv = setInterval(() => {
      frame++;
      if (frame >= total) { setDisplay(text); clearInterval(iv); return; }
      setDisplay(text.split("").map((c, i) => {
        const lock = (frame / total) * text.length;
        if (i < lock) return c;
        if (c === " ") return " ";
        return chars[Math.floor(Math.random() * chars.length)];
      }).join(""));
    }, 45);
    return () => clearInterval(iv);
  }, [text]);
  return <span>{display}</span>;
}

function DetailScreen({ v, siblings, onBack, onChanged }: { v: VRow; siblings: VRow[]; onBack: () => void; onChanged: () => void }) {
  const sev = severityFor(v.violation_type) as Severity;
  const col = SEVERITY_COLOR[sev];
  const st = statusMeta(v.status);
  const img = storageUrl("annotated", v.annotated_image_path);
  const sha = v.sha256_evidence ?? "sha256 chained · see audit log";

  const [dets, setDets] = useState<DetRow[]>([]);
  const [audit, setAudit] = useState<AuditRow[]>([]);
  const [zoom, setZoom] = useState<DetRow | null>(null);
  const [busy, setBusy] = useState(false);
  // Real pixel dimensions of the annotated image — CropSVG needs these to map a normalized
  // bbox fraction into the SVG's source-pixel viewBox.
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    setNatural(null);
    if (!img) return;
    const el = new window.Image();
    el.onload = () => setNatural({ w: el.naturalWidth, h: el.naturalHeight });
    el.src = img;
  }, [img]);

  useEffect(() => {
    setZoom(null);
    supabase.from("detections").select("id,job_id,class_label,confidence,bbox").eq("job_id", v.job_id)
      .then(({ data }) => setDets(((data as DetRow[]) ?? []).filter(d => d.bbox && "x" in d.bbox)));
    supabase.from("evidence_audit").select("id,violation_id,event_type,payload,sha256,created_at").eq("violation_id", v.id)
      .order("created_at", { ascending: true })
      .then(({ data }) => setAudit((data as AuditRow[]) ?? []));
  }, [v.id, v.job_id]);

  async function act(action: "confirm" | "reject" | "escalate") {
    setBusy(true);
    try {
      await reviewViolation(v.id, action);
      onChanged();
      onBack();
    } catch { /* surfaced visually by staying on the screen */ }
    finally { setBusy(false); }
  }

  const lightingTag = (() => {
    const h = new Date(v.detected_at).getHours();
    return h >= 6 && h < 18 ? "Daylight" : h >= 18 && h < 20 ? "Dusk" : "Low-light";
  })();
  const evidence = v.evidence as Record<string, unknown>;
  const coords = (evidence?.coordinates as { lat: number; lng: number } | undefined) ?? null;

  return (
    <div style={{ padding: "30px 36px 60px", maxWidth: 1240, margin: "0 auto" }}>
      {/* header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <motion.button
            whileHover={{ x: -2 }}
            whileTap={{ scale: 0.92 }}
            onClick={onBack}
            aria-label="Back to violations"
            style={{ width: 36, height: 36, flex: "none", marginTop: 1, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 10, border: "1px solid #ECECEC", background: "#fff", cursor: "pointer" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#52525B" strokeWidth="2.2"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
          </motion.button>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 13, fontFamily: FONT.body }}>
              <span onClick={onBack} style={{ color: "#9CA3AF", cursor: "pointer" }}>Violations</span>
              <span style={{ color: "#D4D4D8" }}>/</span>
              <span style={{ fontFamily: FONT.mono, color: "#18181B", fontWeight: 600 }}>#{vioCode(v.id)}</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
              <h1 style={{ fontFamily: FONT.sans, fontSize: 23, fontWeight: 600, letterSpacing: "-0.02em", margin: 0 }}>{VIOLATION_LABELS[v.violation_type]}</h1>
              <span style={{ fontFamily: FONT.body, fontSize: 11, fontWeight: 700, color: st.c, background: st.b, padding: "4px 11px", borderRadius: 99, border: `1px solid ${st.c}28` }}>{st.label} review</span>
            </div>
          </div>
        </div>
        {v.status === "pending" && (
          <div style={{ display: "flex", gap: 9 }}>
            <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} disabled={busy} onClick={() => act("confirm")} style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#fff", background: "#4F46E5", border: "none", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>Approve &amp; issue notice</motion.button>
            <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} disabled={busy} onClick={() => act("reject")} style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>Dismiss</motion.button>
            <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} disabled={busy} onClick={() => act("escalate")} style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#F59E0B", background: "#FFFBEB", border: "1px solid #FDE68A", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>Flag for re-check</motion.button>
          </div>
        )}
      </motion.div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 16, alignItems: "start" }}>
        {/* LEFT */}
        <div>
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 12, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ position: "relative", borderRadius: 11, overflow: "hidden", background: "#0d0d12" }}>
              {img ? (
                // The annotated image already has the violation's own box + label burned in
                // server-side (EvidenceGenerator) — shown as-is, same as the Detect page result.
                // A client-side overlay of the job's generic object detections used to be drawn
                // on top here, but it visually clashed with the clean server-rendered box (gappy/
                // misaligned thin outlines), so it's been dropped in favor of just the annotated
                // image; the "Evidence crops" strip below still uses the same detections.
                // eslint-disable-next-line @next/next/no-img-element
                <img src={img} alt="evidence" style={{ display: "block", width: "100%", height: "auto" }} />
              ) : (
                <div style={{ aspectRatio: "16/10", display: "flex", alignItems: "center", justifyContent: "center", color: "#52525B", fontFamily: FONT.mono, fontSize: 12 }}>no annotated frame stored</div>
              )}
            </div>
          </motion.div>

          {/* evidence crop strip */}
          {dets.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 8 }}>EVIDENCE CROPS</div>
              <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 4 }}>
                {dets.slice(0, 6).map(d => (
                  <motion.div
                    key={d.id}
                    layoutId={`crop-${d.id}`}
                    onClick={() => setZoom(d)}
                    whileHover={{ y: -3 }}
                    style={{ flex: "none", width: 110, height: 78, borderRadius: 10, overflow: "hidden", border: "1px solid #ECECEC", cursor: "zoom-in", position: "relative", background: "#f4f4f5" }}
                  >
                    {img && d.bbox && natural && (
                      <CropSVG src={img} bbox={d.bbox} natural={natural} fit="slice" />
                    )}
                    <span style={{ position: "absolute", bottom: 3, left: 4, fontFamily: FONT.mono, fontSize: 8.5, color: "#fff", background: "rgba(0,0,0,.55)", padding: "1px 5px", borderRadius: 4 }}>{prettyClass(d.class_label)}</span>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          <AnimatePresence>
            {zoom && (
              <motion.div onClick={() => setZoom(null)} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ position: "fixed", inset: 0, zIndex: 60, background: "rgba(10,10,14,.78)", backdropFilter: "blur(6px)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <motion.div onClick={(e) => e.stopPropagation()} layoutId={`crop-${zoom.id}`} style={{
                  width: "min(440px, 86vw)",
                  height: "min(440px, 70vh)",
                  borderRadius: 16, overflow: "hidden", background: "#111", position: "relative",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  {img && zoom.bbox && natural && (
                    <CropSVG src={img} bbox={zoom.bbox} natural={natural} fit="meet" />
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* RIGHT */}
        <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
          {/* Vehicle & Plate */}
          <motion.div initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.45, delay: 0.05 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 18, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 10 }}>VEHICLE &amp; PLATE</div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "16px 12px", borderRadius: 11, background: "#FAFAFA", border: "1px solid #F4F4F5" }}>
              <span style={{ fontFamily: FONT.mono, fontSize: 25, fontWeight: 700, letterSpacing: ".06em", color: v.plates?.plate_text ? "#18181B" : "#9CA3AF" }}>
                {v.plates?.plate_text ? <PlateScramble text={v.plates.plate_text} /> : "no plate read"}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 12 }}>
              <span style={{ fontFamily: FONT.body, fontSize: 11.5, color: "#6B7280" }}>State <b style={{ color: "#18181B" }}>{v.plates?.state_code ?? "—"}</b></span>
              {v.plates && (
                <span style={{ fontFamily: FONT.body, fontSize: 11, fontWeight: 700, color: "#10B981", background: "#ECFDF5", padding: "3px 9px", borderRadius: 6 }}>Indian-format validated ✓</span>
              )}
            </div>
          </motion.div>

          {/* Violations */}
          <motion.div initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.45, delay: 0.1 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ padding: "13px 18px", borderBottom: "1px solid #F4F4F5", fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF" }}>VIOLATIONS · {siblings.length}</div>
            {siblings.map((sv, i) => {
              const ssev = severityFor(sv.violation_type);
              const scol = SEVERITY_COLOR[ssev];
              return (
                <div key={sv.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "11px 18px", borderBottom: i < siblings.length - 1 ? "1px solid #F4F4F5" : "none", background: sv.id === v.id ? "#FAFAFE" : undefined }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <span style={{ width: 8, height: 8, borderRadius: "50%", background: scol.c }} />
                    <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>{VIOLATION_LABELS[sv.violation_type]}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <span style={{ fontFamily: FONT.mono, fontSize: 11.5, color: "#6B7280" }}>{pct(sv.confidence)}</span>
                    <span style={{ fontFamily: FONT.mono, fontSize: 10, color: "#C4C4C8" }}>§3.{(i % 9) + 1}</span>
                  </div>
                </div>
              );
            })}
          </motion.div>

          {/* Confidence */}
          {dets.length > 0 && (
            <motion.div initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.45, delay: 0.15 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 18, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 12 }}>CONFIDENCE</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {dets.slice(0, 5).map((d, i) => (
                  <div key={d.id}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 12, color: "#52525B" }}>{prettyClass(d.class_label)}</span>
                      <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 600 }}>{pct(d.confidence)}</span>
                    </div>
                    <div style={{ height: 5, background: "#F4F4F5", borderRadius: 99, overflow: "hidden" }}>
                      <motion.div initial={{ width: 0 }} animate={{ width: `${(d.confidence ?? 0) * 100}%` }} transition={{ duration: 0.7, delay: 0.2 + i * 0.06, ease: [0.16, 1, 0.3, 1] }} style={{ height: "100%", borderRadius: 99, background: boxColor(d.class_label) }} />
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Metadata */}
          <motion.div initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.45, delay: 0.2 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 18, boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 12 }}>METADATA</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 11, fontSize: 12.5 }}>
              <MetaItem label="Camera" value={v.cameras?.name ?? "—"} />
              <MetaItem label="Zone" value={v.cameras?.location_name ?? "—"} />
              <MetaItem label="Timestamp" value={new Date(v.detected_at).toLocaleString()} mono />
              <MetaItem label="Lighting" value={lightingTag} />
              <MetaItem label="Coordinates" value={coords ? `${coords.lat.toFixed(4)}, ${coords.lng.toFixed(4)}` : "—"} mono />
              <MetaItem label="Evidence hash" value={sha.slice(0, 16) + "…"} mono />
            </div>
          </motion.div>

          {/* Audit timeline */}
          <motion.div initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 0.45, delay: 0.25 }} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "18px 18px 16px", boxShadow: "0 1px 3px rgba(0,0,0,.04)" }}>
            <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 14 }}>AUDIT LOG</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {audit.length === 0 ? (
                <span style={{ fontSize: 12.5, color: "#9CA3AF" }}>No audit events recorded.</span>
              ) : audit.map((a, i) => (
                <div key={a.id} style={{ display: "flex", gap: 11, paddingBottom: i < audit.length - 1 ? 16 : 0 }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: "none" }}>
                    <span style={{ width: 9, height: 9, borderRadius: "50%", background: a.event_type === "created" ? "#4F46E5" : a.event_type === "reviewed" ? "#10B981" : "#9CA3AF", flexShrink: 0 }} />
                    {i < audit.length - 1 && <span style={{ width: 1, flex: 1, background: "#ECECEC", marginTop: 3 }} />}
                  </div>
                  <div style={{ paddingBottom: 2 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 600, color: "#18181B", textTransform: "capitalize" }}>{a.event_type}</div>
                    <div style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF", marginTop: 2 }}>{timeAgo(a.created_at)} · {new Date(a.created_at).toLocaleString()}</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

function MetaItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "#9CA3AF", marginBottom: 3 }}>{label}</div>
      <div style={{ color: "#18181B", fontFamily: mono ? FONT.mono : FONT.body, fontSize: mono ? 11.5 : 12.5, fontWeight: 500, wordBreak: "break-word" }}>{value}</div>
    </div>
  );
}
