"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { supabase, storageUrl } from "@/lib/supabase";
import type { Violation, ViolationType, ConfidenceBand } from "@/lib/types";
import {
  VIOLATION_LABELS,
  BAND_STYLE,
  BAND_LABELS,
  STATUS_STYLE,
  timeAgo,
  pct,
} from "@/lib/format";
import { reviewViolation } from "@/lib/api";

const TYPES = Object.keys(VIOLATION_LABELS) as ViolationType[];
const BANDS: ConfidenceBand[] = ["auto_confirm", "human_review", "discard"];

export default function ViolationsPage() {
  const [rows, setRows] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [type, setType] = useState<ViolationType | "all">("all");
  const [band, setBand] = useState<ConfidenceBand | "all">("all");
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<Violation | null>(null);

  async function load() {
    const { data } = await supabase
      .from("violations")
      .select("*, plates(plate_text, plate_normalized, state_code)")
      .order("detected_at", { ascending: false })
      .limit(500);
    setRows((data as Violation[]) ?? []);
    setLoading(false);
  }

  useEffect(() => {
    load();
    const ch = supabase
      .channel("violations-list")
      .on("postgres_changes", { event: "*", schema: "public", table: "violations" }, () => load())
      .subscribe();
    return () => {
      supabase.removeChannel(ch);
    };
  }, []);

  const filtered = useMemo(() => {
    return rows.filter((v) => {
      if (type !== "all" && v.violation_type !== type) return false;
      if (band !== "all" && v.confidence_band !== band) return false;
      if (q && !v.plates?.plate_normalized?.toLowerCase().includes(q.toLowerCase())) return false;
      return true;
    });
  }, [rows, type, band, q]);

  async function act(v: Violation, action: "confirm" | "reject") {
    try {
      await reviewViolation(v.id, action);
      setSelected(null);
      load();
    } catch {
      /* surfaced via backend; no-op for demo */
    }
  }

  return (
    <div className="mx-auto max-w-6xl">
      <motion.header
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="mb-8"
      >
        <h1 className="font-display text-4xl font-bold tracking-tight text-slate-900">Violations</h1>
        <p className="mt-2 text-slate-500">
          {filtered.length} record{filtered.length === 1 ? "" : "s"} · searchable, calibrated,
          auditable.
        </p>
      </motion.header>

      {/* Filters */}
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search plate…"
          className="font-plate w-48 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-800 outline-none placeholder:text-slate-400 focus:border-indigo-400"
        />
        <Select value={type} onChange={(v) => setType(v as ViolationType | "all")} options={["all", ...TYPES]} labels={VIOLATION_LABELS} />
        <Select value={band} onChange={(v) => setBand(v as ConfidenceBand | "all")} options={["all", ...BANDS]} labels={BAND_LABELS} />
      </div>

      {/* List */}
      {loading ? (
        <div className="space-y-2">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="shimmer relative h-14 overflow-hidden rounded-xl" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="surface rounded-2xl p-10 text-center text-slate-500">
          No matching violations.
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((v, i) => (
            <motion.button
              key={v.id}
              onClick={() => setSelected(v)}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.3) }}
              className="surface surface-hover flex w-full items-center gap-4 rounded-xl px-4 py-3 text-left"
            >
              <span className={`rounded-md px-2 py-1 text-xs ${BAND_STYLE[v.confidence_band]}`}>
                {BAND_LABELS[v.confidence_band]}
              </span>
              <span className="font-medium text-slate-800">{VIOLATION_LABELS[v.violation_type]}</span>
              {v.plates?.plate_text && (
                <span className="font-plate rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600">
                  {v.plates.plate_text}
                </span>
              )}
              <span className={`text-xs capitalize ${STATUS_STYLE[v.status]}`}>
                {v.status.replace("_", " ")}
              </span>
              <span className="ml-auto font-plate text-xs text-slate-500">{pct(v.confidence)}</span>
              <span className="w-16 text-right text-xs text-slate-400">{timeAgo(v.detected_at)}</span>
            </motion.button>
          ))}
        </div>
      )}

      {/* Detail drawer */}
      <AnimatePresence>
        {selected && (
          <motion.div
            className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelected(null)}
          >
            <motion.div
              className="h-full w-full max-w-md overflow-y-auto border-l border-slate-200 bg-white p-6"
              initial={{ x: 40 }}
              animate={{ x: 0 }}
              exit={{ x: 40 }}
              transition={{ type: "spring", stiffness: 320, damping: 34 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h2 className="font-display text-2xl font-semibold text-slate-900">
                  {VIOLATION_LABELS[selected.violation_type]}
                </h2>
                <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-slate-700">
                  ✕
                </button>
              </div>

              <div className="mt-2 flex items-center gap-2">
                <span className={`rounded-md px-2 py-1 text-xs ${BAND_STYLE[selected.confidence_band]}`}>
                  {BAND_LABELS[selected.confidence_band]}
                </span>
                <span className={`text-xs capitalize ${STATUS_STYLE[selected.status]}`}>
                  {selected.status.replace("_", " ")}
                </span>
                <span className="ml-auto font-plate text-sm text-slate-500">{pct(selected.confidence)}</span>
              </div>

              {storageUrl("annotated", selected.annotated_image_path) && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={storageUrl("annotated", selected.annotated_image_path)!}
                  alt="evidence"
                  className="mt-4 w-full rounded-xl border border-slate-200"
                />
              )}

              {selected.vlm_caption && (
                <p className="mt-4 rounded-xl bg-slate-50 p-3 text-sm text-slate-600">
                  {selected.vlm_caption}
                </p>
              )}

              {selected.plates?.plate_text && (
                <div className="mt-4">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Plate</div>
                  <div className="font-plate mt-1 text-xl text-indigo-700">
                    {selected.plates.plate_text}
                  </div>
                </div>
              )}

              <div className="mt-4">
                <div className="text-xs uppercase tracking-wider text-slate-400">Evidence hash</div>
                <div className="font-plate mt-1 break-all text-xs text-slate-500">
                  {(selected.evidence as { sha?: string })?.sha ?? "sha256 chained · see audit log"}
                </div>
              </div>

              <div className="mt-8 flex gap-3">
                <button
                  onClick={() => act(selected, "confirm")}
                  className="flex-1 rounded-xl bg-indigo-600 px-4 py-2.5 font-medium text-white transition hover:bg-indigo-500"
                >
                  Confirm
                </button>
                <button
                  onClick={() => act(selected, "reject")}
                  className="flex-1 rounded-xl border border-rose-300 px-4 py-2.5 font-medium text-rose-600 transition hover:bg-rose-50"
                >
                  Reject
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
  labels,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  labels: Record<string, string>;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-700 outline-none focus:border-indigo-400"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o === "all" ? "All" : labels[o] ?? o}
        </option>
      ))}
    </select>
  );
}
