"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from "recharts";
import { supabase } from "@/lib/supabase";
import type { Violation, ConfidenceBand } from "@/lib/types";
import { VIOLATION_LABELS, BAND_STYLE, BAND_LABELS, timeAgo, pct } from "@/lib/format";
import StatCard from "@/components/StatCard";
import Reveal from "@/components/Reveal";

interface ModelRow {
  module: string;
  model_name: string;
  variant: string | null;
  metric_name: string | null;
  metric_value: number | null;
}

const BAND_COLOR: Record<ConfidenceBand, string> = {
  auto_confirm: "#10b981",
  human_review: "#f59e0b",
  discard: "#94a3b8",
};

export default function Dashboard() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [models, setModels] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    const { data } = await supabase
      .from("violations")
      .select("*, plates(plate_text, state_code, plate_normalized)")
      .order("detected_at", { ascending: false })
      .limit(500);
    setViolations((data as Violation[]) ?? []);
    setLoading(false);
  }

  useEffect(() => {
    load();
    supabase
      .from("model_registry")
      .select("module, model_name, variant, metric_name, metric_value")
      .eq("is_active", true)
      .then(({ data }) => setModels((data as ModelRow[]) ?? []));

    const ch = supabase
      .channel("dash-violations")
      .on("postgres_changes", { event: "*", schema: "public", table: "violations" }, () => load())
      .subscribe();
    return () => {
      supabase.removeChannel(ch);
    };
  }, []);

  const stats = useMemo(() => {
    const band = (b: ConfidenceBand) => violations.filter((v) => v.confidence_band === b).length;
    return {
      total: violations.length,
      auto: band("auto_confirm"),
      review: band("human_review"),
      confirmed: violations.filter((v) => v.status === "confirmed").length,
    };
  }, [violations]);

  const byType = useMemo(() => {
    const m = new Map<string, number>();
    violations.forEach((v) => m.set(v.violation_type, (m.get(v.violation_type) ?? 0) + 1));
    return Array.from(m.entries()).map(([k, count]) => ({
      type: VIOLATION_LABELS[k as keyof typeof VIOLATION_LABELS] ?? k,
      count,
    }));
  }, [violations]);

  const byBand = useMemo(() => {
    return (["auto_confirm", "human_review", "discard"] as ConfidenceBand[])
      .map((b) => ({
        name: BAND_LABELS[b],
        value: violations.filter((v) => v.confidence_band === b).length,
        band: b,
      }))
      .filter((d) => d.value > 0);
  }, [violations]);

  return (
    <div className="mx-auto max-w-6xl">
      {/* Hero */}
      <motion.header
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="mb-10"
      >
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-indigo-600">
          <span className="live-dot h-1.5 w-1.5 rounded-full bg-emerald-500" /> Live Console
        </div>
        <h1 className="font-display mt-3 text-5xl font-bold leading-tight tracking-tight text-slate-900 md:text-6xl">
          Traffic Violation
          <br />
          <span className="bg-gradient-to-r from-indigo-600 via-violet-500 to-sky-500 bg-clip-text text-transparent">
            Intelligence
          </span>
        </h1>
        <p className="mt-4 max-w-xl text-slate-500">
          Photo-first detection across 7 violation classes with automatic plate recognition,
          calibrated confidence and tamper-evident evidence.
        </p>
      </motion.header>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Total Violations" value={stats.total} accent="indigo" delay={0.05} />
        <StatCard label="Auto-confirmed" value={stats.auto} accent="sky" delay={0.1} />
        <StatCard label="Needs Review" value={stats.review} accent="amber" delay={0.15} />
        <StatCard label="Confirmed" value={stats.confirmed} accent="emerald" delay={0.2} />
      </div>

      {/* Charts */}
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Reveal className="surface rounded-2xl p-5 lg:col-span-2" delay={0.05}>
          <div className="mb-4 text-sm font-medium text-slate-700">Violations by type</div>
          {byType.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={byType} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
                <XAxis
                  dataKey="type"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  interval={0}
                />
                <Tooltip
                  cursor={{ fill: "rgba(99,102,241,0.06)" }}
                  contentStyle={{
                    background: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 12,
                    color: "#0f172a",
                    boxShadow: "0 10px 30px -12px rgba(15,23,42,0.18)",
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Reveal>

        <Reveal className="surface rounded-2xl p-5" delay={0.1}>
          <div className="mb-4 text-sm font-medium text-slate-700">Confidence bands</div>
          {byBand.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={byBand}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={3}
                  stroke="none"
                >
                  {byBand.map((d) => (
                    <Cell key={d.band} fill={BAND_COLOR[d.band]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 12,
                    color: "#0f172a",
                    boxShadow: "0 10px 30px -12px rgba(15,23,42,0.18)",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Reveal>
      </div>

      {/* Deployed models */}
      <Reveal className="mt-10" delay={0.05}>
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-xl font-semibold text-slate-900">Deployed models</h2>
          <span className="text-xs text-slate-400">benchmarked · honest metrics</span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {models.map((m, i) => (
            <motion.div
              key={m.module}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.04, duration: 0.5 }}
              className="surface surface-hover rounded-xl p-4"
            >
              <div className="text-xs uppercase tracking-wider text-slate-400">
                {m.module.replace(/_/g, " ")}
              </div>
              <div className="mt-1 text-sm font-medium text-slate-700">{m.model_name}</div>
              <div className="font-display mt-3 text-2xl font-bold text-indigo-600">
                {m.metric_value != null ? m.metric_value.toFixed(m.metric_value < 1 ? 4 : 0) : "—"}
              </div>
              <div className="text-[11px] text-slate-400">{m.metric_name}</div>
            </motion.div>
          ))}
        </div>
      </Reveal>

      {/* Live feed */}
      <Reveal className="mt-10" delay={0.05}>
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-xl font-semibold text-slate-900">Recent activity</h2>
          <Link href="/violations" className="text-xs text-indigo-600 hover:text-indigo-700">
            View all →
          </Link>
        </div>

        {loading ? (
          <FeedSkeleton />
        ) : violations.length === 0 ? (
          <div className="surface rounded-2xl p-10 text-center">
            <div className="text-slate-700">No violations yet</div>
            <p className="mt-1 text-sm text-slate-500">
              Upload traffic evidence to run the detection pipeline.
            </p>
            <Link
              href="/upload"
              className="mt-5 inline-block rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-indigo-500"
            >
              Upload evidence
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {violations.slice(0, 8).map((v, i) => (
              <motion.div
                key={v.id}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.03 }}
                className="surface surface-hover flex items-center gap-4 rounded-xl px-4 py-3"
              >
                <span className={`rounded-md px-2 py-1 text-xs ${BAND_STYLE[v.confidence_band]}`}>
                  {BAND_LABELS[v.confidence_band]}
                </span>
                <span className="font-medium text-slate-800">
                  {VIOLATION_LABELS[v.violation_type]}
                </span>
                {v.plates?.plate_text && (
                  <span className="font-plate rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600">
                    {v.plates.plate_text}
                  </span>
                )}
                <span className="ml-auto text-xs text-slate-500">{pct(v.confidence)}</span>
                <span className="w-16 text-right text-xs text-slate-400">
                  {timeAgo(v.detected_at)}
                </span>
              </motion.div>
            ))}
          </div>
        )}
      </Reveal>
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="flex h-[220px] items-center justify-center text-sm text-slate-400">
      Awaiting data
    </div>
  );
}

function FeedSkeleton() {
  return (
    <div className="space-y-2">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="shimmer relative h-12 overflow-hidden rounded-xl" />
      ))}
    </div>
  );
}
