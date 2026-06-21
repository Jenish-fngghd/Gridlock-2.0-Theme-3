"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";

export default function StatCard({
  label,
  value,
  accent = "indigo",
  hint,
  delay = 0,
}: {
  label: string;
  value: number;
  accent?: "indigo" | "emerald" | "amber" | "sky";
  hint?: string;
  delay?: number;
}) {
  const [n, setN] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const duration = 900;
    let raf = 0;
    const tick = () => {
      const p = Math.min(1, (Date.now() - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setN(Math.round(eased * value));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const accents: Record<string, string> = {
    indigo: "text-indigo-600",
    emerald: "text-emerald-600",
    amber: "text-amber-600",
    sky: "text-sky-600",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay, ease: [0.22, 1, 0.36, 1] }}
      className="surface surface-hover relative overflow-hidden rounded-2xl p-5"
    >
      <div className="text-xs uppercase tracking-widest text-slate-400">{label}</div>
      <div className={`font-display mt-3 text-4xl font-bold tabular-nums ${accents[accent]}`}>
        {n.toLocaleString()}
      </div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </motion.div>
  );
}
