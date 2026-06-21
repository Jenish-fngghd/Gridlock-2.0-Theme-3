"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/violations", label: "Violations" },
  { href: "/upload", label: "Upload Evidence" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <aside className="sticky top-0 flex h-screen w-60 shrink-0 flex-col border-r border-slate-200 bg-white/70 px-4 py-7 backdrop-blur-xl">
      <Link href="/" className="mb-10 block px-2">
        <div className="font-display text-2xl font-bold tracking-tight text-slate-900">
          Gridlock <span className="text-indigo-600">2.0</span>
        </div>
        <div className="mt-0.5 text-[10px] uppercase tracking-[0.3em] text-slate-400">
          Violation Console
        </div>
      </Link>

      <nav className="space-y-1">
        {links.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`relative block rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                active
                  ? "text-indigo-700"
                  : "text-slate-500 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-xl border border-indigo-200 bg-indigo-50"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
              <span className="relative z-10">{l.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto space-y-3 px-2">
        <div className="flex items-center gap-2 text-[11px] text-slate-500">
          <span className="live-dot h-1.5 w-1.5 rounded-full bg-emerald-500" />
          Live · Supabase realtime
        </div>
        <div className="text-[10px] leading-relaxed text-slate-400">
          Photo-first enforcement · 7 violation types + ANPR
        </div>
      </div>
    </aside>
  );
}
