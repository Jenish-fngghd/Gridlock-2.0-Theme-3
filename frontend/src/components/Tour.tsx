"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import { FONT } from "@/lib/ui";
import { SHELL_KEY, SHELL_STEPS, TOURS, keyForPath, pageStorageKey, type PageKey, type TourStep } from "@/lib/tours";
import { TOUR_EVENT, type TourRequestDetail } from "@/lib/tourBus";

interface RunningTour {
  steps: TourStep[];
  pageKey: PageKey;
  includesShell: boolean;
}

const START_DELAY = 700; // let real page content (data-tour targets) mount first
const POLL_MS = 150;
const MAX_POLLS = 24; // ~3.6s before giving up on a missing target

function useTargetRect(selector: string | null) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    setRect(null);
    setMissing(false);
    if (!selector) return;
    const sel = selector;
    let tries = 0;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    function attempt() {
      const el = document.querySelector(sel);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        // wait a tick for smooth-scroll to settle before measuring
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (!cancelled) setRect(el.getBoundingClientRect());
        }));
        return;
      }
      tries++;
      if (tries >= MAX_POLLS) { if (!cancelled) setMissing(true); return; }
      timer = setTimeout(attempt, POLL_MS);
    }
    attempt();

    function onScrollResize() {
      const el = document.querySelector(sel);
      if (el) setRect(el.getBoundingClientRect());
    }
    window.addEventListener("scroll", onScrollResize, true);
    window.addEventListener("resize", onScrollResize);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener("scroll", onScrollResize, true);
      window.removeEventListener("resize", onScrollResize);
    };
  }, [selector]);

  return { rect, missing };
}

export default function Tour() {
  const pathname = usePathname();
  const [running, setRunning] = useState<RunningTour | null>(null);
  const [idx, setIdx] = useState(0);
  const [helpOpen, setHelpOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  useEffect(() => setMounted(true), []);

  const finish = useCallback((tour: RunningTour | null) => {
    if (tour) {
      try {
        localStorage.setItem(pageStorageKey(tour.pageKey), "1");
        if (tour.includesShell) localStorage.setItem(SHELL_KEY, "1");
      } catch { /* localStorage unavailable — non-fatal */ }
    }
    setRunning(null);
    setIdx(0);
    setPos(null);
  }, []);

  const begin = useCallback((pageKey: PageKey, opts: { forceShell?: boolean } = {}) => {
    const steps = TOURS[pageKey];
    if (!steps || steps.length === 0) return;
    const isConsole = pageKey !== "landing";
    let shellSeen = true;
    try { shellSeen = !!localStorage.getItem(SHELL_KEY); } catch { /* ignore */ }
    const includeShell = isConsole && (opts.forceShell || !shellSeen);
    setPos(null);
    setIdx(0);
    setRunning({ steps: includeShell ? [...SHELL_STEPS, ...steps] : steps, pageKey, includesShell: includeShell });
  }, []);

  // Auto-launch for first-time visitors.
  useEffect(() => {
    const key = keyForPath(pathname);
    if (!key) return;
    let seen = false;
    try { seen = !!localStorage.getItem(pageStorageKey(key)); } catch { /* ignore */ }
    if (seen) return;
    const t = setTimeout(() => begin(key), START_DELAY);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  // Manual restart via the help button / nav links, decoupled through a DOM event.
  useEffect(() => {
    function onRequest(e: Event) {
      const detail = (e as CustomEvent<TourRequestDetail>).detail ?? {};
      const key = detail.key ?? keyForPath(pathname);
      if (!key) return;
      begin(key, { forceShell: detail.forceShell });
    }
    window.addEventListener(TOUR_EVENT, onRequest);
    return () => window.removeEventListener(TOUR_EVENT, onRequest);
  }, [pathname, begin]);

  const step = running?.steps[idx] ?? null;
  const { rect, missing } = useTargetRect(step?.selector ?? null);

  // Skip gracefully past steps whose target never mounts (e.g. a different phase of the page).
  useEffect(() => {
    if (missing && running) {
      if (idx < running.steps.length - 1) setIdx((i) => i + 1);
      else finish(running);
    }
  }, [missing, running, idx, finish]);

  useLayoutEffect(() => {
    if (!rect || !cardRef.current || !step) { setPos(null); return; }
    const card = cardRef.current.getBoundingClientRect();
    const gap = 16;
    const placement = step.placement ?? "bottom";
    let top: number, left: number;
    if (placement === "center") {
      top = window.innerHeight / 2 - card.height / 2;
      left = window.innerWidth / 2 - card.width / 2;
    } else if (placement === "top") {
      top = rect.top - card.height - gap;
      left = rect.left + rect.width / 2 - card.width / 2;
    } else if (placement === "left") {
      top = rect.top + rect.height / 2 - card.height / 2;
      left = rect.left - card.width - gap;
    } else if (placement === "right") {
      top = rect.top + rect.height / 2 - card.height / 2;
      left = rect.right + gap;
    } else {
      top = rect.bottom + gap;
      left = rect.left + rect.width / 2 - card.width / 2;
    }
    top = Math.max(12, Math.min(top, window.innerHeight - card.height - 12));
    left = Math.max(12, Math.min(left, window.innerWidth - card.width - 12));
    setPos({ top, left });
  }, [rect, step]);

  // Escape to skip.
  useEffect(() => {
    if (!running) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") finish(running);
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, idx]);

  function next() {
    if (!running) return;
    if (idx < running.steps.length - 1) setIdx((i) => i + 1);
    else finish(running);
  }
  function prev() {
    if (idx > 0) setIdx((i) => i - 1);
  }

  const pageKeyForHelp = keyForPath(pathname);

  return (
    <>
      {mounted && createPortal(
        <AnimatePresence>
          {running && step && rect && (
            <motion.div
              key="tour-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ position: "fixed", inset: 0, zIndex: 9998 }}
              onClick={() => finish(running)}
            >
              {/* dim layer with a spotlight cutout via box-shadow */}
              <motion.div
                initial={false}
                animate={{
                  top: rect.top - 6, left: rect.left - 6,
                  width: rect.width + 12, height: rect.height + 12,
                }}
                transition={{ type: "spring", stiffness: 340, damping: 32 }}
                style={{
                  position: "fixed",
                  borderRadius: 12,
                  boxShadow: "0 0 0 9999px rgba(12,12,16,.58)",
                  border: "2px solid #4F46E5",
                  pointerEvents: "none",
                }}
              />

              <motion.div
                ref={cardRef}
                onClick={(e) => e.stopPropagation()}
                initial={{ opacity: 0, scale: 0.96, y: 6 }}
                animate={{ opacity: pos ? 1 : 0, scale: 1, y: 0, top: pos?.top ?? -9999, left: pos?.left ?? -9999 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 360, damping: 30 }}
                style={{
                  position: "fixed",
                  width: 320,
                  background: "#fff",
                  borderRadius: 16,
                  border: "1px solid #ECECEC",
                  boxShadow: "0 24px 60px -20px rgba(24,24,27,.35)",
                  padding: 18,
                  zIndex: 9999,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                  <div style={{ display: "flex", gap: 4 }}>
                    {running.steps.map((_, i) => (
                      <span key={i} style={{ width: i === idx ? 16 : 6, height: 6, borderRadius: 99, background: i === idx ? "#4F46E5" : i < idx ? "#C7D2FE" : "#ECECEC", transition: "all .25s" }} />
                    ))}
                  </div>
                  <span style={{ fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF" }}>{idx + 1}/{running.steps.length}</span>
                </div>
                <div style={{ fontFamily: FONT.sans, fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em", color: "#18181B" }}>{step.title}</div>
                <p style={{ fontSize: 13, color: "#52525B", lineHeight: 1.55, margin: "8px 0 0" }}>{step.body}</p>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 16 }}>
                  <span onClick={() => finish(running)} style={{ fontFamily: FONT.body, fontSize: 12, fontWeight: 600, color: "#9CA3AF", cursor: "pointer" }}>Skip tour</span>
                  <div style={{ display: "flex", gap: 8 }}>
                    {idx > 0 && (
                      <button onClick={prev} style={{ fontFamily: FONT.body, fontSize: 12.5, fontWeight: 600, color: "#52525B", background: "#fff", border: "1px solid #ECECEC", padding: "7px 13px", borderRadius: 9, cursor: "pointer" }}>Back</button>
                    )}
                    <button onClick={next} style={{ fontFamily: FONT.body, fontSize: 12.5, fontWeight: 600, color: "#fff", background: "#4F46E5", border: "none", padding: "7px 14px", borderRadius: 9, cursor: "pointer" }}>
                      {idx === running.steps.length - 1 ? "Done" : "Next"}
                    </button>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      {/* floating help / replay button — present on every page */}
      <div data-tour="help-button" style={{ position: "fixed", bottom: 22, right: 22, zIndex: 9997 }}>
        <AnimatePresence>
          {helpOpen && (
            <motion.div
              initial={{ opacity: 0, y: 8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.96 }}
              transition={{ duration: 0.18 }}
              style={{ position: "absolute", bottom: "calc(100% + 10px)", right: 0, background: "#fff", border: "1px solid #ECECEC", borderRadius: 13, boxShadow: "0 16px 40px -16px rgba(24,24,27,.25)", padding: 6, width: 210 }}
            >
              {pageKeyForHelp && (
                <MenuItem label="Tour this page" onClick={() => { setHelpOpen(false); begin(pageKeyForHelp, { forceShell: false }); }} />
              )}
              {pageKeyForHelp && pageKeyForHelp !== "landing" && (
                <MenuItem label="Restart full walkthrough" onClick={() => { setHelpOpen(false); begin(pageKeyForHelp, { forceShell: true }); }} />
              )}
            </motion.div>
          )}
        </AnimatePresence>
        <motion.button
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.94 }}
          onClick={() => setHelpOpen((o) => !o)}
          style={{ width: 42, height: 42, borderRadius: "50%", background: "#18181B", color: "#fff", border: "none", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", boxShadow: "0 10px 28px -8px rgba(24,24,27,.5)" }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.1"><path d="M9.1 9a3 3 0 1 1 4.6 2.4c-.7.5-1.7 1.1-1.7 2.4" /><circle cx="12" cy="17.5" r="0.6" fill="#fff" /></svg>
        </motion.button>
      </div>
    </>
  );
}

function MenuItem({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <div onClick={onClick} className="gl-row" style={{ fontSize: 12.5, fontWeight: 500, color: "#18181B", padding: "8px 10px", borderRadius: 8, cursor: "pointer" }}>
      {label}
    </div>
  );
}
