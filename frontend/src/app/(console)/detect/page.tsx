"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { processFile, type ProcessResult, type Detection, BACKEND_URL } from "@/lib/api";
import { VIOLATION_LABELS, pct } from "@/lib/format";
import { FONT, severityFor, SEVERITY_COLOR, boxColor, prettyClass } from "@/lib/ui";
import type { ViolationType } from "@/lib/types";

type Phase = "idle" | "processing" | "result" | "error";

// The rail mirrors the real ml/ pipeline stages (see backend _build_stages).
const STEP_LABELS = [
  "Uploading", "Object detection", "Plate detection & OCR", "Rider / helmet & triple-riding",
  "Seatbelt classifier", "Signal / wrong-side / red-light", "Compiling evidence",
];
const STEP_SUBS = ["", "RF-DETR", "TrOCR", "pose + count", "windshield→cls", "phase + lane", "evidence"];
const STEP_MS = 620;

export default function DetectPage() {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>("idle");
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [step, setStep] = useState(-1);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const filmTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopFilm = useCallback(() => {
    if (filmTimer.current) {
      clearTimeout(filmTimer.current);
      filmTimer.current = null;
    }
  }, []);

  useEffect(() => stopFilm, [stopFilm]);

  // Advance the rail through the stages, then HOLD on the last stage (spinning) until the
  // real backend response arrives — so the animation tracks real work, not a fixed timer.
  const startFilm = useCallback(() => {
    let i = 0;
    setStep(0);
    const tick = () => {
      i++;
      if (i <= STEP_LABELS.length - 1) {
        setStep(i);
        if (i < STEP_LABELS.length - 1) filmTimer.current = setTimeout(tick, STEP_MS);
      }
    };
    filmTimer.current = setTimeout(tick, STEP_MS);
  }, []);

  const analyze = useCallback(async (file: File) => {
    setFrameSrc((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return file.type.startsWith("image") ? URL.createObjectURL(file) : null;
    });
    setResult(null);
    setError("");
    setPhase("processing");
    startFilm();
    try {
      const r = await processFile(file);
      stopFilm();
      setStep(STEP_LABELS.length); // all stages done
      setResult(r);
      setPhase("result");
    } catch (e) {
      stopFilm();
      setError(e instanceof Error ? e.message : "Processing failed");
      setPhase("error");
    }
  }, [startFilm, stopFilm]);

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) analyze(f);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) analyze(f);
  };
  const reset = () => {
    stopFilm();
    setFrameSrc((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    setResult(null);
    setError("");
    setStep(-1);
    setPhase("idle");
  };

  const isProcessing = phase === "processing";
  const isResult = phase === "result";
  const masterPct = isResult ? 100 : Math.min(95, Math.max(0, Math.round(((step + 1) / STEP_LABELS.length) * 100)));

  const detections: Detection[] = result?.detections ?? [];
  const annotated = result?.annotated_image_url ?? null;
  const displaySrc = frameSrc ?? annotated;
  const drawBoxes = isResult && frameSrc != null && detections.length > 0;
  const plate = result?.violations.find((v) => v.plate)?.plate ?? null;
  const objCount = detections.length;
  const vCount = result?.violations.length ?? 0;

  return (
    <div style={{ padding: "30px 36px 48px", maxWidth: 1180, margin: "0 auto" }}>
      {phase === "idle" && (
        <div style={{ position: "relative", overflow: "hidden", borderRadius: 24 }}>
          <div aria-hidden style={{ position: "absolute", inset: "-25% -12%", zIndex: 0, pointerEvents: "none", background: "radial-gradient(38% 48% at 18% 28%,rgba(79,70,229,.16),transparent 70%),radial-gradient(34% 42% at 82% 22%,rgba(14,165,233,.13),transparent 70%),radial-gradient(44% 50% at 62% 88%,rgba(139,92,246,.13),transparent 70%)", filter: "blur(10px)", animation: "meshdrift 16s ease-in-out infinite" }} />
          <div style={{ position: "relative", zIndex: 1, textAlign: "center", maxWidth: 780, margin: "0 auto", padding: "30px 20px 12px" }}>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#fff", border: "1px solid #ECECEC", borderRadius: 999, padding: "6px 13px 6px 9px", fontSize: 12.5, fontWeight: 500, color: "#52525B", boxShadow: "0 1px 2px rgba(24,24,27,.04)", animation: "wordup .6s cubic-bezier(.16,1,.3,1) both" }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#10B981", animation: "pulse 2s ease-in-out infinite" }} />
              Live vision model · RF-DETR + TrOCR
            </div>
            <h1 style={{ fontFamily: FONT.sans, fontSize: 54, lineHeight: 1.06, fontWeight: 600, letterSpacing: "-0.03em", margin: "18px 0 0" }}>
              Catch every violation in one{" "}
              <span style={{ background: "linear-gradient(90deg,#4F46E5,#0EA5E9,#8B5CF6,#10B981,#4F46E5)", backgroundSize: "300% auto", WebkitBackgroundClip: "text", backgroundClip: "text", WebkitTextFillColor: "transparent", animation: "shine 5s linear infinite" }}>frame</span>
            </h1>
            <p style={{ fontSize: 16.5, lineHeight: 1.55, color: "#6B7280", maxWidth: 520, margin: "18px auto 0" }}>
              Upload a traffic frame — Gridlock runs the full pipeline, reads plates and flags every violation in a single pass.
            </p>

            <div
              data-tour="detect-dropzone"
              onClick={() => inputRef.current?.click()}
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              className="gl-press"
              style={{ position: "relative", overflow: "hidden", margin: "30px auto 0", maxWidth: 680, border: "2px dashed rgba(79,70,229,.42)", borderRadius: 20, background: "#fff", padding: "44px 32px", cursor: "pointer", boxShadow: "0 1px 2px rgba(24,24,27,.04)" }}
            >
              <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 13 }}>
                <div style={{ width: 56, height: 56, borderRadius: 16, background: "linear-gradient(135deg,#4F46E5,#6366F1)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 10px 26px -10px rgba(79,70,229,.7)", animation: "float 4.5s ease-in-out infinite" }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 16V5M7 10l5-5 5 5" /><path d="M5 19h14" /></svg>
                </div>
                <div style={{ fontFamily: FONT.sans, fontSize: 19, fontWeight: 600, color: "#18181B" }}>Drag &amp; drop a traffic frame</div>
                <div style={{ fontFamily: FONT.mono, fontSize: 12, color: "#9CA3AF" }}>or click to browse · JPG · PNG · up to 20 MB</div>
              </div>
              <input ref={inputRef} type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 14, marginTop: 18 }}>
              <button data-tour="detect-browse" onClick={() => inputRef.current?.click()} className="gl-press" style={{ display: "inline-flex", alignItems: "center", gap: 8, fontFamily: FONT.body, fontSize: 13.5, fontWeight: 600, color: "#4F46E5", background: "#EEF0FF", border: "1px solid #E0E2FF", padding: "10px 18px", borderRadius: 11, cursor: "pointer" }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="#4F46E5"><path d="M8 5v14l11-7z" /></svg>Browse a frame
              </button>
              <span style={{ fontFamily: FONT.mono, fontSize: 11.5, color: "#9CA3AF" }}>runs the full 7-stage pipeline on AWS</span>
            </div>
          </div>
        </div>
      )}

      {(isProcessing || isResult) && (
        <div>
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", animation: "bfu .5s cubic-bezier(.16,1,.3,1) both" }}>
            <div>
              <h1 style={{ fontFamily: FONT.sans, fontSize: 28, fontWeight: 600, letterSpacing: "-0.025em", margin: 0 }}>Detect</h1>
              {isProcessing && (
                <p style={{ fontSize: 14, color: "#0EA5E9", fontWeight: 500, margin: "6px 0 0", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 12, height: 12, border: "2px solid #0EA5E9", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                  The model is looking — running the pipeline…
                </p>
              )}
              {isResult && (
                <p style={{ fontSize: 14, color: "#10B981", fontWeight: 600, margin: "6px 0 0", display: "flex", alignItems: "center", gap: 7 }}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="2.6"><path d="M20 6 9 17l-5-5" /></svg>
                  Detection complete · {objCount} object{objCount === 1 ? "" : "s"} · {result?.processing_ms ?? 0} ms
                </p>
              )}
            </div>
            <button onClick={reset} className="gl-press gl-btn-ghost" style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "9px 16px", borderRadius: 11, cursor: "pointer" }}>Analyze another</button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.42fr 1fr", gap: 20, marginTop: 22, alignItems: "start" }}>
            {/* frame */}
            <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 14, boxShadow: "0 1px 2px rgba(24,24,27,.04)", animation: "springpop .65s cubic-bezier(.16,1,.3,1) both" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px 12px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10B981", animation: "ringpulse 2.4s infinite" }} />
                  <span style={{ fontFamily: FONT.mono, fontSize: 11.5, fontWeight: 500, color: "#52525B" }}>UPLOAD · LIVE PIPELINE</span>
                </div>
                <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF" }}>{result?.model_version ?? "pipeline"}</span>
              </div>

              <div style={{ position: "relative", borderRadius: 13, overflow: "hidden", background: "#0d0d12", minHeight: displaySrc ? undefined : 280 }}>
                {displaySrc ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={displaySrc}
                    alt="frame"
                    style={{ display: "block", width: "100%", height: "auto", transition: "filter .85s ease", filter: isProcessing ? "blur(7px) brightness(0.6) saturate(0.85)" : "none" }}
                  />
                ) : (
                  <div style={{ aspectRatio: "16 / 10", display: "flex", alignItems: "center", justifyContent: "center", color: "#52525B", fontFamily: FONT.mono, fontSize: 12 }}>
                    no preview · video frame
                  </div>
                )}

                {/* real bounding boxes (normalized → stretched to image rect) */}
                {drawBoxes && (
                  <>
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}>
                      {detections.map((d, i) => (
                        <rect
                          key={i}
                          x={d.bbox.x * 100}
                          y={d.bbox.y * 100}
                          width={d.bbox.w * 100}
                          height={d.bbox.h * 100}
                          rx={1}
                          fill="none"
                          stroke={boxColor(d.class_label)}
                          strokeWidth={1.6}
                          vectorEffect="non-scaling-stroke"
                          style={{ animation: "chipin .45s cubic-bezier(.16,1,.3,1) both", animationDelay: `${(i * 0.09).toFixed(2)}s` }}
                        />
                      ))}
                    </svg>
                    {detections.map((d, i) => {
                      const c = boxColor(d.class_label);
                      const labelTop = d.bbox.y <= 0.06; // flip below the box when too near the top
                      return (
                        <div
                          key={i}
                          style={{
                            position: "absolute",
                            left: `${d.bbox.x * 100}%`,
                            top: labelTop ? `${(d.bbox.y + d.bbox.h) * 100}%` : `${d.bbox.y * 100}%`,
                            transform: labelTop ? "translateY(3px)" : "translateY(-100%) translateY(-3px)",
                            background: c,
                            color: "#fff",
                            fontFamily: FONT.mono,
                            fontSize: 9.5,
                            fontWeight: 600,
                            padding: "2px 6px",
                            borderRadius: 5,
                            whiteSpace: "nowrap",
                            pointerEvents: "none",
                            animation: "chipin .45s cubic-bezier(.16,1,.3,1) both",
                            animationDelay: `${(0.2 + i * 0.09).toFixed(2)}s`,
                          }}
                        >
                          {prettyClass(d.class_label)}{" "}
                          <span style={{ opacity: 0.85 }}>{d.confidence != null ? d.confidence.toFixed(2) : ""}</span>
                        </div>
                      );
                    })}
                  </>
                )}

                {/* processing overlay */}
                {isProcessing && (
                  <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                    <div style={{ position: "absolute", inset: 0, backgroundImage: "linear-gradient(rgba(14,165,233,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(14,165,233,.07) 1px,transparent 1px)", backgroundSize: "34px 34px" }} />
                    <div style={{ position: "absolute", left: 0, right: 0, height: 3, background: "linear-gradient(90deg,transparent,#0EA5E9,transparent)", boxShadow: "0 0 20px 4px rgba(14,165,233,.7)", animation: "scanline 1.6s ease-in-out infinite" }} />
                    <div style={{ position: "absolute", top: 14, right: 14, fontFamily: FONT.mono, fontSize: 10.5, color: "#0EA5E9", display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 9, height: 9, border: "2px solid #0EA5E9", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />RUNNING INFERENCE
                    </div>
                  </div>
                )}
              </div>

              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 6px 2px" }}>
                <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF" }}>{result?.persisted === false ? "NOT PERSISTED (dev)" : isResult ? "EVIDENCE STORED" : "…"}</span>
                {isResult && (
                  <span style={{ fontFamily: FONT.mono, fontSize: 11.5, color: "#52525B" }}>
                    <span style={{ color: "#EF4444", fontWeight: 600 }}>{vCount}</span> violations · <span style={{ color: "#18181B", fontWeight: 600 }}>{objCount}</span> objects
                  </span>
                )}
              </div>
            </div>

            {/* right column */}
            <div>
              {isProcessing && (
                <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 18, boxShadow: "0 1px 2px rgba(24,24,27,.04)", animation: "bfl .55s cubic-bezier(.16,1,.3,1) both" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 7 }}>
                    <span style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF" }}>PIPELINE</span>
                    <span style={{ fontFamily: FONT.mono, fontSize: 12, fontWeight: 600, color: "#4F46E5" }}>{masterPct}%</span>
                  </div>
                  <div style={{ height: 7, borderRadius: 4, background: "#F1F1F3", overflow: "hidden", position: "relative", marginBottom: 14 }}>
                    <div style={{ height: "100%", borderRadius: 4, background: "linear-gradient(90deg,#6366F1,#4F46E5)", width: `${masterPct}%`, transition: "width .55s cubic-bezier(.16,1,.3,1)", position: "relative", overflow: "hidden" }}>
                      <div style={{ position: "absolute", inset: 0, width: "40%", background: "linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent)", animation: "barshimmer 1.3s linear infinite" }} />
                    </div>
                  </div>
                  {STEP_LABELS.map((label, i) => {
                    const isDone = i < step;
                    const isActive = i === step;
                    return (
                      <div key={label} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", transition: "opacity .4s,transform .4s", opacity: isActive ? 1 : isDone ? 0.5 : 0.55, transform: isActive ? "scale(1)" : isDone ? "scale(.97)" : "scale(1)", transformOrigin: "left center" }}>
                        <div style={{ width: 22, height: 22, flex: "none", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          {isDone ? (
                            <span style={{ width: 18, height: 18, borderRadius: "50%", background: "#ECFDF5", display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="3.2"><path d="M20 6 9 17l-5-5" /></svg>
                            </span>
                          ) : isActive ? (
                            <span style={{ width: 16, height: 16, border: "2px solid #4F46E5", borderTopColor: "transparent", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
                          ) : (
                            <span style={{ width: 9, height: 9, borderRadius: "50%", background: "#D4D4D8" }} />
                          )}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                            <span style={{ fontSize: 13, fontWeight: 600, color: isActive ? "#18181B" : isDone ? "#6B7280" : "#9CA3AF" }}>{label}</span>
                            <span style={{ fontFamily: FONT.mono, fontSize: 9, color: "#B4B4BB" }}>{STEP_SUBS[i]}</span>
                          </div>
                          <div style={{ marginTop: 6, height: 5, borderRadius: 3, background: "#F1F1F3", overflow: "hidden" }}>
                            <div style={{ height: "100%", borderRadius: 3, background: "linear-gradient(90deg,#34D399,#10B981)", width: isDone || isActive ? "100%" : "0%", transition: "width .62s linear" }} />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {isResult && result && (
                <div style={{ animation: "bfl .6s cubic-bezier(.16,1,.3,1) both" }}>
                  {/* plate OCR */}
                  <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 16, boxShadow: "0 1px 2px rgba(24,24,27,.04)", animation: "springpop .6s cubic-bezier(.16,1,.3,1) both" }}>
                    <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF", marginBottom: 11 }}>PLATE OCR · TrOCR</div>
                    <div style={{ position: "relative", borderRadius: 11, overflow: "hidden", background: "#fff", border: "1px solid #E4E4E7", padding: "14px 12px", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "inset 0 1px 4px rgba(0,0,0,.06)" }}>
                      <span style={{ position: "relative", fontFamily: FONT.mono, fontSize: 25, fontWeight: 600, letterSpacing: ".07em", color: plate ? "#18181B" : "#9CA3AF" }}>
                        {plate?.plate_text ?? "no plate read"}
                      </span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 11, fontFamily: FONT.mono, fontSize: 11, color: "#6B7280" }}>
                      <span>STATE <span style={{ color: "#18181B", fontWeight: 600 }}>{plate?.state_code ?? "—"}</span></span>
                      <span>OBJECTS <span style={{ color: "#10B981", fontWeight: 600 }}>{objCount}</span></span>
                    </div>
                  </div>

                  {/* real pipeline stages recap */}
                  {result.stages.length > 0 && (
                    <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", marginTop: 14, boxShadow: "0 1px 2px rgba(24,24,27,.04)" }}>
                      <div style={{ padding: "12px 16px", borderBottom: "1px solid #F4F4F5", fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF" }}>PIPELINE STAGES</div>
                      {result.stages.map((s, i) => (
                        <div key={s.key} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 16px", borderBottom: i < result.stages.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                          <span style={{ width: 16, height: 16, flex: "none", borderRadius: "50%", background: s.ran ? "#ECFDF5" : "#F4F4F5", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            {s.ran && <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="3.4"><path d="M20 6 9 17l-5-5" /></svg>}
                          </span>
                          <span style={{ fontSize: 12.5, fontWeight: 500, color: "#18181B", flex: "none" }}>{s.label}</span>
                          <span style={{ marginLeft: "auto", fontFamily: FONT.mono, fontSize: 10.5, color: "#9CA3AF", textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 130 }}>{s.detail}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* flagged violations */}
                  <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, overflow: "hidden", marginTop: 14, boxShadow: "0 1px 2px rgba(24,24,27,.04)" }}>
                    <div style={{ padding: "12px 16px", borderBottom: "1px solid #F4F4F5", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <span style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".08em", color: "#9CA3AF" }}>FLAGGED VIOLATIONS</span>
                      <span style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 600, color: "#EF4444", background: "#FEF2F2", padding: "1px 7px", borderRadius: 6 }}>{vCount}</span>
                    </div>
                    {vCount === 0 ? (
                      <div style={{ padding: "20px 16px", fontSize: 13.5, color: "#6B7280" }}>No violations detected in this frame.</div>
                    ) : (
                      result.violations.map((v, i) => {
                        const sev = severityFor(v.violation_type);
                        const col = SEVERITY_COLOR[sev];
                        return (
                          <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "13px 16px", borderBottom: i < result.violations.length - 1 ? "1px solid #F4F4F5" : "none", animation: "bfu .5s cubic-bezier(.16,1,.3,1) both", animationDelay: `${(0.12 + i * 0.1).toFixed(2)}s` }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                              <span style={{ width: 9, height: 9, borderRadius: "50%", flex: "none", background: col.c }} />
                              <span style={{ fontSize: 13.5, fontWeight: 500, color: "#18181B" }}>{VIOLATION_LABELS[v.violation_type as ViolationType] ?? v.violation_type}</span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "none" }}>
                              <span style={{ fontFamily: FONT.mono, fontSize: 11.5, fontWeight: 600, color: "#6B7280" }}>{pct(v.confidence)}</span>
                              <span style={{ fontFamily: FONT.body, fontSize: 10.5, fontWeight: 600, color: col.c, background: col.b, padding: "2px 8px", borderRadius: 6 }}>{sev}</span>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>

                  <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                    <button onClick={() => router.push("/violations")} className="gl-press gl-btn-primary" style={{ flex: 1, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8, fontFamily: FONT.body, fontSize: 13.5, fontWeight: 600, color: "#fff", border: "none", padding: 12, borderRadius: 11, cursor: "pointer" }}>
                      View full report
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
                    </button>
                    <button onClick={reset} className="gl-press gl-btn-ghost" style={{ flex: "none", fontFamily: FONT.body, fontSize: 13.5, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "12px 16px", borderRadius: 11, cursor: "pointer" }}>Analyze another</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {phase === "error" && (
        <div style={{ marginTop: 24, maxWidth: 680 }}>
          <div style={{ borderRadius: 16, border: "1px solid #FECACA", background: "#FEF2F2", padding: 20, color: "#B91C1C" }}>
            <div style={{ fontFamily: FONT.sans, fontWeight: 600, fontSize: 16 }}>Processing failed</div>
            <div style={{ marginTop: 6, fontSize: 13.5 }}>{error}</div>
            <div style={{ marginTop: 8, fontSize: 12, color: "#DC2626" }}>
              Is the backend reachable at <code>{BACKEND_URL}</code>?
            </div>
            <button onClick={reset} className="gl-press" style={{ marginTop: 14, fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#fff", background: "#18181B", border: "none", padding: "9px 16px", borderRadius: 10, cursor: "pointer" }}>Try again</button>
          </div>
        </div>
      )}
    </div>
  );
}
