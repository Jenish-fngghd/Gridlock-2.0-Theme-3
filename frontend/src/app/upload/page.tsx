"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { processFile, type ProcessResult } from "@/lib/api";
import { VIOLATION_LABELS, BAND_STYLE, BAND_LABELS, pct } from "@/lib/format";
import type { ViolationType, ConfidenceBand } from "@/lib/types";

type Phase = "idle" | "preview" | "processing" | "done" | "error";

export default function UploadPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
    setPhase("preview");
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) choose(f);
    },
    [choose]
  );

  async function analyze() {
    if (!file) return;
    setPhase("processing");
    try {
      const r = await processFile(file);
      setResult(r);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed");
      setPhase("error");
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setPhase("idle");
  }

  return (
    <div className="mx-auto max-w-3xl">
      <motion.header
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="mb-8"
      >
        <h1 className="font-display text-4xl font-bold tracking-tight text-slate-900">
          Upload Evidence
        </h1>
        <p className="mt-2 text-slate-500">
          Drop a traffic photo or short clip — the pipeline detects violations, reads plates and
          stores tamper-evident evidence.
        </p>
      </motion.header>

      {/* Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`surface relative cursor-pointer overflow-hidden rounded-3xl border-2 border-dashed p-10 text-center transition ${
          dragging ? "border-indigo-500 bg-indigo-50/60" : "border-slate-300"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*"
          hidden
          onChange={(e) => e.target.files?.[0] && choose(e.target.files[0])}
        />
        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mx-auto max-w-md"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="preview" className="mx-auto max-h-72 rounded-xl object-contain" />
              <div className="mt-3 truncate text-sm text-slate-500">{file?.name}</div>
            </motion.div>
          ) : (
            <motion.div key="prompt" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="font-display text-lg text-slate-700">
                Drag & drop, or click to browse
              </div>
              <div className="mt-1 text-sm text-slate-400">JPG / PNG / MP4 · up to 50 MB</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Actions */}
      <div className="mt-5 flex gap-3">
        <button
          onClick={analyze}
          disabled={!file || phase === "processing"}
          className="relative flex-1 overflow-hidden rounded-xl bg-indigo-600 px-6 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {phase === "processing" ? "Analyzing…" : "Run detection"}
        </button>
        {file && (
          <button
            onClick={reset}
            className="rounded-xl border border-slate-300 px-5 py-3 text-slate-600 transition hover:bg-slate-100"
          >
            Clear
          </button>
        )}
      </div>

      {/* Processing shimmer */}
      <AnimatePresence>
        {phase === "processing" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="shimmer relative mt-6 h-24 overflow-hidden rounded-2xl"
          />
        )}
      </AnimatePresence>

      {/* Error */}
      {phase === "error" && (
        <div className="mt-6 rounded-2xl border border-rose-200 bg-rose-50 p-5 text-rose-700">
          {error}
          <div className="mt-1 text-xs text-rose-500">
            Is the backend running on {process.env.NEXT_PUBLIC_BACKEND_URL}? Start it with{" "}
            <code>uvicorn backend.main:app</code>.
          </div>
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {phase === "done" && result && (
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-8"
          >
            <div className="mb-4 flex items-center gap-3">
              <h2 className="font-display text-xl font-semibold text-slate-900">
                {result.violations.length} violation{result.violations.length === 1 ? "" : "s"} detected
              </h2>
              <span className="text-xs text-slate-400">{result.processing_ms} ms</span>
              {!result.persisted && (
                <span className="rounded-md bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
                  not persisted (dev)
                </span>
              )}
            </div>

            <div className="space-y-3">
              {result.violations.map((v, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="surface surface-hover rounded-2xl p-5"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`rounded-md px-2 py-1 text-xs ${
                        BAND_STYLE[v.confidence_band as ConfidenceBand] ?? ""
                      }`}
                    >
                      {BAND_LABELS[v.confidence_band as ConfidenceBand] ?? v.confidence_band}
                    </span>
                    <span className="text-lg font-medium text-slate-900">
                      {VIOLATION_LABELS[v.violation_type as ViolationType] ?? v.violation_type}
                    </span>
                    <span className="ml-auto font-plate text-sm text-slate-500">
                      {pct(v.confidence)}
                    </span>
                  </div>
                  {v.vlm_caption && <p className="mt-2 text-sm text-slate-500">{v.vlm_caption}</p>}
                  {v.plate && (
                    <div className="mt-3 inline-flex items-center gap-2">
                      <span className="text-xs text-slate-400">Plate</span>
                      <span className="font-plate rounded-md border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-indigo-700">
                        {v.plate.plate_text}
                      </span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
