"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import { supabase, storageUrl } from "@/lib/supabase";
import { FONT } from "@/lib/ui";
import { requestTour } from "@/lib/tourBus";
import Brand from "@/components/Brand";
import Reveal from "@/components/Reveal";
import CountUp from "@/components/CountUp";

interface HeroBox { x: number; y: number; w: number; h: number }
interface HeroData {
  imgUrl: string;
  cameraLabel: string;
  timestamp: string;
  violationCount: number;
  objectCount: number;
  status: string;
  boxes: HeroBox[]; // violation bboxes, absolute pixels in the source image's own coordinates
}

const serif: React.CSSProperties = { fontFamily: FONT.serif, fontWeight: 400, fontStyle: "italic" };

const FEATURES = [
  { code: "NO_HELMET", title: "No-helmet riding", desc: "Detects bare-headed riders and pillions across two-wheelers.", bg: "#FEF2F2", dot: "#EF4444" },
  { code: "TRIPLE_RIDE", title: "Triple riding", desc: "Counts occupants per vehicle and flags 3+ on a single ride.", bg: "#FFFBEB", dot: "#F59E0B" },
  { code: "NO_SEATBELT", title: "No seatbelt", desc: "Reads driver and front-passenger belts through the windshield.", bg: "#FFFBEB", dot: "#F59E0B" },
  { code: "WRONG_SIDE", title: "Wrong-side driving", desc: "Tracks heading vectors against lane direction to catch counterflow.", bg: "#FEF2F2", dot: "#EF4444" },
  { code: "RED_LIGHT", title: "Red-light jump", desc: "Syncs signal phase with stop-line crossing to catch jumps.", bg: "#FEF2F2", dot: "#EF4444" },
  { code: "ILLEGAL_PARK", title: "Illegal parking", desc: "Flags vehicles in no-parking zones and on yellow-line stretches.", bg: "#F0F9FF", dot: "#0EA5E9" },
];

const STEPS = [
  { n: "01", t: "Ingest frame", d: "Pull from a live camera feed or drop in an uploaded image." },
  { n: "02", t: "Restore & gate", d: "Quality gate + learned low-light / deblur restoration when needed." },
  { n: "03", t: "Detect", d: "RF-DETR finds vehicles, riders, plates and signals in one pass." },
  { n: "04", t: "Read plate", d: "TrOCR decodes the number plate from the cropped region." },
  { n: "05", t: "Score & verify", d: "Confidence cascade + VLM verification assign a calibrated band." },
  { n: "06", t: "Build evidence", d: "Signed, hash-chained bundle — challan-ready and reproducible." },
];

const CHIPS = [
  "RF-DETR Detection", "TrOCR Plate OCR", "7 Violation Classes", "Tamper-evident Evidence",
  "Confidence Cascade", "VLM Verification", "Retinexformer Restore", "Realtime Console",
];

// Curated wrong-side violation used as the hero image. The env var lets us rotate it without a
// code push; the hardcoded fallback ensures the right frame shows even when the var isn't set
// (Vercel preview builds, local dev without .env.local). Never falls through to "most recent".
const PINNED_HERO_JOB_ID = process.env.NEXT_PUBLIC_HERO_JOB_ID ?? "35c4c7ff-5160-41e3-a9c1-c659ab397b87";

const FAQ = [
  { q: "How accurate is the violation detection?", a: "On a held-out benchmark of 5,000 IDD images the detector reaches mAP@0.5 ≈ 0.59 — RF-DETR for live inference, with SAM-3 used offline to auto-label rare classes (autorickshaw, vehicle-fallback) so the detector sees more of them in training. Every detection still carries a per-class confidence so reviewers can set their own thresholds. We report honest, benchmarked numbers — never a headline figure we can't reproduce." },
  { q: "Which violations can it flag in a single pass?", a: "Seven classes — no-helmet, triple-riding, no-seatbelt, wrong-side, stop-line, red-light and illegal-parking — plus number-plate OCR and per-subject evidence for every flagged vehicle, rider and driver." },
  { q: "Does it read number plates reliably?", a: "The ANPR head reaches ~78% exact-match on our plate benchmark — roughly +33 points over a PaddleOCR baseline — and returns the raw crop alongside the decoded string for audit." },
  { q: "How are red-light and wrong-side detected?", a: "A per-camera scene-context model encodes the stop-line, lane vectors and signal ROI. Signal-state classification runs at 99.7% on the LISA benchmark; wrong-side and red-light reach F1 ≈ 0.96 / 0.90 respectively." },
  { q: "Can the output be used as legal evidence?", a: "Every detection bundles the cropped frame, timestamp, camera ID and confidence into a signed, hash-chained record — append-only and reproducible from the immutable evidence audit trail." },
  { q: "What runs the pipeline?", a: "A FastAPI backend on AWS handles inference and persistence; Supabase stores violations, plates and the model registry with realtime updates straight into this console." },
];

export default function Landing() {
  const router = useRouter();
  const [openFaq, setOpenFaq] = useState<number>(0);
  const [sent, setSent] = useState(false);
  const [mapPct, setMapPct] = useState<number | null>(null);
  const [hero, setHero] = useState<HeroData | null>(null);

  useEffect(() => {
    supabase
      .from("model_registry")
      .select("module, metric_name, metric_value, is_active")
      .eq("is_active", true)
      .then(({ data }) => {
        const det = (data ?? []).find(
          (m) => /detect/i.test(m.module ?? "") && m.metric_value != null
        );
        if (det?.metric_value != null) {
          setMapPct(det.metric_value <= 1 ? det.metric_value * 100 : det.metric_value);
        } else {
          setMapPct(58.6);
        }
      });
  }, []);

  useEffect(() => {
    (async () => {
      const pinnedJobId = PINNED_HERO_JOB_ID;
      const query = supabase
        .from("violations")
        .select("job_id, status, detected_at, annotated_image_path, cameras(name, location_name)")
        .not("annotated_image_path", "is", null)
        .eq("job_id", pinnedJobId);
      const { data: vRows } = await query.limit(1);

      const row = vRows?.[0] as {
        job_id: string; status: string; detected_at: string; annotated_image_path: string;
        cameras: { name: string; location_name: string | null } | null;
      } | undefined;
      if (!row) return;
      const imgUrl = storageUrl("annotated", row.annotated_image_path);
      if (!imgUrl) return;

      const [{ data: jobViolations }, { count: objectCount }] = await Promise.all([
        supabase.from("violations").select("evidence").eq("job_id", row.job_id),
        supabase.from("detections").select("id", { count: "exact", head: true }).eq("job_id", row.job_id),
      ]);
      const boxes: HeroBox[] = (jobViolations ?? [])
        .map((v) => (v.evidence as { bbox?: number[] } | null)?.bbox)
        .filter((bb): bb is number[] => Array.isArray(bb) && bb.length === 4)
        .map(([x, y, w, h]) => ({ x, y, w, h }));

      setHero({
        imgUrl,
        cameraLabel: row.cameras?.location_name ?? row.cameras?.name ?? "Live camera",
        timestamp: new Date(row.detected_at).toLocaleTimeString([], { hour12: false }),
        violationCount: jobViolations?.length ?? 1,
        objectCount: objectCount ?? 0,
        status: row.status === "pending" ? "Pending review" : "Auto-confirmed",
        boxes,
      });
    })();
  }, []);

  // Natural pixel size of the hero image -- needed so the animated box overlay below aligns
  // exactly with the violation's own box already burned into the image, even though the image
  // is displayed with object-fit: cover (cropped to fill a 16:10 frame).
  const [heroNatural, setHeroNatural] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    setHeroNatural(null);
    if (!hero?.imgUrl) return;
    const el = new window.Image();
    el.onload = () => setHeroNatural({ w: el.naturalWidth, h: el.naturalHeight });
    el.src = hero.imgUrl;
  }, [hero?.imgUrl]);
  const ready = !!hero && !!heroNatural;

  // Crop the SVG viewBox in around the violation's own box instead of showing the full wide
  // frame -- a modest zoom, not a tight close-up. Computed from real coordinates (not a fixed
  // px offset) so it centers correctly regardless of image resolution or where the box sits.
  const heroZoom = (() => {
    if (!heroNatural || !hero?.boxes.length) return null;
    const b = hero.boxes[0];
    const aspect = 1.6; // matches the hero frame's own 16:10 box
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    let h = Math.max(b.h * 2.2, (b.w * 2.2) / aspect, heroNatural.h * 0.3);
    let w = h * aspect;
    if (w > heroNatural.w) { w = heroNatural.w; h = w / aspect; }
    const x = Math.max(0, Math.min(cx - w / 2, heroNatural.w - w));
    const y = Math.max(0, Math.min(cy - h / 2, heroNatural.h - h));
    return { x, y, w, h };
  })();

  const jump = (id: string) => {
    const el = document.getElementById(id);
    if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 70, behavior: "smooth" });
  };

  return (
    <div style={{ minHeight: "100vh", background: "#FAFAFA", color: "#18181B", fontFamily: FONT.body }}>
      {/* nav */}
      <div style={{ position: "sticky", top: 0, zIndex: 50, backdropFilter: "blur(14px)", background: "rgba(250,250,250,.78)", borderBottom: "1px solid #ECECEC" }}>
        <div data-tour="landing-nav" style={{ maxWidth: 1180, margin: "0 auto", padding: "15px 32px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Brand />
          <div style={{ display: "flex", alignItems: "center", gap: 32, fontSize: 14, color: "#52525B", fontWeight: 500 }}>
            <span className="gl-link" onClick={() => jump("features")}>Features</span>
            <span className="gl-link" onClick={() => jump("workflow")}>Workflow</span>
            <span className="gl-link" onClick={() => jump("stack")}>Stack</span>
            <span className="gl-link" onClick={() => jump("faq")}>FAQ</span>
            <span className="gl-link" onClick={() => requestTour({ key: "landing", forceShell: false })}>Take the tour</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <span onClick={() => router.push("/dashboard")} style={{ fontSize: 14, fontWeight: 500, color: "#52525B", cursor: "pointer" }}>Sign in</span>
            <button onClick={() => jump("demo")} className="gl-press gl-btn-dark" style={{ fontFamily: FONT.body, fontSize: 13.5, fontWeight: 600, color: "#fff", background: "#18181B", border: "none", padding: "9px 16px", borderRadius: 10, cursor: "pointer", boxShadow: "0 1px 2px rgba(24,24,27,.2)" }}>Request demo</button>
          </div>
        </div>
      </div>

      {/* hero */}
      <div style={{ maxWidth: 1180, margin: "0 auto", padding: "88px 32px 48px", display: "grid", gridTemplateColumns: "1.04fr .96fr", gap: 54, alignItems: "center" }}>
        <div>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#fff", border: "1px solid #ECECEC", borderRadius: 999, padding: "6px 13px 6px 9px", fontSize: 12.5, fontWeight: 500, color: "#52525B", boxShadow: "0 1px 2px rgba(24,24,27,.04)" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#10B981", animation: "pulse 2s ease-in-out infinite" }} />
            Computer vision for traffic enforcement
          </div>
          <h1 style={{ fontFamily: FONT.sans, fontSize: 60, lineHeight: 1.04, margin: "18px 0 0", fontWeight: 600, letterSpacing: "-0.03em" }}>
            Read every traffic <span style={{ ...serif, letterSpacing: "-0.01em" }}>violation</span> in a single frame.
          </h1>
          <p style={{ fontSize: 17.5, lineHeight: 1.55, color: "#6B7280", maxWidth: 480, margin: "22px 0 0" }}>
            Padlock reads any traffic image, flags seven violation classes, reads number plates, and pinpoints every rider, driver and vehicle — with calibrated, auditable evidence.
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 30 }}>
            <button data-tour="landing-cta" onClick={() => router.push("/detect")} className="gl-press gl-btn-primary" style={{ display: "inline-flex", alignItems: "center", gap: 9, fontFamily: FONT.body, fontSize: 15, fontWeight: 600, color: "#fff", border: "none", padding: "13px 22px", borderRadius: 12, cursor: "pointer" }}>
              Launch console
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
            </button>
            <button onClick={() => jump("demo")} className="gl-press gl-btn-ghost" style={{ fontFamily: FONT.body, fontSize: 15, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "13px 22px", borderRadius: 12, cursor: "pointer" }}>Request a demo</button>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 22, marginTop: 30, flexWrap: "wrap" }}>
            {["Built for traffic authorities", "Court-ready evidence trail", "Honest, benchmarked metrics"].map((t) => (
              <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13.5, color: "#52525B", fontWeight: 500 }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" strokeWidth="2.6"><path d="M20 6 9 17l-5-5" /></svg>{t}
              </span>
            ))}
          </div>
        </div>

        {/* hero visual */}
        <div style={{ position: "relative", animation: "bfu 1s cubic-bezier(.16,1,.3,1) both", animationDelay: ".18s" }}>
          <div style={{ position: "absolute", inset: "24px -18px -18px 30px", background: "#fff", border: "1px solid #ECECEC", borderRadius: 20, transform: "rotate(2.4deg)", boxShadow: "0 24px 60px -28px rgba(24,24,27,.18)" }} />
          <div style={{ position: "relative", background: "#fff", border: "1px solid #ECECEC", borderRadius: 20, padding: 14, boxShadow: "0 36px 70px -30px rgba(24,24,27,.3)" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 6px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10B981", animation: "ringpulse 2.4s infinite" }} />
                <span style={{ fontFamily: FONT.mono, fontSize: 11.5, fontWeight: 500, color: "#52525B" }}>{hero ? hero.cameraLabel.toUpperCase() : "CAM-07 · NH-48 JUNCTION"}</span>
              </div>
              <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF" }}>{hero?.timestamp ?? "14:08:22"}</span>
            </div>
            <div style={{ position: "relative", borderRadius: 13, overflow: "hidden", aspectRatio: "16 / 10", background: ready ? "#0d0d12" : "repeating-linear-gradient(48deg,#1c1c20,#1c1c20 11px,#212126 11px,#212126 22px)" }}>
              <AnimatePresence mode="wait">
                {ready ? (
                  // hero.imgUrl is the server-rendered evidence image (EvidenceGenerator), which
                  // already has the violation's own box + label burned in -- same as the
                  // Detect/Violations pages. The generic-object client overlay that used to be
                  // drawn on top is gone (it clashed with the baked-in box), but we still want
                  // the signature "box draws itself in" flourish on load, so an SVG re-draws an
                  // animated outline exactly on top of the real box. Using an SVG <image> +
                  // viewBox (rather than a plain <img> + % positioned overlay) keeps the overlay
                  // pixel-aligned with the baked-in box even though the frame is cropped via
                  // object-fit: cover -- preserveAspectRatio="xMidYMid slice" is the SVG
                  // equivalent of "cover", in the same coordinate space as the overlay rects.
                  // Gated on `ready` (hero data AND the image's natural size both resolved, i.e.
                  // it's already decoded in the browser) and only swapped in via the fade below,
                  // so there's never a blank/half-loaded frame visible -- the placeholder mockup
                  // stays up the entire time the real photo is still loading.
                  <motion.div key="real" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} style={{ position: "absolute", inset: 0 }}>
                    <svg viewBox={heroZoom ? `${heroZoom.x} ${heroZoom.y} ${heroZoom.w} ${heroZoom.h}` : `0 0 ${heroNatural!.w} ${heroNatural!.h}`} preserveAspectRatio="xMidYMid slice" width="100%" height="100%" style={{ display: "block" }}>
                      <image href={hero!.imgUrl} x={0} y={0} width={heroNatural!.w} height={heroNatural!.h} />
                      {hero!.boxes.map((b, i) => (
                        // One-time settle-into-place: box starts slightly below its real
                        // position and fades/slides up to where it actually belongs, rather
                        // than tracing its own outline. Offset is sized relative to the image
                        // (not a fixed px) so it reads the same on a tiny vs. a 4K frame.
                        <motion.rect
                          key={i}
                          x={b.x} y={b.y} width={b.w} height={b.h}
                          fill="none" stroke="#EF4444" strokeWidth={Math.max(2, heroNatural!.w / 280)}
                          vectorEffect="non-scaling-stroke"
                          initial={{ opacity: 0, y: b.y + heroNatural!.h / 35 }}
                          animate={{ opacity: 1, y: b.y }}
                          transition={{ duration: 0.55, delay: 0.5 + i * 0.15, ease: [0.16, 1, 0.3, 1] }}
                        />
                      ))}
                    </svg>
                  </motion.div>
                ) : (
                  <motion.div key="placeholder" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }} style={{ position: "absolute", inset: 0 }}>
                    <div style={{ position: "absolute", inset: 0, background: "radial-gradient(circle at 70% 20%,rgba(79,70,229,.18),transparent 55%)" }} />
                    <span style={{ position: "absolute", top: 10, left: 12, fontFamily: FONT.mono, fontSize: 9.5, letterSpacing: ".06em", color: "rgba(255,255,255,.42)" }}>FRAME 1920×1200 · 60fps</span>
                    <div style={{ position: "absolute", left: "13%", top: "26%", width: "25%", height: "50%", border: "2px solid #EF4444", borderRadius: 7, boxShadow: "0 0 0 1px rgba(239,68,68,.25),inset 0 0 28px rgba(239,68,68,.12)", animation: "bfu .6s ease both", animationDelay: ".6s" }}>
                      <div style={{ position: "absolute", top: -21, left: -2, background: "#EF4444", color: "#fff", fontFamily: FONT.mono, fontSize: 9.5, fontWeight: 600, padding: "2px 6px", borderRadius: 5, whiteSpace: "nowrap" }}>NO HELMET 0.91</div>
                    </div>
                    <div style={{ position: "absolute", left: "58%", top: "58%", width: "22%", height: "13%", border: "2px solid #6366F1", borderRadius: 6, boxShadow: "0 0 0 1px rgba(99,102,241,.3)", animation: "bfu .6s ease both", animationDelay: ".78s" }}>
                      <div style={{ position: "absolute", bottom: -20, left: -2, background: "#4F46E5", color: "#fff", fontFamily: FONT.mono, fontSize: 9.5, fontWeight: 600, padding: "2px 6px", borderRadius: 5, whiteSpace: "nowrap" }}>DL 3C AB 1234</div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 6px 4px" }}>
              <span style={{ fontFamily: FONT.mono, fontSize: 11.5, color: "#52525B" }}><span style={{ color: "#EF4444", fontWeight: 600 }}>{hero?.violationCount ?? 2}</span> violations · <span style={{ color: "#18181B", fontWeight: 600 }}>{hero?.objectCount ?? 4}</span> objects</span>
              <span style={{ fontFamily: FONT.body, fontSize: 11.5, fontWeight: 600, color: "#10B981", background: "#ECFDF5", padding: "3px 9px", borderRadius: 6 }}>{hero?.status ?? "Auto-confirmed"}</span>
            </div>
          </div>
          {!hero && (
            <div style={{ position: "absolute", bottom: -14, left: -22, background: "#18181B", color: "#fff", borderRadius: 11, padding: "10px 13px", boxShadow: "0 16px 34px -14px rgba(24,24,27,.55)", animation: "float 4.5s ease-in-out infinite" }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 9, letterSpacing: ".08em", color: "#A1A1AA" }}>RIDER LOCATED</div>
              <div style={{ fontFamily: FONT.mono, fontSize: 13, fontWeight: 600, marginTop: 2 }}>28.6139°N · 77.2090°E</div>
            </div>
          )}
        </div>
      </div>

      {/* tech / stack marquee */}
      <div id="stack" style={{ marginTop: 30, borderTop: "1px solid #ECECEC", padding: "32px 0 6px" }}>
        <div style={{ textAlign: "center", fontSize: 13, color: "#9CA3AF", marginBottom: 20 }}>Built on a transparent, benchmarked detection stack</div>
        <div style={{ position: "relative", overflow: "hidden", WebkitMaskImage: "linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent)", maskImage: "linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent)" }}>
          <div style={{ display: "flex", gap: 14, width: "max-content", animation: "marquee 36s linear infinite" }}>
            {[...CHIPS, ...CHIPS].map((c, i) => (
              <span key={i} style={{ flex: "none", fontFamily: FONT.mono, fontSize: 13, fontWeight: 500, color: "#52525B", background: "#fff", border: "1px solid #ECECEC", borderRadius: 10, padding: "9px 16px", whiteSpace: "nowrap" }}>{c}</span>
            ))}
          </div>
        </div>
      </div>

      {/* features */}
      <div id="features" style={{ maxWidth: 1180, margin: "0 auto", padding: "64px 32px 24px" }}>
        <div style={{ textAlign: "center", maxWidth: 640, margin: "0 auto 38px" }}>
          <Reveal style={{ fontFamily: FONT.mono, fontSize: 12, letterSpacing: ".1em", color: "#4F46E5", fontWeight: 500 }}>DETECTION SUITE</Reveal>
          <Reveal delay={0.06}><h2 style={{ fontFamily: FONT.sans, fontSize: 38, fontWeight: 600, letterSpacing: "-0.03em", margin: "10px 0 0" }}>Seven violation classes, read in <span style={serif}>one</span> pass.</h2></Reveal>
          <Reveal delay={0.12}><p style={{ fontSize: 16, color: "#6B7280", lineHeight: 1.55, margin: "14px 0 0" }}>One forward pass over any frame returns every flagged violation — with confidence, plate and location attached. Benchmarked on the public IDD and LISA datasets.</p></Reveal>
        </div>
        <div data-tour="landing-features" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
          {FEATURES.map((f, i) => (
            <Reveal key={f.code} delay={i * 0.05}>
              <div className={i % 2 ? "gl-card-rt" : "gl-card-lt"} style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 22, boxShadow: "0 1px 2px rgba(24,24,27,.04)", transition: "transform .35s cubic-bezier(.16,1,.3,1),box-shadow .35s", height: "100%" }}>
                <div style={{ width: 38, height: 38, borderRadius: 10, background: f.bg, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ width: 13, height: 13, borderRadius: "50%", background: f.dot }} />
                </div>
                <div style={{ fontFamily: FONT.mono, fontSize: 10.5, letterSpacing: ".05em", color: "#9CA3AF", marginTop: 16 }}>{f.code}</div>
                <div style={{ fontFamily: FONT.sans, fontSize: 18, fontWeight: 600, marginTop: 3 }}>{f.title}</div>
                <p style={{ fontSize: 13.5, color: "#6B7280", lineHeight: 1.5, margin: "8px 0 0" }}>{f.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>

        {/* wide capability card */}
        <Reveal delay={0.08}>
          <div style={{ background: "#18181B", borderRadius: 20, padding: 32, marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 34, alignItems: "center", position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", top: "-40%", left: "-5%", width: 340, height: 340, background: "radial-gradient(circle,rgba(79,70,229,.45),transparent 65%)" }} />
            <div style={{ position: "relative" }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 11, letterSpacing: ".1em", color: "#818CF8" }}>BEYOND DETECTION</div>
              <h3 style={{ fontFamily: FONT.sans, fontSize: 26, fontWeight: 600, color: "#fff", letterSpacing: "-0.02em", lineHeight: 1.15, margin: "10px 0 0" }}>Reads plates. Locates subjects. Builds the <span style={serif}>case file</span>.</h3>
              <p style={{ fontSize: 14.5, color: "#A1A1AA", lineHeight: 1.6, margin: "12px 0 0", maxWidth: 380 }}>Every flagged subject is OCR&apos;d, projected to map coordinates, and packaged into a signed, challan-ready evidence bundle.</p>
            </div>
            <div style={{ position: "relative", background: "rgba(255,255,255,.04)", border: "1px solid rgba(255,255,255,.08)", borderRadius: 14, padding: "6px 14px" }}>
              {[
                { c: "#EF4444", t: "No-helmet riding", p: "DL 3C AB 1234" },
                { c: "#F59E0B", t: "Triple riding", p: "UP 16 BT 4521" },
                { c: "#EF4444", t: "Red-light jump", p: "HR 26 DK 8830" },
                { c: "#0EA5E9", t: "Subject located", p: "28.61°N · 77.20°E" },
              ].map((r, i, a) => (
                <div key={r.t} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 2px", borderBottom: i < a.length - 1 ? "1px solid rgba(255,255,255,.07)" : "none" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <span style={{ width: 7, height: 7, borderRadius: "50%", background: r.c }} />
                    <span style={{ fontSize: 12.5, color: "#E4E4E7" }}>{r.t}</span>
                  </div>
                  <span style={{ fontFamily: FONT.mono, fontSize: 11, color: "#A1A1AA" }}>{r.p}</span>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      {/* workflow */}
      <div id="workflow" style={{ maxWidth: 1180, margin: "0 auto", padding: "60px 32px 24px" }}>
        <div style={{ textAlign: "center", maxWidth: 620, margin: "0 auto 38px" }}>
          <Reveal style={{ fontFamily: FONT.mono, fontSize: 12, letterSpacing: ".1em", color: "#4F46E5", fontWeight: 500 }}>HOW IT WORKS</Reveal>
          <Reveal delay={0.06}><h2 style={{ fontFamily: FONT.sans, fontSize: 38, fontWeight: 600, letterSpacing: "-0.03em", margin: "10px 0 0" }}>From frame to evidence in <span style={serif}>one</span> pipeline.</h2></Reveal>
          <Reveal delay={0.12}><p style={{ fontSize: 16, color: "#6B7280", lineHeight: 1.55, margin: "14px 0 0" }}>Six stages. No manual triage. Every detection logged and reproducible from the source frame.</p></Reveal>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
          {STEPS.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.05}>
              <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: 22, boxShadow: "0 1px 2px rgba(24,24,27,.04)", height: "100%" }}>
                <div style={{ fontFamily: FONT.mono, fontSize: 11, fontWeight: 600, letterSpacing: ".06em", color: "#4F46E5" }}>STEP {s.n}</div>
                <div style={{ fontFamily: FONT.sans, fontSize: 18, fontWeight: 600, marginTop: 10 }}>{s.t}</div>
                <p style={{ fontSize: 13.5, color: "#6B7280", lineHeight: 1.5, margin: "8px 0 0" }}>{s.d}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>

      {/* honest stats strip */}
      <div style={{ maxWidth: 1180, margin: "48px auto 0", padding: "0 32px" }}>
        <Reveal delay={0.05}>
          <div data-tour="landing-stats" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 0, borderTop: "1px solid #ECECEC", borderBottom: "1px solid #ECECEC", padding: "32px 0" }}>
            {[
              { node: <CountUp end={mapPct ?? 58.6} decimals={1} suffix="%" style={{ fontFamily: FONT.mono, fontSize: 38, fontWeight: 600, letterSpacing: "-0.02em" }} />, label: "Detection mAP@0.5 (IDD)" },
              { node: <CountUp end={78} suffix="%" style={{ fontFamily: FONT.mono, fontSize: 38, fontWeight: 600, letterSpacing: "-0.02em" }} />, label: "ANPR exact-match" },
              { node: <CountUp end={99.7} decimals={1} suffix="%" style={{ fontFamily: FONT.mono, fontSize: 38, fontWeight: 600, letterSpacing: "-0.02em" }} />, label: "Signal-state accuracy (LISA)" },
              { node: <span style={{ fontFamily: FONT.mono, fontSize: 38, fontWeight: 600, letterSpacing: "-0.02em" }}>7</span>, label: "Violation classes + ANPR" },
            ].map((s, i) => (
              <div key={i} style={{ padding: "0 24px", borderRight: i < 3 ? "1px solid #ECECEC" : "none" }}>
                <div>{s.node}</div>
                <div style={{ fontSize: 13.5, color: "#6B7280", marginTop: 4 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </Reveal>
      </div>

      {/* faq */}
      <div id="faq" style={{ maxWidth: 760, margin: "0 auto", padding: "64px 32px 24px" }}>
        <div style={{ textAlign: "center", marginBottom: 30 }}>
          <Reveal style={{ fontFamily: FONT.mono, fontSize: 12, letterSpacing: ".1em", color: "#4F46E5", fontWeight: 500 }}>FAQ</Reveal>
          <Reveal delay={0.06}><h2 style={{ fontFamily: FONT.sans, fontSize: 38, fontWeight: 600, letterSpacing: "-0.03em", margin: "10px 0 0" }}>Questions, <span style={serif}>answered</span>.</h2></Reveal>
        </div>
        <Reveal delay={0.1}>
          <div style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 16, padding: "4px 22px", boxShadow: "0 1px 2px rgba(24,24,27,.04)" }}>
            {FAQ.map((f, i) => {
              const open = openFaq === i;
              return (
                <div key={i} style={{ borderBottom: i < FAQ.length - 1 ? "1px solid #F4F4F5" : "none" }}>
                  <div onClick={() => setOpenFaq(open ? -1 : i)} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "18px 2px", cursor: "pointer" }}>
                    <span style={{ fontFamily: FONT.sans, fontSize: 16.5, fontWeight: 600, letterSpacing: "-0.01em" }}>{f.q}</span>
                    <span style={{ flex: "none", width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: FONT.mono, fontSize: 21, color: "#4F46E5", lineHeight: 1, transform: open ? "rotate(45deg)" : "rotate(0deg)", transition: "transform .3s ease" }}>+</span>
                  </div>
                  <div style={{ maxHeight: open ? 260 : 0, opacity: open ? 1 : 0, overflow: "hidden", transition: "max-height .35s ease,opacity .3s ease" }}>
                    <p style={{ fontSize: 14.5, color: "#6B7280", lineHeight: 1.6, margin: 0, padding: "0 30px 18px 2px" }}>{f.a}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </Reveal>
      </div>

      {/* demo */}
      <div id="demo" style={{ maxWidth: 1180, margin: "0 auto", padding: "72px 32px 44px" }}>
        <Reveal delay={0.05}>
          <div data-tour="landing-demo" style={{ background: "#fff", border: "1px solid #ECECEC", borderRadius: 24, overflow: "hidden", display: "grid", gridTemplateColumns: "1fr 1fr", boxShadow: "0 24px 60px -34px rgba(24,24,27,.2)" }}>
            <div style={{ padding: 44, borderRight: "1px solid #ECECEC" }}>
              <div style={{ fontFamily: FONT.mono, fontSize: 12, letterSpacing: ".1em", color: "#4F46E5", fontWeight: 500 }}>READY WHEN YOU ARE</div>
              <h2 style={{ fontFamily: FONT.sans, fontSize: 34, fontWeight: 600, letterSpacing: "-0.03em", margin: "12px 0 0", lineHeight: 1.1 }}>See Padlock <span style={serif}>live</span>.</h2>
              <p style={{ fontSize: 15.5, color: "#6B7280", lineHeight: 1.6, margin: "16px 0 0", maxWidth: 380 }}>Tell us about your enforcement setup and we&apos;ll set up a tailored walkthrough on your own camera feeds.</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 13, marginTop: 26 }}>
                {["30-minute tailored walkthrough", "Run on a sample of your frames", "Deployment scoped to your network"].map((t) => (
                  <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 10, fontSize: 14, color: "#52525B", fontWeight: 500 }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" strokeWidth="2.6"><path d="M20 6 9 17l-5-5" /></svg>{t}
                  </span>
                ))}
              </div>
            </div>
            <div style={{ padding: 44 }}>
              {!sent ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    <Field label="Full name" placeholder="Priya Nair" />
                    <Field label="Work email" placeholder="priya@citypolice.gov" />
                  </div>
                  <Field label="Organisation" placeholder="City Traffic Police" />
                  <div>
                    <Label>Camera network size</Label>
                    <select className="gl-field" style={fieldStyle}>
                      <option>Select…</option>
                      <option>Under 50 cameras</option>
                      <option>50–200 cameras</option>
                      <option>200–1,000 cameras</option>
                      <option>1,000+ cameras</option>
                    </select>
                  </div>
                  <div>
                    <Label>What are you enforcing? <span style={{ color: "#9CA3AF", fontWeight: 400 }}>(optional)</span></Label>
                    <textarea rows={2} placeholder="Helmet compliance on the NH-48 corridor…" className="gl-field" style={{ ...fieldStyle, resize: "none" }} />
                  </div>
                  <button onClick={() => setSent(true)} className="gl-press gl-btn-primary" style={{ width: "100%", fontFamily: FONT.body, fontSize: 14.5, fontWeight: 600, color: "#fff", border: "none", padding: 12, borderRadius: 11, cursor: "pointer" }}>Request demo</button>
                  <p style={{ fontSize: 12, color: "#9CA3AF", textAlign: "center", margin: 0 }}>We respond within one business day.</p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", justifyContent: "center", height: "100%", gap: 14, animation: "bfu .6s cubic-bezier(.16,1,.3,1) both" }}>
                  <div style={{ width: 46, height: 46, borderRadius: 13, background: "#ECFDF5", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="2.6"><path d="M20 6 9 17l-5-5" /></svg>
                  </div>
                  <h3 style={{ fontFamily: FONT.sans, fontSize: 24, fontWeight: 600, letterSpacing: "-0.02em", margin: 0 }}>Request received.</h3>
                  <p style={{ fontSize: 14.5, color: "#6B7280", lineHeight: 1.6, margin: 0 }}>Thanks — our team will reach out within one business day to schedule your walkthrough on your own feeds.</p>
                  <button onClick={() => router.push("/detect")} className="gl-press gl-btn-ghost" style={{ marginTop: 4, display: "inline-flex", alignItems: "center", gap: 8, fontFamily: FONT.body, fontSize: 14, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "10px 16px", borderRadius: 11, cursor: "pointer" }}>
                    Explore the console
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#18181B" strokeWidth="2.4"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
                  </button>
                </div>
              )}
            </div>
          </div>
        </Reveal>
      </div>

      {/* footer */}
      <div style={{ borderTop: "1px solid #ECECEC", marginTop: 40 }}>
        <div style={{ maxWidth: 1180, margin: "0 auto", padding: "48px 32px 28px", display: "grid", gridTemplateColumns: "1.7fr 1fr 1fr 1.5fr", gap: 32 }}>
          <div>
            <Brand size={60} />
            <p style={{ fontSize: 13.5, color: "#6B7280", lineHeight: 1.6, margin: "14px 0 0", maxWidth: 280 }}>Computer-vision traffic enforcement. Read violations, plates and subject locations from any frame — court-ready in one pass.</p>
            <div style={{ fontFamily: FONT.mono, fontSize: 11, color: "#9CA3AF", marginTop: 16, display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#10B981" }} />All systems operational
            </div>
          </div>
          <FooterCol title="Product" items={[["Features", () => jump("features")], ["Workflow", () => jump("workflow")], ["Stack", () => jump("stack")], ["FAQ", () => jump("faq")]]} />
          <FooterCol title="Console" items={[["Detect", () => router.push("/detect")], ["Dashboard", () => router.push("/dashboard")], ["Violations", () => router.push("/violations")], ["Reports", () => router.push("/reports")]]} />
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: "#18181B", marginBottom: 10 }}>Get a demo</div>
            <p style={{ fontSize: 13, color: "#6B7280", lineHeight: 1.55, margin: "0 0 14px" }}>See Padlock live on sample traffic feeds.</p>
            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={() => jump("demo")} className="gl-press" style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#fff", background: "#18181B", border: "none", padding: "9px 14px", borderRadius: 10, cursor: "pointer" }}>Request demo</button>
              <button onClick={() => router.push("/dashboard")} className="gl-press gl-btn-ghost" style={{ fontFamily: FONT.body, fontSize: 13, fontWeight: 600, color: "#18181B", background: "#fff", border: "1px solid #ECECEC", padding: "9px 14px", borderRadius: 10, cursor: "pointer" }}>Sign in</button>
            </div>
          </div>
        </div>
        <div style={{ borderTop: "1px solid #ECECEC" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 13, color: "#9CA3AF" }}>
            <span>© 2026 Team Padlock</span>
            <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
              <span className="gl-link">Privacy</span>
              <span className="gl-link">Terms</span>
              <span className="gl-link">Security</span>
              <span style={{ fontFamily: FONT.mono }}>v1.0.0</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const fieldStyle: React.CSSProperties = {
  width: "100%",
  fontFamily: FONT.body,
  fontSize: 14,
  color: "#18181B",
  background: "#FAFAFA",
  border: "1px solid #ECECEC",
  borderRadius: 10,
  padding: "10px 12px",
  outline: "none",
};

function Label({ children }: { children: React.ReactNode }) {
  return <label style={{ fontSize: 12, fontWeight: 600, color: "#52525B", marginBottom: 6, display: "block" }}>{children}</label>;
}

function Field({ label, placeholder }: { label: string; placeholder: string }) {
  return (
    <div>
      <Label>{label}</Label>
      <input placeholder={placeholder} className="gl-field" style={fieldStyle} />
    </div>
  );
}

function FooterCol({ title, items }: { title: string; items: [string, () => void][] }) {
  return (
    <div>
      <div style={{ fontSize: 12.5, fontWeight: 600, color: "#18181B", marginBottom: 14 }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13.5, color: "#6B7280" }}>
        {items.map(([label, fn]) => (
          <span key={label} className="gl-link" onClick={fn}>{label}</span>
        ))}
      </div>
    </div>
  );
}
