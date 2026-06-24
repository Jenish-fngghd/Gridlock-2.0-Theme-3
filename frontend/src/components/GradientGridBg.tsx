/** Light grid lines + a soft purple radial glow, ported 1:1 from the
 *  21st.dev "gradient-blur-bg" reference (inline-style background-image
 *  recipe — no Tailwind classes needed, so it drops straight into this
 *  codebase's plain-style convention). */
export default function GradientGridBg({ side = "right" }: { side?: "left" | "right" }) {
  const x = side === "right" ? "100%" : "0%";
  return (
    <div
      aria-hidden
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
        backgroundImage: `
          linear-gradient(to right, #f0f0f0 1px, transparent 1px),
          linear-gradient(to bottom, #f0f0f0 1px, transparent 1px),
          radial-gradient(circle 800px at ${x} 200px, #d5c5ff, transparent)
        `,
        backgroundSize: "96px 64px, 96px 64px, 100% 100%",
      }}
    />
  );
}
