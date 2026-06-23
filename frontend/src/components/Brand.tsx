import { FONT } from "@/lib/ui";

/** The Gridlock wordmark + 2.0 chip, used in the sidebar, nav and footer. */
export default function Brand({ size = 28 }: { size?: number }) {
  const inner = Math.round(size * 0.39);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div
        style={{
          width: size,
          height: size,
          borderRadius: 8,
          background: "linear-gradient(135deg,#4F46E5,#6366F1)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 2px 8px -2px rgba(79,70,229,.6)",
        }}
      >
        <div style={{ width: inner, height: inner, border: "2px solid #fff", borderRadius: 3 }} />
      </div>
      <span style={{ fontFamily: FONT.sans, fontWeight: 600, fontSize: 19, letterSpacing: "-0.02em" }}>
        Gridlock
      </span>
      <span
        style={{
          fontFamily: FONT.mono,
          fontSize: 11,
          fontWeight: 500,
          color: "#4F46E5",
          background: "#EEF0FF",
          border: "1px solid #E0E2FF",
          padding: "2px 7px",
          borderRadius: 6,
        }}
      >
        2.0
      </span>
    </div>
  );
}
