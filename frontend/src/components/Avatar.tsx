import { FONT } from "@/lib/ui";

function initialsOf(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/** Colored initials avatar — derives initials from `name`, no image required. */
export default function Avatar({ name, size = 32 }: { name: string; size?: number }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: Math.round(size * 0.28),
        background: "linear-gradient(135deg,#0EA5E9,#4F46E5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#fff",
        fontFamily: FONT.sans,
        fontWeight: 600,
        fontSize: Math.round(size * 0.4),
        flex: "none",
      }}
    >
      {initialsOf(name)}
    </div>
  );
}
