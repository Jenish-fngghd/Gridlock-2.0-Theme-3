import Image from "next/image";
import { FONT } from "@/lib/ui";

/** The Team Padlock wordmark, used in the sidebar, nav and footer. */
export default function Brand({ size = 44 }: { size?: number }) {
  const fontSize = Math.round(size * 0.46);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: Math.round(size * 0.24) }}>
      <Image
        src="/logo.png"
        alt="Team Padlock"
        width={size}
        height={size}
        style={{ width: size, height: size, objectFit: "contain", flex: "none" }}
        priority
      />
      <span style={{ fontFamily: FONT.sans, fontWeight: 500, fontSize, letterSpacing: "-0.02em", color: "#52525B" }}>
        Team
      </span>
      <span style={{ fontFamily: FONT.sans, fontWeight: 600, fontSize, letterSpacing: "-0.02em" }}>
        Padlock
      </span>
    </div>
  );
}
