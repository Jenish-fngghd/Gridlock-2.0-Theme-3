import Image from "next/image";
import { FONT } from "@/lib/ui";

/** The Team Padlock wordmark, used in the sidebar, nav and footer.
 *  `size` only controls the icon — the wordmark stays a fixed, legible size
 *  so a larger icon doesn't blow out tight spaces like the sidebar. */
export default function Brand({ size = 72 }: { size?: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <Image
        src="/logo.png"
        alt="Team Padlock"
        width={size}
        height={size}
        style={{ width: size, height: size, objectFit: "contain", flex: "none" }}
        priority
      />
      <span style={{ fontFamily: FONT.sans, fontWeight: 500, fontSize: 20, letterSpacing: "-0.02em", color: "#52525B" }}>
        Team
      </span>
      <span style={{ fontFamily: FONT.sans, fontWeight: 600, fontSize: 20, letterSpacing: "-0.02em" }}>
        Padlock
      </span>
    </div>
  );
}
