"use client";

import { usePathname } from "next/navigation";
import { FONT } from "@/lib/ui";

const TITLES: Record<string, string> = {
  "/detect": "Detect",
  "/dashboard": "Dashboard",
  "/violations": "Violations",
  "/reports": "Reports",
};

export default function Topbar() {
  const path = usePathname();
  const title = TITLES[path] ?? Object.entries(TITLES).find(([k]) => path.startsWith(k))?.[1] ?? "";

  return (
    <div
      style={{
        height: 64,
        borderBottom: "1px solid #ECECEC",
        background: "rgba(250,250,250,.82)",
        backdropFilter: "blur(10px)",
        position: "sticky",
        top: 0,
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 36px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 9, fontFamily: FONT.mono, fontSize: 12, color: "#9CA3AF" }}>
        Console <span style={{ color: "#D4D4D8" }}>/</span>{" "}
        <span style={{ color: "#18181B", fontWeight: 500 }}>{title}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: "#fff",
            border: "1px solid #ECECEC",
            borderRadius: 9,
            padding: "7px 12px",
            width: 230,
            color: "#9CA3AF",
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2">
            <circle cx="11" cy="11" r="7" />
            <path d="m21 21-4-4" />
          </svg>
          <span style={{ fontSize: 13 }}>Search plate or ID…</span>
          <span
            style={{
              marginLeft: "auto",
              fontFamily: FONT.mono,
              fontSize: 10,
              border: "1px solid #ECECEC",
              borderRadius: 4,
              padding: "1px 5px",
            }}
          >
            ⌘K
          </span>
        </div>
        <div
          style={{
            position: "relative",
            width: 34,
            height: 34,
            borderRadius: 9,
            border: "1px solid #ECECEC",
            background: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#52525B" strokeWidth="1.9">
            <path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.7 21a2 2 0 0 1-3.4 0" />
          </svg>
          <span
            style={{
              position: "absolute",
              top: 6,
              right: 7,
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "#EF4444",
            }}
          />
        </div>
      </div>
    </div>
  );
}
