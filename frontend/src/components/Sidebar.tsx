"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { FONT } from "@/lib/ui";
import Brand from "./Brand";
import Avatar from "./Avatar";

const USER_NAME = "Jenish Sorathiya";
const USER_SUBTITLE = "Gridlock 2.0 Team";

type Item = { href: string; label: string; icon: React.ReactNode };

const ICONS = {
  detect: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
      <rect x="3" y="6" width="18" height="13" rx="2.5" />
      <circle cx="12" cy="12.5" r="3.2" />
      <path d="M8 6l1.5-2.5h5L16 6" />
    </svg>
  ),
  dashboard: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.6" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.6" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.6" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.6" />
    </svg>
  ),
  violations: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  ),
  reports: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9">
      <path d="M6 3h8l4 4v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
      <path d="M13 3v5h5" />
    </svg>
  ),
};

const ITEMS: Item[] = [
  { href: "/detect", label: "Detect", icon: ICONS.detect },
  { href: "/dashboard", label: "Dashboard", icon: ICONS.dashboard },
  { href: "/violations", label: "Violations", icon: ICONS.violations },
  { href: "/reports", label: "Reports", icon: ICONS.reports },
];

export default function Sidebar() {
  const path = usePathname();
  const [openCount, setOpenCount] = useState<number | null>(null);

  useEffect(() => {
    supabase
      .from("violations")
      .select("id", { count: "exact", head: true })
      .eq("status", "pending")
      .then(({ count }) => setOpenCount(count ?? 0));
  }, []);

  return (
    <aside
      style={{
        width: 248,
        flex: "none",
        borderRight: "1px solid #ECECEC",
        background: "#fff",
        display: "flex",
        flexDirection: "column",
        position: "sticky",
        top: 0,
        height: "100vh",
        padding: "22px 14px",
      }}
    >
      <Link href="/" style={{ textDecoration: "none", color: "inherit", padding: "4px 8px 22px" }}>
        <Brand />
      </Link>

      <div
        style={{
          fontFamily: FONT.mono,
          fontSize: 10,
          letterSpacing: ".1em",
          color: "#A1A1AA",
          padding: "4px 10px 8px",
        }}
      >
        WORKSPACE
      </div>

      {ITEMS.map((it) => {
        const active = path === it.href || path.startsWith(it.href + "/");
        return (
          <Link
            key={it.href}
            href={it.href}
            className="gl-nav"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 11,
              padding: "9px 11px",
              borderRadius: 10,
              cursor: "pointer",
              fontSize: 14,
              fontWeight: 500,
              textDecoration: "none",
              color: active ? "#4F46E5" : "#52525B",
              background: active ? "#EEF0FF" : "transparent",
              marginBottom: 2,
              transition: "background .15s",
            }}
          >
            {it.icon}
            {it.label}
            {it.href === "/violations" && openCount != null && openCount > 0 && (
              <span
                style={{
                  marginLeft: "auto",
                  fontFamily: FONT.mono,
                  fontSize: 10.5,
                  fontWeight: 600,
                  color: "#EF4444",
                  background: "#FEF2F2",
                  padding: "1px 7px",
                  borderRadius: 6,
                }}
              >
                {openCount}
              </span>
            )}
          </Link>
        );
      })}

      <div
        style={{
          marginTop: "auto",
          borderTop: "1px solid #ECECEC",
          paddingTop: 14,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <Avatar name={USER_NAME} size={32} />
        <div style={{ lineHeight: 1.2 }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{USER_NAME}</div>
          <div style={{ fontSize: 11, color: "#9CA3AF", fontFamily: FONT.mono }}>{USER_SUBTITLE}</div>
        </div>
      </div>
    </aside>
  );
}
