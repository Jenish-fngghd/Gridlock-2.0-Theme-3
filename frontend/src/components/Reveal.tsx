"use client";

import { motion } from "motion/react";
import type { ReactNode, CSSProperties } from "react";

/** Blur-fade-up on scroll into view — matches the canvas [data-reveal] effect. */
export default function Reveal({
  children,
  delay = 0,
  className,
  style,
  as = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  style?: CSSProperties;
  as?: "div" | "section";
}) {
  const M = as === "section" ? motion.section : motion.div;
  return (
    <M
      className={className}
      style={style}
      initial={{ opacity: 0, y: 18, filter: "blur(10px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-7%" }}
      transition={{ duration: 0.75, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </M>
  );
}
