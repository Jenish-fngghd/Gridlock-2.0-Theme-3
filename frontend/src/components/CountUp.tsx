"use client";

import { useEffect, useRef, useState } from "react";

/** Eased count-up that runs once when scrolled into view — mirrors the
 *  design canvas's data-countup behaviour. */
export default function CountUp({
  end,
  decimals = 0,
  suffix = "",
  duration = 1100,
  style,
  className,
}: {
  end: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
  style?: React.CSSProperties;
  className?: string;
}) {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const ran = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting || ran.current) return;
        ran.current = true;
        const start = performance.now();
        const tick = (t: number) => {
          let p = Math.min(1, (t - start) / duration);
          p = 1 - Math.pow(1 - p, 3);
          setVal(end * p);
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.3 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [end, duration]);

  const text =
    (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
  return (
    <span ref={ref} style={style} className={className}>
      {text}
    </span>
  );
}
