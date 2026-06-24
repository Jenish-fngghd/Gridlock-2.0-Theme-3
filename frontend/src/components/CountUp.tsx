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
    let cancelled = false;
    const animate = () => {
      const start = performance.now();
      const tick = (t: number) => {
        if (cancelled) return;
        let p = Math.min(1, (t - start) / duration);
        p = 1 - Math.pow(1 - p, 3);
        setVal(end * p);
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };
    // Already played once (e.g. real data replacing an initial 0 while loading) -- animate
    // straight to the new value instead of waiting on the observer, which won't fire again
    // for an element already in view.
    if (ran.current) {
      animate();
      return () => { cancelled = true; };
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) return;
        ran.current = true;
        animate();
      },
      { threshold: 0.3 }
    );
    io.observe(el);
    return () => { cancelled = true; io.disconnect(); };
  }, [end, duration]);

  const text =
    (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
  return (
    <span ref={ref} style={style} className={className}>
      {text}
    </span>
  );
}
