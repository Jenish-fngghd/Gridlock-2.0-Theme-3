/** Step config for the guided product tour — modeled on Linear/Vercel-style
 *  first-run spotlight coachmarks: one highlighted element + a small card, advanced
 *  by the visitor, never auto-advancing. */

export type Placement = "top" | "bottom" | "left" | "right" | "center";

export interface TourStep {
  selector: string;
  title: string;
  body: string;
  placement?: Placement;
}

export type PageKey = "landing" | "detect" | "dashboard" | "violations" | "reports";
export type TourKey = "shell" | PageKey;

export const SHELL_KEY = "gridlock_tour_shell_v1";
export function pageStorageKey(key: PageKey) {
  return `gridlock_tour_${key}_v1`;
}

/** Runs once, the first time a first-time visitor reaches any console page. */
export const SHELL_STEPS: TourStep[] = [
  {
    selector: '[data-tour="sidebar-brand"]',
    title: "Welcome to Team Padlock",
    body: "This is your enforcement console — every screen here reads live detection data straight out of Supabase. This quick tour takes about a minute.",
    placement: "right",
  },
  {
    selector: '[data-tour="sidebar-nav"]',
    title: "Four workspaces",
    body: "Detect runs the live pipeline on a single frame. Dashboard, Violations and Reports all read that same data, updated in real time as new frames come in.",
    placement: "right",
  },
  {
    selector: '[data-tour="topbar-search"]',
    title: "Jump to any record",
    body: "Search a plate or violation ID from anywhere in the console — press ⌘K to open it instantly.",
    placement: "bottom",
  },
  {
    selector: '[data-tour="help-button"]',
    title: "Replay this tour anytime",
    body: "Click here whenever you want a walkthrough of the page you're on, or the full tour from the start.",
    placement: "left",
  },
];

export const TOURS: Record<PageKey, TourStep[]> = {
  landing: [
    {
      selector: '[data-tour="landing-nav"]',
      title: "Welcome to Team Padlock",
      body: "An AI that reads a traffic frame, flags violations, reads number plates, and locates every rider, driver and vehicle — in one pass. Let's take a 60-second look around.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="landing-cta"]',
      title: "Launch the console",
      body: "This drops you straight into Detect, where you can upload a real frame and watch the full pipeline run.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="landing-features"]',
      title: "Seven violation classes",
      body: "No-helmet, triple-riding, no-seatbelt, wrong-side, red-light, stop-line and illegal-parking — all flagged from a single forward pass.",
      placement: "top",
    },
    {
      selector: '[data-tour="landing-stats"]',
      title: "Honest, benchmarked numbers",
      body: "Every metric on this page is a real benchmark result, not a marketing figure — reproducible from the same evaluation harness used internally.",
      placement: "top",
    },
    {
      selector: '[data-tour="landing-demo"]',
      title: "Ready to go deeper?",
      body: "Request a tailored walkthrough on your own camera feeds, or just sign in and explore the live console yourself.",
      placement: "top",
    },
  ],

  detect: [
    {
      selector: '[data-tour="detect-dropzone"]',
      title: "Drop in a frame",
      body: "Drag a traffic image here, or click to browse. Team Padlock runs the full 7-stage pipeline on it — detection, plate OCR, and every violation classifier — in one pass.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="detect-browse"]',
      title: "Or pick a file directly",
      body: "Same result either way — once a frame is in, you'll see the pipeline stages run live against the real backend.",
      placement: "top",
    },
  ],

  dashboard: [
    {
      selector: '[data-tour="dash-range"]',
      title: "Pick your window",
      body: "Switch between Today, 7 days and 30 days — every card and chart below recalculates from the same live violation data.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="dash-kpis"]',
      title: "Network at a glance",
      body: "Total violations, frames processed, plates read and cameras online — with day-over-day deltas so you can spot a spike immediately.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="dash-donut"]',
      title: "What's being flagged",
      body: "A live breakdown of violations by type for the selected window.",
      placement: "right",
    },
    {
      selector: '[data-tour="dash-area"]',
      title: "Trend over time",
      body: "Hover anywhere on the chart for an exact count at that point — the crosshair follows your cursor.",
      placement: "left",
    },
    {
      selector: '[data-tour="dash-pipeline"]',
      title: "Where the time goes",
      body: "Average latency per pipeline stage — useful for spotting which stage to optimize first.",
      placement: "right",
    },
    {
      selector: '[data-tour="dash-feed"]',
      title: "Live incidents",
      body: "New detections stream in here in real time. Click any row to open its full evidence record.",
      placement: "left",
    },
  ],

  violations: [
    {
      selector: '[data-tour="viol-toolbar"]',
      title: "Search and filter",
      body: "Search by plate, violation ID or camera, or filter straight to Critical, High or Pending-review records.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="viol-export"]',
      title: "Export for offline review",
      body: "Whatever's currently filtered exports to a CSV evidence sheet — plates, confidence, severity and status included.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="viol-table"]',
      title: "Open the full evidence screen",
      body: "Click any row to see the annotated frame, plate OCR, per-detection confidence and the full audit trail for that violation.",
      placement: "top",
    },
  ],

  reports: [
    {
      selector: '[data-tour="rep-controls"]',
      title: "Scope the report",
      body: "Pick a date range and narrow by zone or violation type — every tile and chart below updates to match.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="rep-timeseries"]',
      title: "Multi-series trend",
      body: "Toggle violation types on or off to compare them directly over time.",
      placement: "bottom",
    },
    {
      selector: '[data-tour="rep-bento"]',
      title: "Breakdowns that matter",
      body: "By type, by zone, by vehicle class, and an hour-of-day heatmap — all reveal as you scroll.",
      placement: "top",
    },
    {
      selector: '[data-tour="rep-hotspots"]',
      title: "Where it's happening",
      body: "Ranked zones, synced to the map — hover a row or a pin to highlight the other.",
      placement: "top",
    },
    {
      selector: '[data-tour="rep-table"]',
      title: "Drill into any incident",
      body: "Sort by recency or confidence, then jump straight to the full evidence record.",
      placement: "top",
    },
  ],
};

export function keyForPath(path: string): PageKey | null {
  if (path === "/") return "landing";
  if (path.startsWith("/detect")) return "detect";
  if (path.startsWith("/dashboard")) return "dashboard";
  if (path.startsWith("/violations")) return "violations";
  if (path.startsWith("/reports")) return "reports";
  return null;
}
