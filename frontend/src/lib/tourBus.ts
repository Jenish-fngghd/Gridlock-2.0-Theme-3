import type { PageKey } from "./tours";

export const TOUR_EVENT = "gridlock-tour:start";

export interface TourRequestDetail {
  key?: PageKey;
  /** Force the shared sidebar/topbar intro steps even if already seen. */
  forceShell?: boolean;
}

/** Fire-and-forget request to (re)start a tour — decouples trigger buttons
 *  (help button, nav links) from the TourProvider without needing context. */
export function requestTour(detail: TourRequestDetail = {}) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<TourRequestDetail>(TOUR_EVENT, { detail }));
}
