"use client";

import "leaflet/dist/leaflet.css";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";

// Real coordinates for the named zones (Mumbai) -- replaces the earlier fabricated
// percentage positions on a blank grid with an actual OpenStreetMap base layer.
export const ZONE_LATLNG: Record<string, [number, number]> = {
  "Andheri East": [19.1136, 72.8697],
  "Bandra-Kurla Complex": [19.0728, 72.8826],
  "Dadar TT Circle": [19.0186, 72.8478],
  Powai: [19.1176, 72.906],
  "Worli Sea Face": [19.0176, 72.8169],
  "Chembur Naka": [19.0522, 72.9006],
};
const MUMBAI_CENTER: [number, number] = [19.0667, 72.8717];

export default function HotspotMap({
  hotspots,
  hoverZone,
  setHoverZone,
}: {
  hotspots: { zone: string; total: number }[];
  hoverZone: string | null;
  setHoverZone: (z: string | null) => void;
}) {
  const max = Math.max(1, ...hotspots.map((h) => h.total));
  return (
    <MapContainer
      center={MUMBAI_CENTER}
      zoom={11}
      scrollWheelZoom={false}
      style={{ height: "100%", width: "100%", minHeight: 260, background: "#F4F4F8" }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {hotspots.map((h) => {
        const pos = ZONE_LATLNG[h.zone];
        if (!pos) return null;
        const active = hoverZone === h.zone;
        const r = 7 + (h.total / max) * 11;
        return (
          <CircleMarker
            key={h.zone}
            center={pos}
            radius={active ? r + 3 : r}
            pathOptions={{
              color: "#fff",
              weight: 2,
              fillColor: "#4F46E5",
              fillOpacity: active ? 0.9 : 0.65,
            }}
            eventHandlers={{
              mouseover: () => setHoverZone(h.zone),
              mouseout: () => setHoverZone(null),
            }}
          >
            <Tooltip direction="top" offset={[0, -r]}>
              {h.zone} · {h.total}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
