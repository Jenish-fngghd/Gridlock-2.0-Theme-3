import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";

export default function ConsoleLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar />
      <div style={{ flex: 1, minWidth: 0 }}>
        <Topbar />
        {children}
      </div>
    </div>
  );
}
