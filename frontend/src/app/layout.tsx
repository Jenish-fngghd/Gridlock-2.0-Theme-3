import type { Metadata } from "next";
import { Geist, Geist_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";
import Preloader from "@/components/Preloader";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const grotesk = Space_Grotesk({
  variable: "--font-grotesk",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Gridlock 2.0 — Violation Console",
  description: "Automated traffic-violation detection, evidence & analytics",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${grotesk.variable} antialiased`}
    >
      <body className="min-h-screen bg-[#f6f7fb] text-slate-900">
        <div className="mesh" />
        <Preloader />
        <div className="flex min-h-screen">
          <Nav />
          <main className="flex-1 overflow-x-hidden px-6 py-8 md:px-10">{children}</main>
        </div>
      </body>
    </html>
  );
}
