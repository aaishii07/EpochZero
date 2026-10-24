"use client";

import dynamic from "next/dynamic";

// The workspace reads saved logs from localStorage when it starts, so it renders in the browser only.
const Workspace = dynamic(() => import("@/components/Workspace"), {
  ssr: false,
  loading: () => <div className="h-screen bg-canvas" />,
});

export default function Home() {
  return <Workspace />;
}
