"use client";
import dynamic from "next/dynamic";

const ScubaDetector = dynamic(() => import("@/components/ScubaDetector"), {
  ssr: false,
});

export default function Home() {
  return <ScubaDetector />;
}
