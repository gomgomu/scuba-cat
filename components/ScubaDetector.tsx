"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type Landmark = { x: number; y: number; z: number };
type HandLandmarks = Landmark[];

const HAND_LANDMARKER_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";

const WRIST = 0;
const INDEX_FINGER_TIP = 8;
const MIDDLE_FINGER_TIP = 12;

function isCoveringMouth(hand: HandLandmarks): boolean {
  const wrist = hand[WRIST];
  const indexTip = hand[INDEX_FINGER_TIP];
  const middleTip = hand[MIDDLE_FINGER_TIP];
  const palmCenterY = (wrist.y + indexTip.y + middleTip.y) / 3;
  const palmCenterX = (wrist.x + indexTip.x + middleTip.x) / 3;
  const nearMouthX = palmCenterX > 0.25 && palmCenterX < 0.75;
  const nearMouthY = palmCenterY > 0.45 && palmCenterY < 0.80;
  return nearMouthX && nearMouthY;
}

function isWaving(handHistory: { x: number; y: number; time: number }[]): boolean {
  if (handHistory.length < 4) return false;
  const recent = handHistory.slice(-12);
  let directionChanges = 0;
  let lastDir = 0;
  for (let i = 1; i < recent.length; i++) {
    const dx = recent[i].x - recent[i - 1].x;
    if (Math.abs(dx) > 0.01) {
      const currentDir = dx > 0 ? 1 : -1;
      if (lastDir !== 0 && currentDir !== lastDir) directionChanges++;
      lastDir = currentDir;
    }
  }
  const minX = Math.min(...recent.map((h) => h.x));
  const maxX = Math.max(...recent.map((h) => h.x));
  return directionChanges >= 1 && maxX - minX > 0.05;
}

function getWristPos(hand: HandLandmarks) {
  return { x: hand[WRIST].x, y: hand[WRIST].y };
}

const CAT_CONFIGS = [
  {
    id: 1,
    style: {
      position: "absolute" as const,
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      width: "min(250px, 45vw)",
      aspectRatio: "1/1",
      zIndex: 10,
    },
    popClass: "scuba-pop-1",
    floatClass: "cat-float-1",
    videoClass: "",
  },
  {
    id: 2,
    style: {
      position: "absolute" as const,
      top: "30%",
      left: "20%",
      transform: "translate(-50%, -50%) rotate(-12deg)",
      width: "min(150px, 30vw)",
      aspectRatio: "1/1",
      zIndex: 5,
    },
    popClass: "scuba-pop-2",
    floatClass: "cat-float-2",
    videoClass: "",
  },
  {
    id: 3,
    style: {
      position: "absolute" as const,
      top: "30%",
      left: "80%",
      transform: "translate(-50%, -50%) rotate(12deg)",
      width: "min(150px, 30vw)",
      aspectRatio: "1/1",
      zIndex: 5,
    },
    popClass: "scuba-pop-3",
    floatClass: "cat-float-3",
    videoClass: "",
  },
  {
    id: 4,
    style: {
      position: "absolute" as const,
      top: "70%",
      left: "20%",
      transform: "translate(-50%, -50%) rotate(-8deg)",
      width: "min(150px, 30vw)",
      aspectRatio: "1/1",
      zIndex: 5,
    },
    popClass: "scuba-pop-4",
    floatClass: "cat-float-4",
    videoClass: "",
  },
  {
    id: 5,
    style: {
      position: "absolute" as const,
      top: "70%",
      left: "80%",
      transform: "translate(-50%, -50%) rotate(8deg)",
      width: "min(150px, 30vw)",
      aspectRatio: "1/1",
      zIndex: 5,
    },
    popClass: "scuba-pop-5",
    floatClass: "cat-float-5",
    videoClass: "",
  },
];

export default function ScubaDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("🔄 กำลังโหลด AI...");
  const [scubaDetected, setScubaDetected] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [debugInfo, setDebugInfo] = useState("");

  const waveHistoryRef = useRef<Map<number, { x: number; y: number; time: number }[]>>(new Map());
  const handLandmarkerRef = useRef<any>(null);
  const animFrameRef = useRef<number>(0);
  const scubaTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastVideoTimeRef = useRef(-1);

  // Centralized Scuba Cat Video Refs
  const catVideoRef = useRef<HTMLVideoElement>(null);
  const catOffscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const catCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  const detectFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !canvas || !handLandmarker || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }
    if (video.currentTime === lastVideoTimeRef.current) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }
    lastVideoTimeRef.current = video.currentTime;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    try {
      const results = handLandmarker.detectForVideo(video, Date.now());
      const hands: HandLandmarks[] = results.landmarks || [];
      const now = Date.now();
      let mouthCoveredHandIdx = -1;
      let wavingHandIdx = -1;

      hands.forEach((hand: HandLandmarks, idx: number) => {
        hand.forEach((lm: Landmark) => {
          const x = (1 - lm.x) * canvas.width;
          const y = lm.y * canvas.height;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0,255,128,0.85)";
          ctx.fill();
        });

        const connections = [
          [0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],
          [0,17],[17,18],[18,19],[19,20],
          [5,9],[9,13],[13,17],
        ];
        ctx.strokeStyle = "rgba(0,200,255,0.7)";
        ctx.lineWidth = 2.5;
        connections.forEach(([a, b]) => {
          const lmA = hand[a], lmB = hand[b];
          ctx.beginPath();
          ctx.moveTo((1 - lmA.x) * canvas.width, lmA.y * canvas.height);
          ctx.lineTo((1 - lmB.x) * canvas.width, lmB.y * canvas.height);
          ctx.stroke();
        });
      });

      hands.forEach((hand: HandLandmarks, idx: number) => {
        const wrist = getWristPos(hand);
        if (!waveHistoryRef.current.has(idx)) waveHistoryRef.current.set(idx, []);
        const history = waveHistoryRef.current.get(idx)!;
        history.push({ x: wrist.x, y: wrist.y, time: now });
        if (history.length > 15) history.shift();
        if (isCoveringMouth(hand)) mouthCoveredHandIdx = idx;
        if (isWaving(history)) wavingHandIdx = idx;
      });

      const currentIds = new Set(hands.map((_: HandLandmarks, i: number) => i));
      waveHistoryRef.current.forEach((_, key) => {
        if (!currentIds.has(key)) waveHistoryRef.current.delete(key);
      });

      const isScuba =
        hands.length >= 2 &&
        mouthCoveredHandIdx !== -1 &&
        wavingHandIdx !== -1 &&
        mouthCoveredHandIdx !== wavingHandIdx;

      setDebugInfo(
        `มือ: ${hands.length} | ปิดปาก: ${mouthCoveredHandIdx >= 0 ? "✅" : "❌"} | สบัด: ${wavingHandIdx >= 0 ? "✅" : "❌"}`
      );

      if (isScuba) {
        setScubaDetected(true);
        if (scubaTimeoutRef.current) clearTimeout(scubaTimeoutRef.current);
        scubaTimeoutRef.current = setTimeout(() => setScubaDetected(false), 2500);
      }
    } catch (e) {}

    animFrameRef.current = requestAnimationFrame(detectFrame);
  }, []);

  useEffect(() => {
    let stream: MediaStream;

    async function init() {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const { HandLandmarker, FilesetResolver } = vision;

        setStatus("⚙️ โหลดโมเดล AI...");
        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
        const handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
          baseOptions: { modelAssetPath: HAND_LANDMARKER_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        handLandmarkerRef.current = handLandmarker;

        setStatus("📷 เปิดกล้อง...");
        // Prefer front camera on mobile, fallback for desktop
        const constraints: MediaStreamConstraints = {
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        // setStatus("✅ พร้อมแล้ว! ทำท่า SCUBA 🤿");
        setIsReady(true);

        animFrameRef.current = requestAnimationFrame(detectFrame);
      } catch (err: any) {
        setStatus(`❌ ${err.message}`);
        setIsReady(true);
      }
    }

    init();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      if (scubaTimeoutRef.current) clearTimeout(scubaTimeoutRef.current);
      stream?.getTracks().forEach((t) => t.stop());
      handLandmarkerRef.current?.close?.();
    };
  }, [detectFrame]);

  useEffect(() => {
    const video = catVideoRef.current;
    if (!video) return;

    if (scubaDetected) {
      if (!catOffscreenCanvasRef.current) {
        catOffscreenCanvasRef.current = document.createElement("canvas");
      }
      const offscreenCanvas = catOffscreenCanvasRef.current;
      const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

      let animId: number;

      const renderLoop = () => {
        if (video.readyState >= 2 && offscreenCtx) {
          if (offscreenCanvas.width !== video.videoWidth || offscreenCanvas.height !== video.videoHeight) {
            offscreenCanvas.width = video.videoWidth;
            offscreenCanvas.height = video.videoHeight;
          }

          // 1. Draw video to offscreen canvas
          offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

          // 2. Perform Chroma Key once for this frame
          const imgData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
          const data = imgData.data;

          for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const diff = g - Math.max(r, b);

            if (g > 60 && diff > 30) {
              data[i + 3] = 0;
            } else if (g > 45 && diff > 15) {
              const ratio = (diff - 15) / (30 - 15);
              data[i + 3] = Math.min(data[i + 3], Math.round((1 - ratio) * 255));
            }
          }
          offscreenCtx.putImageData(imgData, 0, 0);

          // 3. Draw to all active cat canvases
          catCanvasRefs.current.forEach((canvas) => {
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            if (canvas.width !== offscreenCanvas.width || canvas.height !== offscreenCanvas.height) {
              canvas.width = offscreenCanvas.width;
              canvas.height = offscreenCanvas.height;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(offscreenCanvas, 0, 0);
          });
        }

        animId = requestAnimationFrame(renderLoop);
      };

      video.currentTime = 0;
      video.play().catch((err) => console.warn("Cat video play blocked:", err));
      animId = requestAnimationFrame(renderLoop);

      return () => {
        cancelAnimationFrame(animId);
        video.pause();
      };
    } else {
      video.pause();
    }
  }, [scubaDetected]);

  return (
    <div className="min-h-screen min-h-dvh bg-gray-950 flex flex-col items-center justify-start pt-4 pb-6 relative overflow-hidden">
      {/* Ocean BG */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: "linear-gradient(180deg,#001428 0%,#002850 60%,#001428 100%)", opacity: 0.7 }}
      />

      {/* Bubbles */}
      {[...Array(10)].map((_, i) => (
        <div
          key={i}
          className="absolute rounded-full border border-cyan-400"
          style={{
            width: `${6 + (i % 5) * 5}px`,
            height: `${6 + (i % 5) * 5}px`,
            left: `${4 + i * 9}%`,
            bottom: `${8 + (i % 4) * 12}%`,
            opacity: 0.2,
            animation: `bubble-float ${3 + (i % 3)}s ease-in-out ${i * 0.35}s infinite alternate`,
          }}
        />
      ))}

      <style>{`
        @keyframes bubble-float {
          0% { transform: translateY(0) scale(1); opacity: 0.2; }
          100% { transform: translateY(-24px) scale(1.1); opacity: 0.05; }
        }
        @keyframes scuba-pop {
          0% { transform: scale(0.4); opacity: 0; }
          65% { transform: scale(1.12); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }
        @keyframes cat-float-1 { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
        @keyframes cat-float-2 { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        @keyframes cat-float-3 { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        @keyframes cat-float-4 { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
        @keyframes cat-float-5 { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }

        .scuba-pop-1 { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) 0s forwards; opacity: 0; }
        .scuba-pop-2 { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) 0.08s forwards; opacity: 0; }
        .scuba-pop-3 { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) 0.16s forwards; opacity: 0; }
        .scuba-pop-4 { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) 0.24s forwards; opacity: 0; }
        .scuba-pop-5 { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) 0.32s forwards; opacity: 0; }

        .cat-float-1 { animation: cat-float-1 2.8s ease-in-out infinite; }
        .cat-float-2 { animation: cat-float-2 2.2s ease-in-out infinite; }
        .cat-float-3 { animation: cat-float-3 2.6s ease-in-out infinite; }
        .cat-float-4 { animation: cat-float-4 2.4s ease-in-out infinite; }
        .cat-float-5 { animation: cat-float-5 3.0s ease-in-out infinite; }

        .scuba-video {
          border-radius: 1rem;
          overflow: hidden;
          width: 100%;
          height: 100%;
          aspect-ratio: 1 / 1;
          object-fit: cover;
          display: block;
        }
        @keyframes wave-char {
          0%,100% { transform: translateY(0); }
          50% { transform: translateY(-7px); }
        }
        @keyframes pulse-glow {
          0%,100% { box-shadow: 0 0 20px #00ffff44; }
          50% { box-shadow: 0 0 40px #00ffff99, 0 0 60px #00ffff33; }
        }
        .scuba-pop { animation: scuba-pop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) forwards; }
        .wave-char span { display: inline-block; animation: wave-char 0.55s ease-in-out infinite; }
        .wave-char span:nth-child(1){animation-delay:0s}
        .wave-char span:nth-child(2){animation-delay:0.08s}
        .wave-char span:nth-child(3){animation-delay:0.16s}
        .wave-char span:nth-child(4){animation-delay:0.24s}
        .wave-char span:nth-child(5){animation-delay:0.32s}
        .wave-char span:nth-child(6){animation-delay:0.4s}
        .cam-glow { animation: pulse-glow 2s ease-in-out infinite; }
        .cam-wrapper {
          position: relative;
          width: 100%;
          max-height: calc(100dvh - 100px);
        }
        .cam-wrapper video,
        .cam-wrapper canvas {
          position: absolute;
          top: 0; left: 0;
          width: 100% !important;
          height: 100% !important;
          object-fit: cover;
        }
        .cam-wrapper canvas { transform: scaleX(1); }
        /* 16:9 on desktop, portrait 9:16 on mobile (fills screen down to the bottom) */
        .cam-ratio { padding-top: 56.25%; }
        @media (max-width: 600px) { .cam-ratio { padding-top: 177.78%; } }
      `}</style>

      <div className="relative z-10 flex flex-col items-center gap-2 w-full px-2 sm:px-4" style={{ maxWidth: "calc(100vw - 16px)" }}>



        {/* Camera wrapper — full viewport width minus tiny margin */}
        <div
          className="cam-wrapper cam-glow overflow-hidden border-2 border-cyan-800 w-full"
          style={{ borderRadius: "0.875rem" }}
        >
          {/* 16:9 ratio on wide screens, 3:4 portrait on mobile */}
          <div className="cam-ratio w-full" />

          <video
            ref={videoRef}
            playsInline
            muted
            style={{ display: isReady ? "block" : "none", transform: "scaleX(-1)" }}
          />
          <canvas
            ref={canvasRef}
            style={{ display: isReady ? "block" : "none" }}
          />

          {/* Loading placeholder */}
          {!isReady && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900">
              <div className="text-4xl mb-3 animate-spin">⚙️</div>
              <p className="text-cyan-400 text-sm text-center px-4">{status}</p>
            </div>
          )}

          {/* Instruction overlay — inside the camera, top-left */}
          {isReady && !scubaDetected && (
            <div className="absolute top-2 left-2 flex flex-col gap-1 pointer-events-none">
              <div className="flex items-center gap-1 bg-black/55 backdrop-blur-sm rounded-lg px-2 py-1">
                <span className="text-base">🤫</span>
                <span className="text-xs text-cyan-300">มือหนึ่งปิดปาก</span>
              </div>
              <div className="flex items-center gap-1 bg-black/55 backdrop-blur-sm rounded-lg px-2 py-1">
                <span className="text-base">🌊</span>
                <span className="text-xs text-cyan-300">อีกมือสบัดไปมา</span>
              </div>
            </div>
          )}

          {/* Debug bar — bottom of camera */}
          {isReady && (
            <div className="absolute bottom-0 left-0 right-0 bg-black/65 backdrop-blur-sm px-2 py-1">
              <p className="text-xs text-cyan-400 font-mono text-center leading-tight">{debugInfo}</p>
            </div>
          )}

          {/* SCUBA overlay — centered on camera */}
          {scubaDetected && (
            <div className="absolute inset-0 pointer-events-none">
              {/* Dark backdrop */}
              <div className="absolute inset-0 bg-black/25 z-0" />

              <div className="relative w-full h-full z-10 flex flex-col items-center justify-between py-6 sm:py-8">


                {/* 5 Cats Container */}
                <div className="absolute inset-0 pointer-events-none">
                  {CAT_CONFIGS.map((cat, index) => (
                    <div key={cat.id} style={cat.style}>
                      <div className={cat.popClass} style={{ width: "100%", height: "100%" }}>
                        <canvas
                          ref={(el) => {
                            catCanvasRefs.current[index] = el;
                          }}
                          className={`scuba-video ${cat.floatClass} ${cat.videoClass}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Bottom Text */}
                <p className="scuba-pop-5 text-white text-sm sm:text-base font-bold animate-bounce drop-shadow">
                  🎉 ท่า SCUBA สำเร็จ! 🎉
                </p>
              </div>
            </div>
          )}
        </div>

        {isReady && (
          <p className="text-xs text-gray-600 text-center">{status}</p>
        )}
      </div>

      <video
        ref={catVideoRef}
        src="/Scuba_cat_Green Screen.mp4"
        loop
        muted
        playsInline
        style={{ display: "none" }}
      />
    </div>
  );
}
