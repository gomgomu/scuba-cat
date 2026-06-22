import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "🤿 SCUBA Detector",
  description: "ทำท่า SCUBA แล้วดูแมวดำน้ำ!",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="th">
      <body style={{ margin: 0, padding: 0, overscrollBehavior: "none" }}>
        {children}
      </body>
    </html>
  );
}
