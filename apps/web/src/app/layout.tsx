import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RiskItem 模型运维控制台",
  description: "模型监控与生命周期编排控制台",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
