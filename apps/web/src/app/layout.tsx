import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RiskItem — 信贷风控模型智能监测与自主迭代",
  description: "模型全生命周期智能治理系统",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-gray-50">{children}</body>
    </html>
  );
}
