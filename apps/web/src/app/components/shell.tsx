"use client";

import type { ReactNode } from "react";

export default function Shell({ topbar, sidebar, children, sidebarCollapsed }: {
  topbar: ReactNode; sidebar: ReactNode; children: ReactNode; sidebarCollapsed: boolean;
}) {
  return (
    <div className="shell">
      <header className="shell-topbar">{topbar}</header>
      <aside className={`shell-sidebar${sidebarCollapsed ? " collapsed" : ""}`}>{sidebar}</aside>
      <main className={`shell-main${sidebarCollapsed ? " expanded" : ""}`}>{children}</main>
    </div>
  );
}
