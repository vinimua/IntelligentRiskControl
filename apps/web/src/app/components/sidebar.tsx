"use client";

const NAV_ITEMS = [
  { section: "控制台", items: [
    { key: "overview", icon: "◈", label: "总览" },
    { key: "workflow", icon: "◎", label: "流程控制" },
    { key: "deployment", icon: "⬡", label: "部署监控" },
    { key: "task4", icon: "▣", label: "任务四" },
  ]},
  { section: "观测", items: [
    { key: "monitoring", icon: "◉", label: "监控看板" },
    { key: "state", icon: "▤", label: "状态详情" },
  ]},
  { section: "知识", items: [
    { key: "kg", icon: "⬢", label: "KG 校准" },
  ]},
];

export default function Sidebar({ active, onNav, collapsed, onToggle, alerts }: {
  active: string; onNav: (k: string) => void; collapsed: boolean; onToggle: () => void; alerts?: number;
}) {
  return (
    <>
      {/* Brand */}
      <div className="px-4 pt-5 pb-3">
        <div className="flex items-center gap-3">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500 text-white text-sm font-bold">R</span>
          {!collapsed && <div>
            <div className="text-sm font-semibold text-slate-100 leading-tight">RiskItem</div>
            <div className="text-[10px] text-slate-500 font-medium">ModelOps</div>
          </div>}
        </div>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((group) => (
          <div key={group.section} className="mb-3">
            {!collapsed && <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-[.12em] text-slate-500">{group.section}</div>}
            {group.items.map((item) => (
              <div
                key={item.key}
                className={`sidebar-item${active === item.key ? " active" : ""}`}
                onClick={() => onNav(item.key)}
                title={collapsed ? item.label : undefined}
              >
                <span className="icon">{item.icon}</span>
                <span className="label">{item.label}</span>
                {item.key === "deployment" && alerts ? <span className="sidebar-badge">{alerts}</span> : null}
              </div>
            ))}
          </div>
        ))}
      </nav>

      {/* Collapse toggle */}
      <div className="sidebar-collapse-btn" onClick={onToggle}>
        {collapsed ? "▶" : "◀"} <span className="label">{collapsed ? "" : "收起"}</span>
      </div>
    </>
  );
}
