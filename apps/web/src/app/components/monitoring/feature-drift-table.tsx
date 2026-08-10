"use client";

import { useState, useMemo } from "react";
import type { FeatureDriftItem } from "./monitoring-types";
import { formatValue } from "../shared";
import FeatureDriftDrawer from "./feature-drift-drawer";

type Props = {
  items: FeatureDriftItem[];
  loading: boolean;
};

type SortKey = "feature_name" | "psi_7d" | "psi_30d" | "max_psi" | "status" | "model_importance" | "trend";

export default function FeatureDriftTable({ items, loading }: Props) {
  const [sortBy, setSortBy] = useState<SortKey>("max_psi");
  const [sortAsc, setSortAsc] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<FeatureDriftItem | null>(null);

  const sorted = useMemo(() => {
    const arr = [...items];
    const importanceOrder: Record<string, number> = { "高": 3, "中": 2, "低": 1 };
    const statusOrder: Record<string, number> = { critical: 3, warning: 2, normal: 1 };
    const trendOrder: Record<string, number> = { up: 3, down: 2, stable: 1 };

    arr.sort((a, b) => {
      let va: number | string = 0;
      let vb: number | string = 0;
      switch (sortBy) {
        case "feature_name":
          va = a.feature_name;
          vb = b.feature_name;
          break;
        case "psi_7d":
          va = a.psi_7d ?? -1;
          vb = b.psi_7d ?? -1;
          break;
        case "psi_30d":
          va = a.psi_30d ?? -1;
          vb = b.psi_30d ?? -1;
          break;
        case "max_psi":
          va = a.max_psi;
          vb = b.max_psi;
          break;
        case "status":
          va = statusOrder[a.status] ?? 0;
          vb = statusOrder[b.status] ?? 0;
          break;
        case "model_importance":
          va = importanceOrder[a.model_importance ?? ""] ?? 0;
          vb = importanceOrder[b.model_importance ?? ""] ?? 0;
          break;
        case "trend":
          va = trendOrder[a.trend] ?? 0;
          vb = trendOrder[b.trend] ?? 0;
          break;
      }
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
    return arr;
  }, [items, sortBy, sortAsc]);

  function handleSort(key: SortKey) {
    if (sortBy === key) setSortAsc(!sortAsc);
    else {
      setSortBy(key);
      setSortAsc(key === "feature_name");
    }
  }

  function SortIcon({ column }: { column: SortKey }) {
    if (sortBy !== column) return <span className="text-slate-300 ml-0.5">↕</span>;
    return <span className="text-indigo-500 ml-0.5">{sortAsc ? "↑" : "↓"}</span>;
  }

  const trendIcon: Record<string, string> = { up: "↑", down: "↓", stable: "→" };
  const trendColor: Record<string, string> = { up: "text-red-500", down: "text-emerald-500", stable: "text-slate-400" };

  if (loading) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
        加载特征漂移数据中...
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-400">
        该运行无漂移数据
      </div>
    );
  }

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs text-slate-500">
              <Th onClick={() => handleSort("feature_name")}>特征 <SortIcon column="feature_name" /></Th>
              <Th onClick={() => handleSort("psi_7d")}>PSI 7D <SortIcon column="psi_7d" /></Th>
              <Th onClick={() => handleSort("psi_30d")}>PSI 30D <SortIcon column="psi_30d" /></Th>
              <Th onClick={() => handleSort("max_psi")}>最大 PSI <SortIcon column="max_psi" /></Th>
              <Th>阈值</Th>
              <Th onClick={() => handleSort("status")}>状态 <SortIcon column="status" /></Th>
              <Th onClick={() => handleSort("model_importance")}>模型重要性 <SortIcon column="model_importance" /></Th>
              <Th onClick={() => handleSort("trend")}>趋势 <SortIcon column="trend" /></Th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => (
              <tr
                key={item.feature_name}
                className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer transition"
                onClick={() => setSelectedFeature(item)}
              >
                <td className="px-3 py-2.5 font-medium text-slate-700">{item.feature_name}</td>
                <td className="px-3 py-2.5 font-mono text-xs">{formatValue(item.psi_7d)}</td>
                <td className="px-3 py-2.5 font-mono text-xs">{formatValue(item.psi_30d)}</td>
                <td className="px-3 py-2.5 font-mono text-xs font-semibold">{formatValue(item.max_psi)}</td>
                <td className="px-3 py-2.5 font-mono text-xs text-slate-500">{formatValue(item.threshold)}</td>
                <td className="px-3 py-2.5">
                  <span
                    className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      item.status === "critical"
                        ? "bg-red-50 text-red-700 border border-red-200"
                        : item.status === "warning"
                        ? "bg-amber-50 text-amber-700 border border-amber-200"
                        : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    }`}
                  >
                    {item.status === "critical" ? "严重" : item.status === "warning" ? "预警" : "正常"}
                  </span>
                </td>
                <td className="px-3 py-2.5 text-xs text-slate-500">
                  {item.model_importance || "—"}
                </td>
                <td className={`px-3 py-2.5 text-sm font-bold ${trendColor[item.trend]}`}>
                  {trendIcon[item.trend]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedFeature && (
        <FeatureDriftDrawer
          item={selectedFeature}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </>
  );
}

function Th({ children, onClick }: { children: React.ReactNode; onClick?: () => void }) {
  return (
    <th className="px-3 py-2 text-left font-semibold select-none hover:text-slate-700 whitespace-nowrap" onClick={onClick} style={onClick ? { cursor: "pointer" } : undefined}>
      {children}
    </th>
  );
}
