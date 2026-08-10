"use client";

import type { CoverageSummary } from "./monitoring-types";
import { CATEGORY_LABELS, CATEGORY_ORDER } from "./monitoring-types";

type Props = {
  summary: CoverageSummary | null;
  activeCategory: string;
  onSelectCategory: (cat: string) => void;
};

export default function CategoryCards({ summary, activeCategory, onSelectCategory }: Props) {
  if (!summary) return null;

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {CATEGORY_ORDER.map((cat) => {
        const bd = summary.category_breakdown[cat];
        if (!bd) return null;

        const label = CATEGORY_LABELS[cat] || cat;
        const hasWarning = bd.warning > 0;
        const hasCritical = bd.critical > 0;
        const hasUnmonitored = bd.unmonitored > 0;
        const hasUnavailable = bd.unavailable > 0;

        let borderColor = "border-emerald-200 bg-emerald-50";
        let textColor = "text-emerald-700";
        let dotColor = "bg-emerald-500";

        if (hasCritical) {
          borderColor = "border-red-200 bg-red-50";
          textColor = "text-red-700";
          dotColor = "bg-red-500";
        } else if (hasWarning) {
          borderColor = "border-amber-200 bg-amber-50";
          textColor = "text-amber-700";
          dotColor = "bg-amber-500";
        } else if (hasUnmonitored || hasUnavailable) {
          borderColor = "border-slate-200 bg-slate-50";
          textColor = "text-slate-500";
          dotColor = "bg-slate-400";
        }

        const isActive = activeCategory === cat;

        return (
          <button
            key={cat}
            onClick={() => onSelectCategory(cat)}
            className={`rounded-xl border px-4 py-3.5 text-left transition hover:shadow-sm ${borderColor} ${
              isActive ? "ring-2 ring-indigo-400 ring-offset-1" : ""
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`inline-block h-2.5 w-2.5 rounded-full ${dotColor}`} />
              <span className={`text-sm font-semibold ${textColor}`}>{label}</span>
            </div>
            <div className="mt-2 flex items-baseline gap-1">
              <span className={`text-xl font-bold ${textColor}`}>
                {bd.normal}
              </span>
              <span className="text-xs text-slate-400">/ {bd.total} 正常</span>
            </div>
            {(hasWarning || hasCritical || hasUnmonitored) && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {hasCritical && (
                  <span className="inline-flex items-center rounded border border-red-200 bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-600">
                    {bd.critical} 严重
                  </span>
                )}
                {hasWarning && (
                  <span className="inline-flex items-center rounded border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-600">
                    {bd.warning} 预警
                  </span>
                )}
                {hasUnmonitored && (
                  <span className="inline-flex items-center rounded border border-slate-200 bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-500">
                    {bd.unmonitored} 未监控
                  </span>
                )}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
