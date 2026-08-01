"use client";

import type { ReactNode } from "react";

/* ── Types ── */

export type Envelope<T> = {
  success?: boolean; code?: string; message?: string;
  data?: T; trace_id?: string; details?: unknown;
};
export type Items<T> = { items: T[]; total?: number };

/* ── API ── */

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_MODEL_OPS_API_BASE ?? "http://localhost:8001";

export async function requestJson<T>(apiBase: string, path: string, init?: RequestInit): Promise<T> {
  const base = apiBase.trim().replace(/\/+$/, "");
  const url = `/api/modelops${path}`;
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", "X-ModelOps-Api-Base": base, ...(init?.headers ?? {}) },
  });
  const payload = await response.json().catch(() => ({ message: "invalid json" }));
  if (!response.ok) {
    throw new Error((payload as Envelope<unknown>).message || JSON.stringify(payload));
  }
  const envelope = payload as Envelope<T>;
  return envelope.data === undefined ? (payload as T) : envelope.data;
}

export function getApiBase() { return DEFAULT_API_BASE; }

/* ── Formatters ── */

export function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

export function fmtTs(iso?: string | null): string {
  if (!iso) return "-";
  try { return iso.slice(0, 16).replace("T", " "); } catch { return String(iso); }
}

/* ── Status helpers ── */

export function statusColor(status?: string | null): "green" | "amber" | "red" | "blue" {
  const s = (status ?? "").toUpperCase();
  if (/SUCCEEDED|COMPLETED|PROMOTED|PASS/.test(s)) return "green";
  if (/RUNNING|PENDING|PROCESSING/.test(s)) return "blue";
  if (/FAILED|ROLLBACK|ERROR|REJECT/.test(s)) return "red";
  return "amber";
}

export function phaseColor(phase?: string | null): string {
  const p = (phase ?? "").toUpperCase();
  if (/FAILED|ERROR|REJECT|ROLLBACK/.test(p)) return "border-red-200 bg-red-50 text-red-700";
  if (/WAITING|MANUAL|PENDING|HOLD/.test(p)) return "border-amber-200 bg-amber-50 text-amber-700";
  if (/CLOSED|COMPLETED|PROMOTED|SUCCEEDED/.test(p)) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (/AGENT|ITERAT|DIAGNOS|MONITOR|VALIDAT|CANARY|DEPLOY/.test(p)) return "border-sky-200 bg-sky-50 text-sky-700";
  return "border-slate-200 bg-white text-slate-700";
}

/* ── Shared UI Components ── */

export function Panel({ title, action, children, className }: { title: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`dash-card ${className ?? ""}`}>
      <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-0">
        <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
        {action}
      </div>
      <div className="dash-card-body">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="stat-tile">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

export function Badge({ label, color }: { label: string; color: "green" | "amber" | "red" | "slate" | "blue" }) {
  const colors: Record<string, string> = {
    green: "border-emerald-200 bg-emerald-50 text-emerald-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
    red: "border-red-200 bg-red-50 text-red-700",
    slate: "border-slate-200 bg-slate-50 text-slate-600",
    blue: "border-sky-200 bg-sky-50 text-sky-700",
  };
  return (
    <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold ${colors[color]}`}>
      {label}
    </span>
  );
}

export function StatusDot({ status }: { status?: string | null }) {
  const c = statusColor(status);
  const pulse = c === "blue";
  return <span className={`status-dot ${c}${pulse ? " pulse" : ""}`} />;
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />;
}

export function Empty({ text }: { text: string }) {
  return <div className="rounded-lg border border-dashed border-slate-300 p-6 text-sm text-slate-400 text-center">{text}</div>;
}

export function Btn({ children, onClick, disabled, primary, danger }: { children: ReactNode; onClick?: () => void; disabled?: boolean; primary?: boolean; danger?: boolean }) {
  let cls = "rounded-lg px-3 py-2 text-sm font-semibold transition disabled:opacity-40 disabled:cursor-not-allowed ";
  if (danger) cls += "bg-red-600 text-white hover:bg-red-700";
  else if (primary) cls += "bg-indigo-600 text-white hover:bg-indigo-700";
  else cls += "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50";
  return <button className={cls} onClick={onClick} disabled={disabled}>{children}</button>;
}
