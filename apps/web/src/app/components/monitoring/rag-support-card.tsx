"use client";

import { Empty, formatValue } from "../shared";
import type { SupportingDocument } from "./monitoring-types";

type Props = {
  documents: SupportingDocument[] | null;
  loaded: boolean;
  diagnosisStatus: string | null;
};

function formatScore(score?: number | null): string {
  if (score === undefined || score === null) return "-";
  const normalized = score <= 1 ? score * 100 : score;
  return `${normalized.toFixed(1)}%`;
}

export default function RagSupportCard({ documents, loaded, diagnosisStatus }: Props) {
  const docs = documents ?? [];
  const hasDocs = docs.length > 0;
  const diagnosisDone = diagnosisStatus === "COMPLETED";

  return (
    <section className="rounded-xl border border-indigo-100 bg-indigo-50/40 px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-indigo-500" />
            <h3 className="text-sm font-bold text-slate-800">RAG 参考资料</h3>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            仅用于增强诊断报告、补充相似案例和知识片段，不参与根因判定和候选排序。
          </p>
        </div>
        <span className="inline-flex rounded-md border border-indigo-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-indigo-700">
          REPORT_REFERENCE_ONLY · 决策影响 NONE
        </span>
      </div>

      {!loaded ? (
        <div className="mt-3 rounded-lg border border-dashed border-indigo-200 bg-white/60 p-4 text-center text-sm text-slate-400">
          正在读取诊断报告参考资料…
        </div>
      ) : hasDocs ? (
        <div className="mt-3 space-y-2">
          <div className="text-xs text-indigo-700">
            已召回 <span className="font-bold">{docs.length}</span> 条参考资料
          </div>
          <div className="grid gap-2">
            {docs.slice(0, 5).map((doc, index) => (
              <div
                key={`${doc.document_id}-${doc.chunk_id}-${index}`}
                className="rounded-lg border border-indigo-100 bg-white px-3 py-2"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-slate-800">
                      {formatValue(doc.section_path || doc.document_id)}
                    </p>
                    <p className="mt-0.5 break-all font-mono text-[10px] text-slate-400">
                      doc={formatValue(doc.document_id)} · chunk={formatValue(doc.chunk_id)}
                    </p>
                  </div>
                  <span className="shrink-0 rounded bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-700">
                    {formatScore(doc.score)}
                  </span>
                </div>
                {doc.page_number != null && (
                  <p className="mt-1 text-[10px] text-slate-400">页码：{doc.page_number}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-3">
          <Empty
            text={
              diagnosisDone
                ? "RAG 已接入，但本次诊断暂未召回参考资料；根因结论仍由 KG 与证据链给出。"
                : "诊断完成后，如召回到相似案例或知识片段，会在这里展示。"
            }
          />
        </div>
      )}
    </section>
  );
}
