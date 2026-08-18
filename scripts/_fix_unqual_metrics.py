"""一次性: 修复 train.py 不合格分支不写 metrics.json."""
path = "apps/modelops_api/services/monitoring/sentinel/train.py"
with open(path, encoding="utf-8") as f:
    text = f.read()

# 找到 "else:\n            metrics[\"published_artifact_path\"] = None"
old_marker = '        metrics["published_artifact_path"] = None\n        metrics["active_model_changed"] = False'

if old_marker not in text:
    # 可能已经被修了
    print("Already fixed or format changed")
    exit(0)

# 找到 else 分支结束（qualification_failure 块的 } )
else_start = text.rfind("        else:")
pos = text.index(old_marker, else_start)
# 从 pos 往后找到 qualification_failure 的 closing }
qf_start = text.index('"qualification_failure"', pos)
# 找到这个 dict 的结束
brace_count = 0
qf_end = qf_start
for i in range(qf_start, len(text)):
    if text[i] == '{':
        brace_count += 1
    elif text[i] == '}':
        brace_count -= 1
        if brace_count == 0:
            qf_end = i + 1
            break

old_block = text[pos:qf_end]

new_block = '''        metrics["published_artifact_path"] = None
        metrics["active_model_changed"] = False
        metrics["active_artifact_path"] = None

        if active_path.is_file():
            try:
                prev = _json.loads(active_path.read_text(encoding="utf-8"))
                metrics["active_artifact_path"] = str(
                    artifact_dir / prev["artifact_path"]
                )
                metrics["active_sentinel_version"] = prev.get("sentinel_version")
                metrics["active_training_run_id"] = prev.get("training_run_id")
            except Exception:
                pass

        # 不合格候选模型也保留完整失败指标
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            _json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        ''' + text[qf_start:qf_end]

text = text.replace(old_block, new_block)
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed")
