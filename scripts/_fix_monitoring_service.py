"""修复 monitoring_service.py：删除损坏的 _process_rolling_cycle，恢复类结构。"""
import re

path = "apps/modelops_api/services/monitoring/monitoring_service.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 找到 _emit_alert 方法结尾和 broken _process_rolling_cycle / run_full_pipeline 区域
# 删除从 "async def run_full_pipeline(" 第一次被错误替换到 "async def run_full_pipeline(" 正确位置之间的损坏代码

# Step 1: 找出所有 "async def run_full_pipeline" 出现
lines = content.split("\n")
bad_start = None
good_start = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("async def run_full_pipeline("):
        if bad_start is None:
            bad_start = i  # 第一个是损坏的
        else:
            good_start = i  # 第二个是原始正确的（如果存在）

if bad_start is None:
    print("No broken run_full_pipeline found!")
    exit(0)

# 找到 run_full_pipeline 之前正确的 _persist_metric 方法结尾
# 往上找 class MonitoringService 或上一个方法结尾
prev_method_end = bad_start - 1
while prev_method_end > 0:
    line = lines[prev_method_end].strip()
    if line == "" and prev_method_end < bad_start - 1:
        # 找到空行分隔
        pass
    prev_method_end -= 1

# 删掉从第一个 async def run_full_pipeline 到
# 真正的 async def run_full_pipeline（如果存在）之间的所有内容
# 然后用正确的 run_full_pipeline 签名替换

# 找到 _emit_alert 的最后一行
emit_alert_end = None
for i in range(bad_start - 1, 0, -1):
    if "await self.repo.update_metric_triggered" in lines[i]:
        emit_alert_end = i
        break

if emit_alert_end is None:
    print("Could not find _emit_alert end!")
    exit(1)

# 找到文件中正确的 run_full_pipeline（第二个）
correct_run = None
for i in range(bad_start + 1, len(lines)):
    if lines[i].strip().startswith("async def run_full_pipeline(") and "self," in lines[i + 1]:
        correct_run = i
        break

if correct_run is None:
    print("Could not find correct run_full_pipeline!")
    # 尝试找任何包含 self 的 run_full_pipeline
    for i in range(bad_start + 1, len(lines)):
        if "async def run_full_pipeline" in lines[i]:
            print(f"  Found at line {i}: {lines[i].strip()}")
            if i + 1 < len(lines) and "self" in lines[i + 1]:
                correct_run = i
                break

# 删除损坏的代码段（从 bad_start 到 correct_run 之前）
if correct_run:
    print(f"Removing lines {bad_start} to {correct_run - 1}")
    # 保留 correct_run 开始的正常代码
    new_lines = lines[:bad_start] + lines[correct_run:]
else:
    # 如果找不到正确的，手动修复第一个
    print(f"Manually fixing line {bad_start}")
    lines[bad_start] = "    async def run_full_pipeline("
    lines[bad_start + 1] = "        self,"
    new_lines = lines

with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print("Done. Verifying...")

# Verify
import sys
sys.path.insert(0, ".")
try:
    from apps.modelops_api.services.monitoring.monitoring_service import MonitoringService
    assert hasattr(MonitoringService, "run_full_pipeline"), "run_full_pipeline not found!"
    assert hasattr(MonitoringService, "_emit_alert"), "_emit_alert not found!"
    print("Fix verified OK")
except Exception as e:
    print(f"Still broken: {e}")
