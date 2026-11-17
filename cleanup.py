import os
import sys
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_dir = os.path.dirname(os.path.abspath(__file__))

print("--- Cleaning up temporary artifacts & test outputs ---")
removed_count = 0

# 1. Old generated EDA reports
for f in glob.glob(os.path.join(project_dir, "eda_report_*.md")):
    try:
        os.remove(f)
        print(f"🗑️ Removed report: {os.path.basename(f)}")
        removed_count += 1
    except Exception as e:
        print(f"❌ Failed to remove {f}: {e}")

# 2. Temporary chat charts and zero-byte files
charts_dir = os.path.join(project_dir, "charts")
if os.path.exists(charts_dir):
    for f in os.listdir(charts_dir):
        fp = os.path.join(charts_dir, f)
        if f.startswith("chat_chart_") or f == "dist_Unnamed" or (os.path.isfile(fp) and os.path.getsize(fp) == 0):
            try:
                os.remove(fp)
                print(f"🗑️ Removed chart: {f}")
                removed_count += 1
            except Exception as e:
                pass

# 3. Temporary test datasets
data_dir = os.path.join(project_dir, "data")
if os.path.exists(data_dir):
    for f in os.listdir(data_dir):
        if f.startswith("test_") or f.startswith("corrupt"):
            fp = os.path.join(data_dir, f)
            try:
                os.remove(fp)
                print(f"🗑️ Removed test data: {f}")
                removed_count += 1
            except Exception:
                pass

print(f"\n✅ Cleaned {removed_count} temporary artifacts.")
print("\n--- Remaining files in project directory ---")
for f in sorted(os.listdir(project_dir)):
    if not f.startswith(".") and f != "__pycache__":
        print(f"  📁 {f}" if os.path.isdir(os.path.join(project_dir, f)) else f"  📄 {f}")
