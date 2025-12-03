import os
from collections import defaultdict

# CHANGE THIS to your real folder
DATA_FOLDER = r"C:\Users\ScottT\Desktop\HAVOC_DATA"

ext_counts = defaultdict(int)
ext_sizes = defaultdict(int)
all_files = []

for root, dirs, files in os.walk(DATA_FOLDER):
    for f in files:
        path = os.path.join(root, f)
        ext = os.path.splitext(f)[1].lower()
        size = os.path.getsize(path)

        ext_counts[ext] += 1
        ext_sizes[ext] += size
        all_files.append(path)

print("\n================ FILE TYPE SUMMARY ================\n")
for ext in sorted(ext_counts.keys()):
    print(f"{ext or '[NO EXT]'} : {ext_counts[ext]} files, {ext_sizes[ext]/1_000_000:.2f} MB total")

print("\n================ SAMPLE FILE PATHS ================\n")
for path in all_files[:50]:  # print first 50 files only
    print(path)

print("\n==================================================")
print(f"TOTAL FILES FOUND: {len(all_files)}")
