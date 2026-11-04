from ultralytics import YOLO

model = YOLO("runs/detect/train-3/weights/best.pt")
results = model("data/valid/images", save=True, conf=0.25)

counts = {}
confs = []
for r in results:
    n = len(r.boxes)
    counts[n] = counts.get(n, 0) + 1
    for box in r.boxes:
        confs.append(float(box.conf[0]))

total = len(results)
print(f"Total validation images: {total}")
for n, c in sorted(counts.items()):
    print(f"  Images with {n} detection(s): {c} ({100*c/total:.1f}%)")
if confs:
    print(f"Confidence range: {min(confs):.2f} - {max(confs):.2f}, avg {sum(confs)/len(confs):.2f}")