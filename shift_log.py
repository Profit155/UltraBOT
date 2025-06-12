import json
import pathlib

SRC = "logs/run.json"          # исходник
DST = "logs/run_sync.json"     # новый лог

with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)
shift = float(next(iter(data)))          # 1749066824.5830
print("Shift =", shift)

sync = {f"{float(t)-shift:.3f}": evt     # вычитаем shift, округляем до 0.001
        for t, evt in data.items()
        if float(t) >= shift}            # отбрасываем отрицательные ключи

pathlib.Path(DST).write_text(json.dumps(sync))
print("Сохранено", len(sync), "событий  →", DST)
