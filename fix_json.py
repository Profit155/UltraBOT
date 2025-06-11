import json, pathlib, sys

FN_IN  = "logs/ping.json"
FN_OUT = "logs/ping_fixed.json"

raw = pathlib.Path(FN_IN).read_text()
# обрезаем по одному символу с конца, пока json не распарсится
for cut in range(0, 10_000):                 # хватит с запасом
    try:
        data = json.loads(raw[:-cut] + "}")  # добавляем финальную скобку
        print("Fixed! cut", cut, "chars, events:", len(data))
        pathlib.Path(FN_OUT).write_text(json.dumps(data))
        break
    except json.JSONDecodeError:
        continue
else:
    sys.exit("Не удалось починить JSON — файл слишком битый")
