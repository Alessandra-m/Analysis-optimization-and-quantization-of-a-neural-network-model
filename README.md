# Анализ, оптимизация и квантование модели сегментации

Отчёт находится в файле `REPORT.md`, код в папке `candidate/`.

## Структура

```
README.md
REPORT.md
requirements.txt
candidate/
    model.pt, images/, masks/     исходные данные задания
    inspect_model.py              разбор model.pt
    model.py                      восстановленная модель Net
    analyze.py                    загрузка данных, mIoU, таблица слоёв
    optimize.py                   точные упрощения, модель NetOpt
    approx.py                     приближённые упрощения
    benchmark.py                  параметры, MACs, время на CPU
    quantize.py                   квантование INT16 и INT8
    figures/
        draw_unet.py              схема архитектуры
        unet.png                  схема архитектуры (раздел 1 отчёта)
        distributions_*.png       распределения весов и активаций (создаёт quantize.py)
        sensitivity.png           послойная чувствительность (создаёт quantize.py)
```

## Окружение

Python 3.12, PyTorch 2.14.1 (CPU). 

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## Воспроизведение

| Команда | Раздел отчёта | Результат |
|---|---|---|
| `python candidate/inspect_model.py` | 1.1, 2.1 | код forward и подмодулей, формы весов, функция `_p`, аудит всех весов, проверки буферов `m`, `s`, `pi`, `g`, `alpha`, весов b9, мёртвых каналов b4, компенсации масштабов b6 → b7 |
| `python candidate/model.py` | 1.4 | сравнение Net с model.pt: max\|diff\|, совпадение argmax, число параметров |
| `python candidate/analyze.py` | 1.3, 1.4, 2.4 | max\|diff\| и mIoU на изображениях, таблицы слоёв (формы, параметры, MACs) для Net и NetOpt |
| `python candidate/optimize.py` | 2.3, 2.4 | проверка каждого упрощения по отдельности, сравнение NetOpt и Net |
| `python candidate/approx.py` | 2.6 | замена log(softmax) на log_softmax, внесение нормализации в первую свёртку |
| `python candidate/benchmark.py` | 2.5 | параметры, MACs, время инференса (batch 1 и 8, 1 поток и все потоки) |
| `python candidate/quantize.py` | 3 | сравнение распределений калибровки, FP32/INT16/INT8 для обеих моделей, послойная чувствительность, исправления (перцентиль, смешанная точность, CLE); сохраняет графики в `candidate/figures/` |
| `python candidate/figures/draw_unet.py` | 1.2 | схема архитектуры `candidate/figures/unet.png` |

Время инференса зависит от загрузки системы, поэтому `benchmark.py` рекомендуется запускать несколько раз. В отчёте приведена медиана по трём запускам.

## Основные результаты

| | Net (исходная) | NetOpt |
|---|---:|---:|
| Параметры | 893 316 | 505 788 |
| MACs | 843,7 млн | 680,5 млн |
| max\|diff\| относительно model.pt | 0 | 3,3e-5 |
| Ускорение на CPU | 1 | ×1,23…1,41 |
| INT8 после CLE: совпадение argmax с FP32 | 99,94% | 99,994% |
