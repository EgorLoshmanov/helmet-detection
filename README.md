# FORTNITEBALLS — Helmet Detection

Система обнаружения защитных касок в реальном времени для Firefly
ROC-RK3588S-PC. Базовая модель YOLOv8n различает классы `helmet` и
`no_helmet`; финальная версия будет работать через RKNN на NPU RK3588S и
управлять световой и звуковой сигнализацией.

> Модель v2 экспортирована в ONNX и FP16 RKNN и проверена на сохранённых кадрах.
> Проверка на видео целевой камеры пока невозможна: камера не подключена к стенду.
> Экспериментальный INT8 не прошёл сравнение с ONNX и не готов для сигнализации.

## Быстрый старт

### Приложение для Firefly и веб-панель

Добавлено приложение `firefly_app`: RKNN FP16, зона контроля ROI, подтверждение
по времени, состояния SAFE/PENDING/ALARM/FAULT, журнал SQLite со снимками,
метрики CSV и локальная веб-панель. На Windows можно проверить интерфейс без платы:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r firefly_app/requirements.txt
.\.venv\Scripts\python.exe -m firefly_app --demo
```

Панель: <http://127.0.0.1:8080>. Демо использует синтетические данные.
Запуск с камерой/NPU, настройка периферии и ограничения проверок:
[docs/edge-monitor.md](docs/edge-monitor.md).

### Обучение и локальный PyTorch-инференс

Требуется Conda или Miniconda. После клонирования репозитория:

```bash
conda env create -f environment.yml
conda activate fortniteballs
python scripts/download_model.py
python scripts/predict.py --source 0 --show
```

Последняя команда открывает встроенную камеру. macOS может запросить разрешение
на доступ к ней. Для изображения или видео укажите путь:

```bash
python scripts/predict.py --source path/to/image.jpg --save
python scripts/predict.py --source path/to/video.mp4 --save
```

По умолчанию используется `confidence=0.5`. Изменить порог можно аргументом
`--conf`, например `--conf 0.6`.

### Windows с картой NVIDIA

`environment.yml` ставит ultralytics через pip, а на Windows обычный PyPI отдаёт
CPU-сборку PyTorch. Обучение тогда молча уходит на процессор и замедляется
примерно на порядок. Сразу после создания окружения добавьте CUDA-сборку:

```bash
conda activate fortniteballs
pip install -r requirements-win-cuda.txt
python -c "import sys, torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

Проверка должна напечатать `2.14.0+cu126 True`. Если вместо `True` выводится
`False`, обучение пойдёт на процессоре и дальше идти бессмысленно.

## Подготовка датасета

Датасет собирается из нескольких источников, описанных в
[dataset/sources.yaml](dataset/sources.yaml). Каждый источник задаёт собственное
отображение классов: одно и то же имя класса в разных наборах означает разное,
поэтому незаявленное имя прерывает сборку, а не проходит молча.

Загрузка источников:

```bash
python scripts/download_dataset.py    # Hard Hat Workers, около 1.3 ГБ, Kaggle
python scripts/download_shwd.py       # SHWD, около 1.04 ГБ, Google Drive
python scripts/download_shel5k.py     # печатает инструкцию для Mendeley
```

SHEL5K скачивается вручную, потому что Mendeley Data не отдаёт файлы через
стабильный API; после скачивания передайте архив скрипту:

```bash
python scripts/download_shel5k.py --archive path/to/downloaded.zip
```

Загрузчики считают SHA-256, безопасно распаковывают архив, сверяют количество
файлов и печатают реальные имена классов вместе с готовым скелетом `class_map`
для `sources.yaml`.

Сборка базовой версии v1, побайтово совпадающей с исходной:

```bash
python scripts/prepare_dataset.py --config dataset/sources.yaml --only hard-hat-v1 \
  --no-perceptual --output-root dataset/processed/helmet_yolo_v1 \
  --dataset-yaml dataset/dataset.yaml --reports-prefix dataset --overwrite
python scripts/check_ultralytics_dataset.py
```

Сборка версии v2 после включения нужных источников флагом `enabled: true`:

```bash
python scripts/prepare_dataset.py --config dataset/sources.yaml
```

После выполнения появятся:

```text
dataset/source/hard-hat-detection-v1/
dataset/processed/helmet_yolo_v1/
```

Обе директории исключены из Git. Подробнее о происхождении, лицензиях,
отклонённых источниках и правилах разметки:
[dataset/README.md](dataset/README.md). Во время подготовки
`prepare_dataset.py` проверяет структуру и CRC каждого PNG, структуру маркеров
каждого JPEG, удаляет точные и близкие дубликаты между источниками и записывает
SHA-256 и перцептивный хеш изображений в манифест.

## Обучение

Проверочный запуск на 5% обучающей выборки:

```bash
python training/train.py --smoke
```

Полное обучение:

```bash
python training/train.py
```

Устройство выбирается автоматически в порядке CUDA → Apple MPS → CPU. Его
можно указать явно:

```bash
python training/train.py --device cpu
python training/train.py --device mps
python training/train.py --device 0
```

Веса сохраняются в `models/`, а графики и метрики — в `reports/training/`.
Веса и датасеты намеренно не хранятся в обычной истории Git.

Проверенные стенды обучения:

| Стенд | Устройство | Python | PyTorch |
|---|---|---|---|
| macOS, Apple M3 | MPS | 3.10.21 | 2.14.0 |
| Windows 10, RTX 4060 Laptop 8 ГБ | CUDA | 3.10.21 | 2.14.0+cu126 |

В `training/config.yaml` задано `amp: false` — так был получен baseline на MPS,
где смешанная точность не даёт выигрыша. На NVIDIA включение `amp: true`
примерно в полтора-два раза ускоряет обучение и снижает расход видеопамяти;
флаг попадает в `environment.json` рядом с результатами, поэтому прогоны
остаются различимыми. Параметры окружения и устройства записываются туда
автоматически.

## Готовые модели

`scripts/download_model.py` загружает веса из
[GitHub Releases](https://github.com/EgorLoshmanov/helmet-detection/releases)
и проверяет контрольную сумму перед использованием:

```bash
python scripts/download_model.py                    # v0.2.0, текущая модель
python scripts/download_model.py --version v0.1.0 --output models/baseline_v0.1.0.pt
```

| Релиз | Датасет | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|
| `v0.2.0` | v2, 5464 изображения | 0.904 | 0.811 | 0.885 | 0.534 |
| `v0.1.0` | v1, 5000 изображений | 0.936 | 0.874 | 0.941 | 0.618 |

**Эти две строки нельзя сравнивать между собой.** Они измерены на разных
валидационных выборках: в v2 добавлены кадры с десятками мелких голов, которых в
v1 не было, поэтому выборка объективно труднее. Проверка обеих моделей на
изображениях, отложенных при обоих обучениях, приведена в
[dataset/README.md](dataset/README.md).

Что реально улучшилось в v0.2.0 — баланс классов: precision `helmet` 0.905
против `no_helmet` 0.903, тогда как у baseline был выраженный крен в сторону
`helmet`. На отложенной выборке с полной разметкой mAP50 класса `no_helmet`
составляет 0.921 против 0.839 у baseline.

Обе модели обучены на строительных и учебных сценах. Ни одно из этих чисел не
заменяет проверку на видео с целевой камеры.

## Экспорт ONNX

Скачайте текущую модель отдельно от baseline и выполните воспроизводимый экспорт:

```bash
python scripts/download_model.py --version v0.2.0 \
  --output models/helmet_detector_v0.2.0.pt
python export/export_onnx.py
python export/validate_onnx.py
```

Результат — `models/helmet_detector.onnx` со статическим входом
`1×3×640×640`, opset 12 и выходом `1×6×8400`. Зависимости экспорта и проверки
зафиксированы в `environment.yml`; сам ONNX-файл исключён из Git и передаётся
как отдельный артефакт.

Проверка сравнивает сырые выходы и детекции PyTorch/ONNX на 13 изображениях из
локальной тестовой части v1 (каска, голова без каски, оба класса и многолюдный
кадр). Результат записывается в `reports/onnx_comparison.json`. Для своих кадров
используйте `python export/validate_onnx.py --images path/to/first.jpg path/to/second.jpg`.
Этот тест показывает совпадение двух форматов модели; кадры v1 могут встречаться
в обучении v2 и не подходят для независимой оценки качества детектора.

## RKNN на Firefly

Конвертация использует RKNN-Toolkit2 `2.3.2` для Linux ARM64 и Python 3.10.
Проверенная команда на Mac с Docker Desktop:

```bash
docker build --platform linux/arm64 -f export/Dockerfile.rknn-build \
  -t fortniteballs-rknn:2.3.2 .
docker run --rm --platform linux/arm64 -v "$PWD:/work" -w /work \
  fortniteballs-rknn:2.3.2 --mode fp16
```

В результате появляется `models/helmet_detector_fp16.rknn` (артефакт исключён
из Git). На Firefly он проверен через `export/infer_rknn.py` на 13 одинаковых с
ONNX кадрах: сопоставлены 174 из 175 детекций, медиана вызова
`RKNNLite.inference` — 40.98 мс. Это не полный FPS системы. Отчёт:
`reports/rknn_fp16_comparison.json`.

Для воспроизведения **экспериментальной** INT8-сборки сначала восстановите
калибровочные кадры из подготовленного датасета v1, затем разделите выход ONNX:

```bash
python export/prepare_calibration.py --restore
python export/split_onnx_outputs.py
docker run --rm --platform linux/arm64 -v "$PWD:/work" -w /work \
  fortniteballs-rknn:2.3.2 --onnx models/helmet_detector_split.onnx \
  --mode int8 --dataset export/calibration_dataset.txt \
  --output models/helmet_detector_int8_split.rknn
```

Если `reports/rknn_calibration.json` отсутствует (новая выборка), вместо
`--restore` запустите `python export/prepare_calibration.py` с новыми путями
отчёта и манифеста. Объединённый выход исходного ONNX непригоден для INT8:
вероятности классов округляются до нуля. Разделение восстанавливает вероятности,
но текущая INT8-модель всё ещё выдаёт слишком много лишних детекций. До
дальнейшей работы с квантизацией используйте FP16. Подробности, версии и SHA-256:
[reports/rknn_conversion.md](reports/rknn_conversion.md).

Эксперимент со смешанной INT8/FP16-квантизацией воспроизводится скриптом
`export/convert_rknn_hybrid.py` в том же Docker-образе (с `--entrypoint python`).
Он улучшил число детекций, но также не прошёл контрольное сравнение и не является
моделью для тревожной сигнализации.

## Firefly

Плата находится в университетской сети и доступна только участникам команды
через NetBird и SSH. Пароли и приватные SSH-ключи в репозиторий не добавляются.
Проверенное состояние платы описано в [docs/hardware.md](docs/hardware.md).

## Работа в команде

Перед началом новой задачи:

```bash
git pull --rebase
git switch -c feature/short-task-name
```

После изменений:

```bash
git add <files>
git commit -m "Describe the change"
git push -u origin HEAD
```

Основной план и актуальный статус проекта находятся в
[PIPELINE.md](PIPELINE.md).
