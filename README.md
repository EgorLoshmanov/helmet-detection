# FORTNITEBALLS — Helmet Detection

Система обнаружения защитных касок в реальном времени для Firefly
ROC-RK3588S-PC. Базовая модель YOLOv8n различает классы `helmet` и
`no_helmet`; финальная версия будет работать через RKNN на NPU RK3588S и
управлять световой и звуковой сигнализацией.

> Базовая PyTorch-модель обучена и запускается, но пока неудовлетворительно
> переносится с исходного строительного датасета на обычную веб-камеру.
> Перед экспортом в ONNX/RKNN требуется датасет v2 с кадрами целевой камеры.

## Быстрый старт

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

## Подготовка датасета

Исходный набор Hard Hat Workers занимает около 1.3 ГБ. Скрипт использует
официальный клиент KaggleHub, загружает зафиксированную версию 1 и проверяет
количество изображений и аннотаций:

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/check_ultralytics_dataset.py
```

После выполнения появятся:

```text
dataset/source/hard-hat-detection-v1/
dataset/processed/helmet_yolo_v1/
```

Обе директории исключены из Git. Подробнее о происхождении, лицензии и
разметке: [dataset/README.md](dataset/README.md). Во время подготовки
`prepare_dataset.py` дополнительно проверяет структуру и CRC каждого PNG и
записывает SHA-256 изображений в манифест.

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

## Готовая baseline-модель

`scripts/download_model.py` загружает `helmet_detector_best.pt` из
[GitHub Release v0.1.0](https://github.com/EgorLoshmanov/helmet-detection/releases/tag/v0.1.0)
и проверяет контрольную сумму перед использованием.

Текущие метрики на исходной валидационной выборке:

| Precision | Recall | mAP50 | mAP50–95 |
|---:|---:|---:|---:|
| 0.936 | 0.874 | 0.941 | 0.618 |

Эти значения не заменяют проверку на видео с целевой камеры.

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
