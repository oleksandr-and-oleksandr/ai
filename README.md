# Guitar Neck Segmentation (YOLOv8)

Тренування моделі YOLOv8-seg для сегментації грифа гітари на зображеннях. Датасет — `guitar-nut-and-neck` з Roboflow Universe.

## Стек

- Python 3.12, Google Colab (GPU Tesla T4)
- Ultralytics YOLOv8 (segmentation)
- PyTorch 2.10 + CUDA 12.8
- Roboflow SDK (для завантаження датасету)
- pandas, matplotlib (аналіз метрик)

## Структура

- `yolo_train.ipynb` — основний ноутбук: завантаження датасету, тренування, оцінка моделі, побудова графіків метрик.

## Запуск

1. Відкрийте `yolo_train.ipynb` у Google Colab.
2. Увімкніть GPU: **Runtime → Change runtime type → T4 GPU**.
3. Підставте свій `api_key` з [Roboflow](https://roboflow.com/).
4. Запустіть усі клітинки (**Runtime → Run all**).

## Результати тренування

Скрипт зчитує `results.csv` з `runs/guitar/seg_v1/` і будує графіки:

- **Loss:** Box / Segmentation / Classification (train + val)
- **mAP@0.5** і **mAP@0.5:0.95** для bounding box і mask
- **Precision / Recall** для mask

Головна метрика — **Mask mAP50**.

Ваги натренованої моделі: `runs/guitar/seg_v1/weights/best.pt`.

## TODO

- [ ] Інференс у реальному часі з веб-камери
- [ ] Експорт моделі в ONNX / TFLite
- [ ] AR-накладання поверх грифа
