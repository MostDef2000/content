---
name: LoRA training task
about: Обучение или переподготовка LoRA персонажа valery23
title: "[lora] "
labels: training
assignees: ""
---

## Контекст
<!-- зачем переобучаем: новый датасет, другой стиль, дообучение -->

## Датасет
- [ ] `reference/candidates/` выбран эталон
- [ ] `python scripts/queue_workflow.py expand --reference ... --count 8`
- [ ] `python lora/caption.py` и вручную дописаны `dataset/images/*.txt`
- [ ] минимум 15 пар image+txt

## Запуск
```bash
bash scripts/install_trainer.sh
bash scripts/train_lora.sh
```

## Проверка
- [ ] GPU свободен от других тяжёлых задач (busy-лок в manager)
- [ ] чекпоинт скопирован в `runtime/models/loras/valery23.safetensors`
- [ ] тестовый пост через `python scripts/queue_workflow.py post ...`

## Артефакты
<!-- путь к lora/output, размер, сколько шагов -->
