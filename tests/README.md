# tests

Автономный MVP-пакет по ТЗ из `source/1.md`.

Содержит:
- генераторы poset-структур: chain, tree, grid;
- synthetic upper-set classification task;
- модели MLP, GCN и simplified SNN;
- TDA-пайплайн на hidden embeddings;
- графики и JSON-логи по эпохам.

## Запуск

Из папки `tests`:

```bash
python mvp.py --poset tree --epochs 200 --save_every 10 --seed 0 --outdir runs/tree_seed0
```

Для установки зависимостей:

```bash
pip install -r requirements.txt
```

Если `ripser` не ставится, попробуйте запасной backend:

```bash
pip install giotto-tda
```

## Результат

После запуска создаются файлы:
- `history.json`
- `summary.json`
- `meta.json`
- `metrics.png`

## Примечание

Папка называется `tests` по запросу, но по смыслу это отдельный воспроизводимый MVP для исследования, а не unit-test suite.
