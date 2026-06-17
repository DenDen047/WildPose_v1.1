# Manual Calibrator

WildPose v1.1 のカメラ手動校正ツール。エントリポイントは `manual_calibrator.py`。

## Pixi environment

設定: `pixi.toml`（旧 `environment.yml` を pixi へ移行）

```bash
pixi install
pixi run calibrate        # = python manual_calibrator.py --config debug_config.json
```

設計判断:
- **conda は `python=3.8` のみ。残りは全て PyPI で元 `environment.yml` の pip 版に pin** —
  元環境は実質すべて pip 由来だったため、出所を PyPI に統一して ABI 整合を保つ。
- **`environment.yml` の bloat は除外** — `manual_calibrator.py` が実際に import する
  パッケージ（numpy/scipy/torch/torchvision/open3d/opencv-python/matplotlib/
  pandas/seaborn/pillow/tqdm/tensorboard）だけを採用。anthropic/openai/dash/flask/
  playwright/google-cloud 等は未使用なので入れていない。
- **対象 platform は `osx-arm64` のみ**（元 `environment.yml` は osx-64 Intel 向けだった）。
  open3d 0.18.0 の osx-arm64 wheel は macOS 13+ が必要。GPU は MPS が利用可。
- **`open3d` は `torch` より前に import**（`manual_calibrator.py` 冒頭で対応済み）—
  両者が別々の libomp を抱え、torch を先に読むと macOS で
  `OMP: Error #179 ... pthread_mutex_init failed` で即クラッシュするため。
- **`OMP_NUM_THREADS=1` を `[activation.env]` で強制** — 上記 libomp 二重ロードは、
  import 順を直しても torch のマルチスレッド処理（DataLoader collation / `torch.stack`）が
  **0% CPU でデッドロック（＝GUI が固まる）**する。1 スレッドに固定して競合を回避。
  この workload は小さく単スレッドでも実用上問題なし。

## 性能メモ

- `visual_debug()` は `next(iter(data_loader))` で最初の1バッチを `num_workers=0` で
  **同期前処理**してからウィンドウを出す。各フレーム `ptcld_to_depth_image()` の
  `griddata(nearest)` が ~0.35s/枚。`debug_config.json` の `batch_size` が大きいと
  起動前に固まって見える（例: 100枚 ≈ 30s）。スクラブ用なので `batch_size` は 5〜15 で十分。
- キー入力ごとの再描画（`model.loss.debug`）は ~0.09s/回で軽い。

## データ

`data -> ../data` シンボリックリンク。校正データ（例: `data/calibration/...`）は
別途用意が必要（リポジトリには含まれない）。
