# Validation of additive dense capture tool

Executed in this development environment against artificial inputs:

- `xvfb-run -a -s '-screen 0 1440x900x24' python -m unittest discover -s capture_dataset_dense/tests -v`
  - 37 tests: 27 CPU + 10 Tk GUI. All passed.
- `xvfb-run -a -s '-screen 0 1440x900x24' python -m unittest discover -s capture_dataset/tests -v`
  - 30 original Stage 1 tests. All passed, original files unchanged.
- Compiled Python files and tested CLI help/planning.
- Inspected the selector screenshot using an artificial known-free map with a
  gray corridor. The test reserved 42 corridor positions within the total of 200.
- Compared bytes of every original capture module to the supplied original ZIP.
  They were unchanged. Avatar `core.py`, `render_worker.py` and `run.sh` were also unchanged.

The camera/height tests use actual shared camera matrices, but rendering output
is synthetic for file-writing tests. GPU rasterization, actual scene coverage,
clearance, throughput and real-scene appearance were not tested in this container.
Do not interpret the artificial map/preview as a reconstruction of Valmet's plant.

Tests:

```bash
source .venv/bin/activate
python -m unittest discover -s capture_dataset_dense/tests -v
```

GUI tests skip when DISPLAY is not available. No model download or CUDA is needed
for the CPU/GUI tests. Source map and scene are never changed.
