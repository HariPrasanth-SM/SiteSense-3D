# SiteSense — laptop run sheet

For the already-working HP Victus setup. **Do not reinstall packages or rerun preparation before the presentation.**

## Launch the final dashboard — one terminal

Close the avatar viewer and other live Gaussian-rendering applications first.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
source .venv/bin/activate

bash service_demo/run.sh serve \
  --out outputs/service_demo/manual_v1 \
  --title "SITESENSE 3D" \
  --width 640 --height 480 --render-hz 2 \
  --port 8770 --open-browser
```

Open **http://127.0.0.1:8770** if needed. Keep the terminal running. Use **Ctrl+Shift+R** to refresh an older tab; **F11** makes the browser full-screen.

## Recording sequence

1. **Service mission:** select an equipment group and asset → **Plan route** → **Start**. Show the route and camera feed. The head stays level during travel and faces the object on arrival. Open one service-evidence panel.
2. **Image query:** upload a known example → **Search** → select the intended candidate → **Plan route** → **Start**.
3. **Manual:** hold the direction buttons or arrow keys. **W/S** tilt the view; **H** levels it. **Escape/Space** stops motion. Inputs are ignored while typing.

The 17 annotated objects are the service targets. The 30 workbook placeholder positions are not used.

## Simple fallbacks

Stop with **Ctrl+C** and rerun the same command with the relevant addition:

| Problem | Change |
|---|---|
| Rendering feels heavy | Use `--width 480 --height 360 --render-hz 1` |
| Image-search data/model unavailable | Add `--no-search` |
| Need only the map and interface | Add `--no-render --no-search` |
| Port already occupied | Stop the old server, or use `--port 8771` |
| CUDA fails | Run `nvidia-smi`; keep the traceback. Do not reinstall drivers mid-demo. |

Optional arrival highlighting: append `--arrival-highlight --fastsam FastSAM-s.pt` only if you have already checked it with your selected target.

## Demo 1 — existing recorded animation

```bash
xdg-open outputs/animations/valmet_reveal_showcase25_v1/semantic_showcase.mp4
```

Use your actual successful animation folder if different. The all-object video is normally `outputs/animations/valmet_reveal_all_v2/semantic_reveal.mp4`.

**Stop:** Ctrl+C in the dashboard terminal. Source data and annotations remain saved.
