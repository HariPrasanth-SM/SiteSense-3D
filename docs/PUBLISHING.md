# Publish the source, not the working directory

Keep the currently working project in place. Make a **separate staging folder** for GitHub so recordings, checkpoints, and intermediate data do not accidentally enter Git history.

## 1. Extract the documentation kit separately

```bash
cd ~/SpaceBots_2026/valmet-metaverse
KIT="$(mktemp -d)"
unzip -q ~/Downloads/sitesense_repository_kit.zip -d "$KIT"
```

The temporary folder holds the new documentation. The staging helper uses it without replacing the README, .gitignore, or source in your working project. Keep using this same terminal so `$KIT` remains set.

In the new staging folder, replace `GOOGLE_DRIVE_OBJECT_ASSETS_URL` in `README.md` after uploading the object ZIP. Add your recording at `docs/media/sitesense-walkthrough.gif` and replace the placeholder image line. Add a separate full-video link when ready.

## 2. Stage your current source

```bash
source .venv/bin/activate
python "$KIT/tools/stage_repository.py" \
  --project "$PWD" \
  --out ../sitesense-submission
```

The helper copies your **actual current module files**, not historical ZIP versions. It also copies the documentation, the current map into `examples/map/`, and an inspection copy of your annotations into `examples/manual_annotations.json`. Absolute project paths are redacted only in that example copy; original files remain unchanged.

It excludes virtual environments, checkpoints, output directories, old backup folders, and vendor repositories. It does not run models, create a remote repository, or push anything.

Choose a few useful prompt images yourself and put them in:

```text
../sitesense-submission/examples/query_images/
```

The company workbook snapshot is skipped by default. If you have permission to publish it, rerun staging to a NEW output directory with `--include-company-data`. Synthetic does not automatically mean redistributable. Review scene-derived images/annotations before publication too.

## 3. Put the large object exports on Drive

```bash
python ../sitesense-submission/tools/package_object_assets.py \
  --project "$PWD" \
  --enhancement outputs/semantic_enhance/valmet_regions150_v1 \
  --regions outputs/semantic_scene/valmet_regions150_v1 \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --zip outputs/submission/sitesense_object_assets.zip
```

The ZIP preserves individual cleaned clouds, original Gaussian subsets, region metadata, and selected evidence. It records the **actual** exported count in `ASSET_MANIFEST.json`; this may be 112 even though the development collection was around 115 regions. It does not package model weights or the original whole-room splat by default.

This is an **inspection asset bundle**, not a complete runnable dashboard bundle. See [DATA.md](DATA.md) for the additional runtime files.

Upload the resulting ZIP to an approved Google Drive location, check that judges can access it, and paste the share URL into the README placeholder. Do not upload the ZIP to ordinary Git history.

## 4. Create an empty GitHub repository and push

Create an empty repository named `sitesense` in your GitHub account. Then:

```bash
cd ~/SpaceBots_2026/sitesense-submission

git init -b main
git add .
git status --short
git diff --cached --stat

# Inspect the staged list before committing.
git commit -m "Add SiteSense hackathon prototype and documentation"
git remote add origin https://github.com/YOUR_USER_OR_ORG/sitesense.git
git push -u origin main
```

Replace `YOUR_USER_OR_ORG`. Authentication is through your existing GitHub setup; do not put tokens in source files or command examples.

## Before sharing

Check the Drive link and GIF; confirm company-data permissions; remove accidental private paths, credentials, recordings, or caches. Select a code license with your team rather than applying one automatically to company data or third-party models. No project license is imposed by this kit.

GitHub blocks regular Git files larger than 100 MiB. More importantly, do not commit a whole `.venv`, model checkpoints, or multi-gigabyte capture outputs even when individual files are smaller.

Sources: [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [adding local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
