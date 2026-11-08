#!/usr/bin/env bash
set -e

mkdir -p outputs/renders/batch

# Camera positions and targets
# Format: name eye_x eye_y eye_z target_x target_y target_z

while read -r name ex ey ez tx ty tz; do

    echo "Rendering $name"

    python scripts/render_splat.py \
        --eye "$ex" "$ey" "$ez" \
        --target "$tx" "$ty" "$tz" \
        --up 0 -1 0 \
        --out "outputs/renders/batch/$name"

done <<EOF
view_01 0 0 -3 0 0 0
view_02 1 0 -3 0 0 0
view_03 -1 0 -3 0 0 0
view_04 0 0 -4 0 0 0
view_05 1 0 -4 0 0 0
view_06 -1 0 -4 0 0 0
view_07 0 0 -2 0 0 0
view_08 1 0 -2 0 0 0
view_09 -1 0 -2 0 0 0
view_10 0 0 2 0 0 0
view_11 1 0 2 0 0 0
view_12 -1 0 2 0 0 0
EOF
