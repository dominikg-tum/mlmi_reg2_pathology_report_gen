from pathlib import Path
import openslide

input_dir = Path("/mnt/projects/mlmi/TUMUntera/TUM_Untera_data")
output_dir = Path("/mnt/projects/mlmi/reg2/dataset/thumbnails")
output_dir.mkdir(parents=True, exist_ok=True)

thumb_size = (1024, 1024)

svs_files = sorted(input_dir.rglob("*.svs"))

print(f"Found {len(svs_files)} SVS files")

for svs_path in svs_files:
    try:
        relative_path = svs_path.relative_to(input_dir)
        out_path = output_dir / relative_path.with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        slide = openslide.OpenSlide(str(svs_path))
        thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
        thumbnail.save(out_path, "JPEG", quality=95)
        slide.close()

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed: {svs_path}")
        print(f"Error: {e}")