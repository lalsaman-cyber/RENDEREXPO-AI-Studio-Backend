from pathlib import Path

path = Path("training_scripts/train_dreambooth_lora_sd3.py")
text = path.read_text()

# 1️⃣ RENDEREXPO PATCH – Relax strict diffusers version check
search_version = 'check_min_version("0.34.0.dev0")'
if search_version in text:
    text = text.replace(
        search_version,
        '# RENDEREXPO PATCH: relax diffusers version requirement\n'
        f'# {search_version}'
    )
    print("✅ Patched: removed strict diffusers version gate")
else:
    print("⚠️ Could not find the check_min_version() block")

# 2️⃣ RENDEREXPO PATCH – Load only real images; ignore .txt files
needle = "instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]"
replacement = '''# RENDEREXPO PATCH: Load only real image files; ignore .txt caption files
exts = {".jpg", ".jpeg", ".png", ".webp"}
instance_paths = [
    p for p in Path(instance_data_root).iterdir()
    if p.is_file() and p.suffix.lower() in exts
]
if len(instance_paths) == 0:
    raise ValueError(f"No valid images found in dataset folder: {instance_data_root}")
instance_images = [Image.open(p) for p in instance_paths]'''

if needle in text:
    text = text.replace(needle, replacement)
    print("✅ Patched: ignore .txt files, load only image extensions")
else:
    print("⚠️ Could not find instance_images loader to patch")

path.write_text(text)
print("🎉 Finished patching train_dreambooth_lora_sd3.py")
