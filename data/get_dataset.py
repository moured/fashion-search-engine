import kagglehub
import shutil
import os

# Step 1: Download dataset
downloaded_path = kagglehub.dataset_download("vikashrajluhaniwal/fashion-images")
print(f"\n📥 Dataset downloaded to: {downloaded_path}")

# Step 2: Print contents of downloaded path
print("\n📂 Contents of downloaded path:")
for f in os.listdir(downloaded_path):
    full_path = os.path.join(downloaded_path, f)
    print(f"  - {f}  {'[DIR]' if os.path.isdir(full_path) else '[FILE]'}")

# Step 3: Detect subfolders inside downloaded path
subfolders = [f for f in os.listdir(downloaded_path) if os.path.isdir(os.path.join(downloaded_path, f))]

if not subfolders:
    print("\n❌ No subfolders found. Please check if the dataset is nested or not extracted properly.")
    exit()

# Step 4: Use first subfolder as dataset root
data_root = os.path.join(downloaded_path, subfolders[0])
print(f"\n✅ Using dataset folder: {data_root}")

# Step 5: Prepare copy
custom_path = "/home/omoured/Desktop/product-match/data/"
os.makedirs(custom_path, exist_ok=True)

items_to_copy = ["Apparel", "Footwear", "fashion.csv"]

print("\n🚚 Copying selected items:")
for item in items_to_copy:
    src = os.path.join(data_root, item)
    dst = os.path.join(custom_path, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  ✅ Copied folder: {item}")
    elif os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  ✅ Copied file: {item}")
    else:
        print(f"  ⚠️ Not found: {src}")

print(f"\n🎉 Done! Data copied to: {custom_path}")
