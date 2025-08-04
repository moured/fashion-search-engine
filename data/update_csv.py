import os
import pandas as pd

# Root path where all category folders exist
base_path = "/home/omoured/Desktop/product-match/data"

# Load original CSV
csv_path = os.path.join(base_path, "fashion.csv")
df = pd.read_csv(csv_path)

# Construct sub-paths to search for image files
search_dirs = [
    "Apparel/Boys/Images/images_with_product_ids",
    "Apparel/Girls/Images/images_with_product_ids",
    "Footwear/Men/Images/images_with_product_ids",
    "Footwear/Women/Images/images_with_product_ids",
]

# Function to find absolute path of each image file
def find_image_path(image_filename):
    for subdir in search_dirs:
        full_path = os.path.join(base_path, subdir, image_filename)
        if os.path.isfile(full_path):
            return full_path
    return None  # Image not found

# Apply image path lookup
df["image_path"] = df["Image"].apply(find_image_path)

# Drop rows without a valid image
df = df.dropna(subset=["image_path"])

# Save new CSV
output_csv = os.path.join(base_path, "fashion_with_paths.csv")
df.to_csv(output_csv, index=False)
print(f"✅ CSV with image paths saved to: {output_csv}")
