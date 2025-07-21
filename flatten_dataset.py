import os
import shutil

SOURCE_DIR = 'dataset/images'
DEST_DIR = 'dataset/flattened_dataset'

category_mapping = {
    "Plastic": [
        "plastic_water_bottles",
        "plastic_soda_bottles",
        "plastic_detergent_bottles",
        "plastic_shopping_bags",
        "plastic_trash_bags",
        "plastic_food_containers",
        "disposable_plastic_cutlery",
        "plastic_straws",
        "plastic_cup_lids"
    ],
    "Glass": [
        "glass_beverage_bottles",
        "glass_food_jars",
        "glass_cosmetic_containers"
    ],
    "Metal": [
        "aluminum_soda_cans",
        "aluminum_food_cans",
        "steel_food_cans",
        "aerosol_cans"
    ],
    "Paper": [
        "newspaper",
        "office_paper",
        "magazines",
        "cardboard_boxes",
        "cardboard_packaging",
        "paper_cups"
    ],
    "Organic": [
        "food_waste",
        "eggshells",
        "coffee_grounds",
        "tea_bags"
    ],
    "Textile": [
        "clothing",
        "shoes"
    ],
    "Styrofoam": [
        "styrofoam_cups",
        "styrofoam_food_containers"
    ]
}

os.makedirs(DEST_DIR, exist_ok=True)

for broad_category, folders in category_mapping.items():
    dest_folder = os.path.join(DEST_DIR, broad_category)
    os.makedirs(dest_folder, exist_ok=True)

    for subfolder in folders:
        subfolder_path = os.path.join(SOURCE_DIR, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping missing folder: {subfolder_path}")
            continue

        for variant in ['default', 'real_world']:
            variant_path = os.path.join(subfolder_path, variant)
            if not os.path.exists(variant_path):
                continue

            for img in os.listdir(variant_path):
                src = os.path.join(variant_path, img)
                dst = os.path.join(dest_folder, f"{subfolder}_{variant}_{img}")
                shutil.copyfile(src, dst)

print("Images grouped by material class and copied to:", DEST_DIR)
