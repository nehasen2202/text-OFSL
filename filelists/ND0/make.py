import glob
import json
import os
import pandas as pd

# Load the captions CSV
image_captions_file = "/home/xim/xu2202_main/IDEAL-main/filelists/ND0/ND0/splits/ND0_captions.csv"  # Update this path
captions_df = pd.read_csv(image_captions_file)

# Convert the captions dataframe to a dictionary for easy lookup using basenames
captions_dict = {
    os.path.basename(name): caption
    for name, caption in zip(captions_df["image_names"], captions_df["image_data"])
}

# Function to create the JSON file
def create_json(data_file, split_file, output_file, start_label=0):
    with open(split_file, 'r') as f:
        classes = f.readlines()
    
    result = {'label_names': [], 'image_names': [], 'image_labels': [], 'image_data': []}
    label = start_label
    
    for class_name in classes:
        class_name = class_name.strip()
        result['label_names'].append(class_name)
        files = glob.glob(os.path.join(data_file, class_name, "*"))
        
        for file_path in files:
            image_name = os.path.basename(file_path)  # Extract basename
            result['image_names'].append(file_path.replace("\\", "/"))
            result['image_labels'].append(label)
            # Add caption or default to "No Caption Available"
            caption = captions_dict.get(image_name, "No Caption Available")
            result['image_data'].append(caption)
        
        label += 1
    
    with open(output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

# Paths
pathname = os.getcwd()
pathname = pathname.replace('\\', '/')
data_dir = os.path.join(pathname, "/home/xim/xu2202_main/IDEAL-main/filelists/ND0/ND0/data")
splits_dir = os.path.join(pathname, "/home/xim/xu2202_main/IDEAL-main/filelists/ND0/ND0/splits")

# Create JSON files
create_json(data_dir, os.path.join(splits_dir, "train.txt"), "base.json", start_label=0)
create_json(data_dir, os.path.join(splits_dir, "val.txt"), "val.json", start_label=64)
create_json(data_dir, os.path.join(splits_dir, "test.txt"), "novel.json", start_label=80)

