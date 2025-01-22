import os
import pandas as pd
from PIL import Image, ImageDraw
import random

def create_sample_dataset(base_dir='data', num_images=10):
    """
    Creates a sample dataset with simple shapes and a proper CSV file
    """
    # Create directories if they don't exist
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Define shapes and their corresponding class labels
    shapes = {
        0: 'circle',
        1: 'square',
        2: 'triangle'
    }
    
    # Create images and collect filenames and labels
    data = []
    for i in range(num_images):
        # Choose random shape
        label = random.randint(0, 2)
        shape_name = shapes[label]
        filename = f'shape_{i}_{shape_name}.jpg'
        
        # Create image
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw shape
        if shape_name == 'circle':
            draw.ellipse([50, 50, 174, 174], fill='blue')
        elif shape_name == 'square':
            draw.rectangle([50, 50, 174, 174], fill='red')
        else:  # triangle
            draw.polygon([(112, 50), (50, 174), (174, 174)], fill='green')
        
        # Save image
        img_path = os.path.join(img_dir, filename)
        img.save(img_path)
        
        # Add to data
        data.append({
            'filename': filename,
            'class_label': label
        })
    
    # Create and save CSV file
    df = pd.DataFrame(data)
    csv_path = os.path.join(base_dir, 'labels.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Created {num_images} sample images in {img_dir}")
    print(f"Created labels file at {csv_path}")
    print("\nCSV content preview:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    create_sample_dataset()