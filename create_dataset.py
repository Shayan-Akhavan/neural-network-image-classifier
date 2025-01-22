import os
from PIL import Image, ImageDraw
import pandas as pd
import random
import math

def create_training_data(num_samples=30):  # Increased from 9 to 30 samples
    # Create directories
    os.makedirs('data/images', exist_ok=True)
    
    data = []
    shapes = ['circle', 'square', 'triangle']
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Added more colors
    
    print(f"Generating {num_samples} images...")
    
    for i in range(num_samples):
        # Randomly select shape and color
        shape = shapes[i % 3]  # Ensure equal distribution of shapes
        color = random.choice(colors)
        
        # Create image with white background
        img = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(img)
        
        # Randomize position and size
        center_x = random.randint(82, 142)  # Vary center position
        center_y = random.randint(82, 142)
        size = random.randint(60, 100)  # Vary size
        
        # Draw shape with variations
        if shape == 'circle':
            draw.ellipse([
                center_x - size//2, 
                center_y - size//2, 
                center_x + size//2, 
                center_y + size//2
            ], fill=color)
            
        elif shape == 'square':
            # Add rotation to square
            angle = random.randint(0, 45)
            img_tmp = Image.new('RGBA', (224, 224), (0, 0, 0, 0))
            draw_tmp = ImageDraw.Draw(img_tmp)
            
            # Draw rotated square
            points = [
                (-size//2, -size//2),
                (size//2, -size//2),
                (size//2, size//2),
                (-size//2, size//2)
            ]
            
            # Rotate points
            rotated_points = []
            for x, y in points:
                # Rotate point
                angle_rad = math.radians(angle)
                rot_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                rot_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                # Move to center position
                rotated_points.append((rot_x + center_x, rot_y + center_y))
            
            draw_tmp.polygon(rotated_points, fill=color)
            img.paste(img_tmp, (0, 0), img_tmp)
            
        else:  # triangle
            # Add random variation to triangle points
            variance = size // 4
            points = [
                (center_x, center_y - size//2),  # top
                (center_x - size//2, center_y + size//2),  # bottom left
                (center_x + size//2, center_y + size//2)   # bottom right
            ]
            
            # Add some random variation to points
            points = [(x + random.randint(-variance, variance), 
                      y + random.randint(-variance, variance)) 
                     for x, y in points]
            
            draw.polygon(points, fill=color)
        
        # Save the image
        filename = f'shape_{i+1}_{shape}.png'
        img.save(f'data/images/{filename}')
        
        # Add to dataset
        data.append({
            'filename': filename,
            'class_label': shapes.index(shape)
        })
    
    # Create CSV file
    df = pd.DataFrame(data)
    df.to_csv('data/labels.csv', index=False)
    
    print("\nCreated dataset with variations:")
    print(f"- Number of images: {num_samples}")
    print(f"- Shapes: {', '.join(shapes)}")
    print(f"- Colors: {', '.join(colors)}")
    print(f"- Various sizes, positions, and rotations")
    print("\nDataset contents:")
    print(df)

if __name__ == '__main__':
    create_training_data(num_samples=30)  # Generate 30 samples