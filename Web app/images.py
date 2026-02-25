import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('product_images.csv')

# Extract pixel columns (all 784 of them)
pixel_cols = [f'pixel_{i}' for i in range(784)]
pixels = df[pixel_cols].values  # shape: (10000, 784)

# --- View a grid of sample images ---
n_rows, n_cols = 5, 10  # 50 images total
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))

label_names = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

labels = df['label'].values

for idx, ax in enumerate(axes.flat):
    img = pixels[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray_r')
    ax.set_title(label_names[labels[idx]], fontsize=7)
    ax.axis('off')

plt.suptitle('Sample Product Images', fontsize=14)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150)
plt.show()
print("Saved to sample_images.png")

# --- View a single image in detail ---
def show_image(index):
    img = pixels[index].reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray_r')
    plt.title(f'Image #{index}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

show_image(0)  # change the index to view any image