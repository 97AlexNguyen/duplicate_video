import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_dct_and_phash_info(img_path, hash_size=8):
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to load image at", img_path)
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to (hash_size+1)x(hash_size+1)
    resized = cv2.resize(gray, (hash_size+1, hash_size+1), interpolation=cv2.INTER_AREA)

    # Compute the DCT
    dct_input = np.float32(resized)
    dct_all = cv2.dct(dct_input)

    # Extract the low frequency DCT block (8x8)
    dct_lowfreq = dct_all[:hash_size, :hash_size]

    # Compute the median value
    med_val = np.median(dct_lowfreq)
    
    # Threshold the low-frequency DCT to create a binary matrix
    bin_mat_bool = (dct_lowfreq > med_val)
    bin_mat = bin_mat_bool.astype(int)

    # Flatten the binary matrix to compute the 64-bit pHash value
    bits = bin_mat_bool.flatten()
    phash_val = 0
    for bit in bits:
        phash_val = (phash_val << 1) | int(bit)

    # Print pHash information
    print("=== pHash ===")
    print("- DCT low-frequency matrix (8x8) before threshold:")
    print(np.round(dct_lowfreq, 2))
    print(f"- Median of 8x8 block: {med_val:.3f}")
    print("- Binary matrix (8x8) after threshold (0/1):")
    print(bin_mat)
    print("- Flattened binary bits:")
    print(bin_mat.flatten())
    print(f"- pHash (64-bit integer): {phash_val} (decimal)")
    print(f"- pHash (hexadecimal)   : 0x{phash_val:016x}")

    # Plotting
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: DCT visualization in log scale
    plt.subplot(1, 2, 1)
    plt.title(f"DCT (Log Scale) [{hash_size+1}x{hash_size+1}]")
    dct_vis = np.log1p(np.abs(dct_all))
    im1 = plt.imshow(dct_vis, cmap='gray')
    plt.colorbar(im1, label='log(|DCT|)')
    
    # Subplot 2: Binary matrix visualization
    plt.subplot(1, 2, 2)
    plt.title(f"Binary Matrix [{hash_size}x{hash_size}]")
    # Multiply by 255 for better visualization (0 -> 0, 1 -> 255)
    bin_mat_visual = bin_mat * 255 
    im2 = plt.imshow(bin_mat_visual, cmap='gray')
    plt.colorbar(im2, label='0-255')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = r"X:\Tool\duplicate_video\a.jpg"
    show_dct_and_phash_info(image_path, hash_size=8)
