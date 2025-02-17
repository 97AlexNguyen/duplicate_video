import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################
# 1) Compute 8x8 pHash for a grayscale image
##################################################
def compute_phash_8x8(gray_image):
    hash_size = 8
    # Resize the image to (hash_size+1, hash_size+1) => 9x9
    resized = cv2.resize(gray_image, (hash_size+1, hash_size+1), interpolation=cv2.INTER_AREA)
    
    # Compute the DCT
    dct_input = np.float32(resized)
    dct_all = cv2.dct(dct_input)
    
    # Extract the low frequency region (8x8)
    dct_lowfreq = dct_all[:hash_size, :hash_size]
    
    # Compare with the median value
    med_val = np.median(dct_lowfreq)
    bin_mat_bool = (dct_lowfreq > med_val)
    bin_mat = bin_mat_bool.astype(int)
    
    # Flatten the binary matrix to encode as a 64-bit pHash
    bits = bin_mat_bool.flatten()
    phash_val = 0
    for bit in bits:
        phash_val = (phash_val << 1) | int(bit)
    
    return phash_val, dct_lowfreq, bin_mat

##################################################
# 2) Compute Hamming distance between two 64-bit pHash values
##################################################
def hamming_distance(p1, p2):
    x = p1 ^ p2
    return bin(x).count('1')

##################################################
# 3) Compute a sequence of pHash for image blocks
##################################################
def compute_phash_sequence_blocks(image, n_blocks=4):
    """
    Divide the image into n_blocks vertical segments,
    compute pHash for each block to form a sequence.
    """
    h, w = image.shape[:2]
    block_height = h // n_blocks
    phash_seq = []
    for i in range(n_blocks):
        y_start = i * block_height
        y_end = (i+1)*block_height if i < n_blocks-1 else h
        block = image[y_start:y_end, :w]
        gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        pval, _, _ = compute_phash_8x8(gray_block)
        phash_seq.append(pval)
    return phash_seq

##################################################
# 4) DTW to compare two pHash sequences
##################################################
def dtw_phash_sequence(seqA, seqB):
    n = len(seqA)
    m = len(seqB)

    cost = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            cost[i, j] = hamming_distance(seqA[i], seqB[j])
    
    # dp: a (n+1)x(m+1) matrix for dynamic programming
    dp = np.ones((n+1, m+1), dtype=np.float32) * 1e9
    dp[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            c = cost[i-1, j-1]
            dp[i, j] = c + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (dp[i-1, j], (i-1, j)),
            (dp[i, j-1], (i, j-1)),
            (dp[i-1, j-1], (i-1, j-1))
        ]
        _, best_coord = min(candidates, key=lambda x: x[0])
        if best_coord == (i-1, j):
            i -= 1
        elif best_coord == (i, j-1):
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    
    return cost, dp, path


def main():
    imgA_path = r"X:\Tool\duplicate_video\a.jpg"
    imgB_path = r"X:\Tool\duplicate_video\b.jpg"
    

    imgA = cv2.imread(imgA_path)
    imgB = cv2.imread(imgB_path)
    if imgA is None or imgB is None:
        print("Unable to read image, check the path!")
        return
    

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    pA, dctA, binA = compute_phash_8x8(grayA)
    
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    pB, dctB, binB = compute_phash_8x8(grayB)
    

    hamdist_whole = hamming_distance(pA, pB)

    seqA = compute_phash_sequence_blocks(imgA, n_blocks=4)
    seqB = compute_phash_sequence_blocks(imgB, n_blocks=4)
    
    cost, dp, path = dtw_phash_sequence(seqA, seqB)
    dtw_distance = dp[len(seqA), len(seqB)]
    dtw_avg = dtw_distance / max(len(seqA), len(seqB))

    info_text = (
        f"Hamming distance (full image) = {hamdist_whole}\n"
        f"pHash A = {pA}\n"
        f"pHash B = {pB}\n\n"
        f"DTW distance (4 blocks) = {dtw_distance:.2f}\n"
        f"Average = {dtw_avg:.2f}\n"
        f"Number of steps (path length) = {len(path)}"
    )

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Image A DCT (log scale, 8x8)")
    dct_visA = np.log1p(np.abs(dctA))
    imA = ax1.imshow(dct_visA, cmap='gray')
    fig.colorbar(imA, ax=ax1, label='log(|DCT|)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Image B DCT (log scale, 8x8)")
    dct_visB = np.log1p(np.abs(dctB))
    imB = ax2.imshow(dct_visB, cmap='gray')
    fig.colorbar(imB, ax=ax2, label='log(|DCT|)')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Cost Matrix (Hamming Distance) - DTW")
    imC = ax3.imshow(cost, cmap='viridis', origin='upper')
    fig.colorbar(imC, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xlabel("Block B")
    ax3.set_ylabel("Block A")
    for (i, j) in path:
        ax3.plot(j, i, 'rs', markersize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title(f"DP Matrix - DTW\n(Total = {dtw_distance:.2f}, Average = {dtw_avg:.2f})")
    dp_show = dp[1:len(seqA)+1, 1:len(seqB)+1]
    imD = ax4.imshow(dp_show, cmap='magma', origin='upper')
    fig.colorbar(imD, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_xlabel("Block B")
    ax4.set_ylabel("Block A")
    for (i, j) in path:
        ax4.plot(j, i, 'ws', markersize=6)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("Binary Matrix (pHash) - Image A")
    imBinA = ax5.imshow(binA, cmap='binary', interpolation='nearest')
    fig.colorbar(imBinA, ax=ax5)
    ax5.set_xlabel("Pixel")
    ax5.set_ylabel("Pixel")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("Binary Matrix (pHash) - Image B")
    imBinB = ax6.imshow(binB, cmap='binary', interpolation='nearest')
    fig.colorbar(imBinB, ax=ax6)
    ax6.set_xlabel("Pixel")
    ax6.set_ylabel("Pixel")
    
    ax_line = fig.add_subplot(gs[3, :])
    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]
    steps = list(range(len(path)))
    ax_line.plot(steps, path_i, '-o', label='Index A')
    ax_line.plot(steps, path_j, '-s', label='Index B')
    ax_line.set_xlabel("Step (order in path)")
    ax_line.set_ylabel("Block Index")
    ax_line.set_title(f"Optimal DTW Path (Steps = {len(path)})")
    ax_line.legend()
    fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

if __name__ == "__main__":
    main()
