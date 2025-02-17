import os
import shutil
import subprocess
import torch
import cv2
import numpy as np
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# --- Configuration ---
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


MAX_SIZE_BYTES = 500 * 1024 * 1024 
THRESHOLD = 10.0                 # Average DTW threshold for duplicate detection
FRAME_INTERVAL = 1               
folderA = r"X:\3d\video"
folderB = r"X:\3d\dup"
os.makedirs(folderB, exist_ok=True)

# Caching and optimization configuration
USE_CACHE = True


script_dir = Path(__file__).parent 
CACHE_DIR = script_dir / "pHash_cache"  

print("CACHE_DIR:", CACHE_DIR)

os.makedirs(CACHE_DIR, exist_ok=True)

USE_DTW_BAND = True
DTW_BAND = 2    # Allowable frame offset in DTW

USE_APPROXIMATE_SEARCH = True
APPROX_THRESHOLD = 20  # Global pHash Hamming threshold to filter out non-candidates

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

#############################################
# Hamming distance between two 64-bit integers
#############################################
def hamming_distance(p1, p2):
    """
    Compute the Hamming distance between two 64-bit integers.
    """
    x = p1 ^ p2
    return bin(x).count('1')

#############################################
# 1. Compute pHash for a grayscale image
#############################################
def phash(image, hash_size=8):
    """
    Compute the pHash (perceptual hash) of a grayscale image.
    Steps:
      1. Resize the image to (hash_size+1) x (hash_size+1)
      2. Convert to float32 and compute DCT.
      3. Take the top-left (hash_size x hash_size) block.
      4. Threshold using the median to form a binary matrix.
      5. Flatten to a 64-bit integer.
    """
    img = cv2.resize(image, (hash_size + 1, hash_size + 1), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    dct = cv2.dct(img)
    dct_lowfreq = dct[0:hash_size, 0:hash_size]
    med = np.median(dct_lowfreq)
    diff = dct_lowfreq > med
    phash_val = 0
    for bit in diff.flatten():
        phash_val = (phash_val << 1) | int(bit)
    return phash_val

#############################################
# 2. Get video info using ffprobe
#############################################
def ffprobe_video_info(video_path):
    import json
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-print_format", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"ffprobe error for {video_path}: {result.stderr}")
        return None
    info = json.loads(result.stdout)
    w = info["streams"][0]["width"]
    h = info["streams"][0]["height"]
    fps_str = info["streams"][0]["r_frame_rate"]
    num, den = fps_str.split('/')
    fps = float(num) / float(den) if float(den) != 0 else 0
    return (w, h, fps)

#############################################
# 3. Compute video pHash sequence using GPU (ffmpeg)
#############################################
def compute_video_pHash_sequence_gpu(video_path, frame_interval=1):
    """
    Decode video using GPU (ffmpeg -hwaccel cuda) to extract raw frames (bgr24).
    Compute a pHash for every 'frame_interval' seconds.
    Uses caching if enabled.
    Returns a list of pHash values.
    """
    cache_file = os.path.join(CACHE_DIR, os.path.basename(video_path) + ".pkl")
    if USE_CACHE and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                seq = pickle.load(f)
            logger.info(f"Loaded cache for {video_path}")
            return seq
        except Exception as e:
            logger.error(f"Error loading cache for {video_path}: {e}")

    info = ffprobe_video_info(video_path)
    if not info:
        logger.error(f"Could not retrieve video info for {video_path}")
        return None
    w, h, fps = info
    if w <= 0 or h <= 0 or fps <= 0:
        logger.error(f"Invalid video dimensions or FPS for {video_path}")
        return None

    skip_frames = int(frame_interval * fps) if fps > 0 else 1
    cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    frame_count = 0
    phash_seq = []
    while True:
        raw_frame = pipe.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            break
        if frame_count % skip_frames == 0:
            frame_array = np.frombuffer(raw_frame, np.uint8).reshape((h, w, 3))
            gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
            phash_seq.append(phash(gray))
        frame_count += 1
    pipe.stdout.close()
    pipe.wait()
    if USE_CACHE:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(phash_seq, f)
            logger.info(f"Saved cache for {video_path}")
        except Exception as e:
            logger.error(f"Error saving cache for {video_path}: {e}")
    return phash_seq

#############################################
# 4. Convert unsigned 64-bit to signed 64-bit
#############################################
def convert_unsigned64_to_signed(val):
    if val > 0x7FFFFFFFFFFFFFFF:
        val = val - 0x10000000000000000
    return val

#############################################
# 5. Vectorized popcount using torch
#############################################
def popcount_tensor(x):
    """
    Vectorized popcount for each element in tensor x (dtype=torch.long).
    This loops over 64 bits, but the loop is executed on GPU.
    """
    count = torch.zeros_like(x, dtype=torch.long)
    for i in range(64):
        count += ((x >> i) & 1)
    return count

#############################################
# 6. DTW distance on GPU with optional Sakoe-Chiba band
#############################################
def dtw_distance_gpu(seqA, seqB):
    """
    Compute the DTW distance between two pHash sequences (lists of 64-bit integers) using GPU.
    Local cost = Hamming distance between two pHash values.
    If USE_DTW_BAND is enabled, only compute dp[i,j] for |i - j| <= DTW_BAND.
    """
    if not seqA or not seqB:
        return 999999999.0
    seqA_64 = [convert_unsigned64_to_signed(x) for x in seqA]
    seqB_64 = [convert_unsigned64_to_signed(x) for x in seqB]
    A = torch.tensor(seqA_64, dtype=torch.long, device='cuda')
    B = torch.tensor(seqB_64, dtype=torch.long, device='cuda')
    n, m = A.shape[0], B.shape[0]
    X = A.unsqueeze(1) ^ B.unsqueeze(0)
    dist = popcount_tensor(X).to(torch.float32)
    dp = torch.full((n + 1, m + 1), 1e9, dtype=torch.float32, device='cuda')
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = 1
        j_end = m + 1
        if USE_DTW_BAND:
            j_start = max(1, i - DTW_BAND)
            j_end = min(m, i + DTW_BAND) + 1
        for j in range(j_start, j_end):
            cost_ij = dist[i - 1, j - 1]
            dp[i, j] = cost_ij + torch.min(
                dp[i - 1, j],
                torch.min(dp[i, j - 1], dp[i - 1, j - 1])
            )
    return float(dp[n, m])

#############################################
# 7. Global pHash for approximate matching
#############################################
def global_pHash(seq):
    """
    Compute a global pHash from a sequence of pHash values by majority vote on each bit.
    Returns a representative 64-bit integer.
    """
    if not seq:
        return 0
    bits = np.zeros((len(seq), 64), dtype=int)
    for i, ph in enumerate(seq):
        bin_str = f"{ph:064b}"
        bits[i, :] = np.array(list(bin_str), dtype=int)
    majority = (bits.sum(axis=0) >= (len(seq)/2)).astype(int)
    global_hash = 0
    for bit in majority:
        global_hash = (global_hash << 1) | bit
    return global_hash

#############################################
# 8. Get list of video files from folder
#############################################
def get_video_files(folder):
    exts = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.m4v', '.ts')
    all_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                if size <= MAX_SIZE_BYTES:
                    all_files.append(path)
    return all_files

#############################################
# 9. Main: Use multithreading and approximate search
#############################################
def main():
    files = get_video_files(folderA)
    logger.info(f"Total number of qualified videos (<= {MAX_SIZE_BYTES/1024/1024:.0f} MB) in {folderA} = {len(files)}")
    if not files:
        logger.error("No files to process. Exiting.")
        return

    # Use ThreadPoolExecutor to compute pHash sequences in parallel
    pHash_dict = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_vid = {executor.submit(compute_video_pHash_sequence_gpu, vid, FRAME_INTERVAL): vid for vid in files}
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                seq = future.result()
                logger.info(f"Computed pHash sequence for {vid}")
            except Exception as exc:
                logger.error(f"{vid} generated an exception: {exc}")
                seq = []
            pHash_dict[vid] = seq if seq else []

    # Compute global pHash for approximate matching if enabled
    global_dict = {}
    if USE_APPROXIMATE_SEARCH:
        for vid, seq in pHash_dict.items():
            global_dict[vid] = global_pHash(seq)
            logger.info(f"Computed global pHash for {vid}")

    already_moved = set()
    for i in range(len(files)):
        fA = files[i]
        if fA in already_moved:
            continue
        seqA = pHash_dict[fA]
        globalA = global_dict.get(fA, None) if USE_APPROXIMATE_SEARCH else None

        for j in range(i + 1, len(files)):
            fB = files[j]
            if fB in already_moved:
                continue
            seqB = pHash_dict[fB]
            if USE_APPROXIMATE_SEARCH:
                globalB = global_dict.get(fB, None)
                approx_dist = hamming_distance(globalA, globalB)
                if approx_dist > APPROX_THRESHOLD:
                    continue

            dtw_dist = dtw_distance_gpu(seqA, seqB)
            length_max = max(len(seqA), len(seqB))
            if length_max == 0:
                continue
            avg_dist = dtw_dist / length_max

            if avg_dist < THRESHOLD:
                logger.info(f"Duplicate detected:\n - {fA}\n - {fB}")
                logger.info(f"DTW distance = {dtw_dist:.2f}, average = {avg_dist:.2f} < {THRESHOLD}")
                new_path = os.path.join(folderB, os.path.basename(fB))
                logger.info(f"Moving {fB} => {new_path}")
                try:
                    shutil.move(fB, new_path)
                    already_moved.add(fB)
                except Exception as e:
                    logger.error(f"Error moving {fB}: {e}")

    logger.info("Comparison completed.")
    logger.info(f"Moved {len(already_moved)} duplicate files to {folderB}.")

if __name__ == "__main__":
    main()
