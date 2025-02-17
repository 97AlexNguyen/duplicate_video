import pickle

cache_file_path = r"X:\Tool\duplicate_video\pHash_cache\1245.mp4.pkl"
with open(cache_file_path, "rb") as f:
    phash_sequence = pickle.load(f)
