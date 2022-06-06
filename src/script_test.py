#  test
import asyncio
import pandas as pd

from data_helpers import load_song_data


PATH_R = "./MillionSongSubset"

# def filter_hotness(df_, threshold):
#     df = df_.copy()
#     assert threshold > 0 and threshold < 1, "Threshold must be between 0 and 1"
#     df = df.dropna(subset=["song_hotttnesss"])
#     df["song_hotttnesss"] = df["song_hotttnesss"].astype(float)
#     return df[df.song_hotttnesss > threshold]

PATH_HDF5 = "./sample_data/TRAXLZU12903D05F94.h5"

from multiprocessing import Process

merged_df = None


def run_load_song_async(letter, half=None):
    print(f"Starting to load songs from {letter}{half if half else ''}")
    df = load_song_data(
        dataset_root_dir=PATH_R,
        sample_hdf5_file_path=PATH_HDF5,
        letter=letter,
        half=half,
    )
    print(f"Done loading songs from {letter}{half if half else ''}")

    df.to_json(
        f"df{letter}{half if half else ''}.json.gzip",
        orient="records",
        # compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    return df


letters = ("A", "B")
processes = [Process(target=run_load_song_async, args=(letter,)) for letter in letters]

for p in processes:
    p.start()

for p in processes:
    p.join()
