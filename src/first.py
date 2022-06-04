#  test
# import pandas as pd

from lib2to3.pgen2.pgen import DFAState
from hdf5_getters import open_h5_file_read
from data_helpers import get_total_features, load_song_data


PATH_R = "./millionsongsubset"

regex_half = {1: "[A-M]", 2: "[N-Z]"}

colsPart = {}
colsPart[1] = [
    "artist_latitude",
    "artist_longitude",
    "artist_terms",
    "artist_terms_weight",
    "year",
]
colsPart[2] = [
    "artist_hotttnesss",
    "artist_name",
    "artist_terms",
    "artist_terms_weight",
    "loudness",
    "song_hotttnesss",
    "tempo",
    "year",
    "title",
    "X_mean",
    "X_std",
    "X_skew",
    "X_kurtosis",
    "X_median",
]


# def filter_hotness(df_, threshold):
#     df = df_.copy()
#     assert threshold > 0 and threshold < 1, "Threshold must be between 0 and 1"
#     df = df.dropna(subset=["song_hotttnesss"])
#     df["song_hotttnesss"] = df["song_hotttnesss"].astype(float)
#     return df[df.song_hotttnesss > threshold]
PATH_HDF5 = "./sample_data/TRAXLZU12903D05F94.h5"
df = load_song_data(
    letter="A", half=1, path_r=PATH_R, path_hdf5=PATH_HDF5, max_songs=50
)
# print(df.info(verbose=True))
print(df.info(verbose=True))
