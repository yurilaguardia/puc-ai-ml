import glob
import pandas as pd
import numpy as np
import hdf5_getters
from tqdm.notebook import tqdm

tqdm.pandas()

regex_half = {1: "[A-M]", 2: "[N-Z]"}


def get_total_features(path_hdf5):

    # Load the features name from the metadata into a list so that we don't
    # have to insert them manually

    h5_summary = hdf5_getters.open_h5_file_read(path_hdf5)

    metadata = h5_summary.get_node("/metadata/songs/").colnames
    metadata.remove("genre")
    metadata.remove("analyzer_version")
    metadata = [w.replace("idx_", "") for w in metadata]

    analysis = h5_summary.get_node("/analysis/songs/").colnames
    analysis = [w.replace("idx_", "") for w in analysis]

    musicbrainz = h5_summary.get_node("/musicbrainz/songs/").colnames
    musicbrainz = [w.replace("idx_", "") for w in musicbrainz]

    total_features = np.array(metadata + analysis + musicbrainz).ravel()

    total_features = np.append(
        total_features,
        ["artist_terms_freq", "artist_terms_weight", "artist_mbtags_count"],
    )

    total_features = np.sort(total_features)

    return total_features


def load_song_data(letter, half, path_r, path_hdf5, max_songs=None):
    assert half in [1, 2], "half must be one or two"
    path = path_r
    categories = get_total_features(path_hdf5)
    data = []
    file_paths = glob.glob(path + "/" + letter + "/" + regex_half[half] + "/*/*.h5")
    if max_songs:
        file_paths = file_paths[: max_songs - 1]

    for i, file_path in tqdm(enumerate(file_paths), total=len(file_paths)):
        # for file_path in file_paths:
        h5file = hdf5_getters.open_h5_file_read(file_path)
        datapoint = {}
        for cat in categories:
            attr = getattr(hdf5_getters, "get_" + cat)(h5file)
            if isinstance(attr, np.ndarray) and len(attr) == 0:
                attr = np.nan
            datapoint[cat] = attr
        h5file.close()

        # if not pd.isna(datapoint.get("song_hotttnesss")):
        #     data.append(datapoint)
        data.append(datapoint)

    df = pd.DataFrame(data)
    return df
