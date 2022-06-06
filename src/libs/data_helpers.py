import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from . import hdf5_getters

# Allow progress bar
tqdm.pandas()


def get_total_features(path_hdf5: str):
    """Load the features name from the metadata into a list

    So that we don't have to insert them manually
    """
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


def load_song_data(
    dataset_root_dir: str,
    sample_hdf5_file_path: str,
    letter: str = "*",
    half: int = None,
    max_songs: int = None,
) -> pd.DataFrame:
    """Extract song data from HDF5 files into a dataframe

    Args:
        - dataset_root_dir (str): path to the root dir of the dataset
        - sample_hdf5_file_path (str): path to a sample HDF5 file
        - letter (str, optional): a letter representing the name of the dir to be
            searched. Defaults to "*" (which means ALL dirs and subdirs).
        - half (int, optional): Which half of the dir we want to explore.
            Defaults to None (which means: sarch the entire dir and subdirs).
        - max_songs (int, optional): The maximum number of song we want to extract.
            Defaults to None (which means ALL SONGS inside the dir and subdirs)

    Returns:
        pd.DataFrame: a dataframe contained information about the extracted songs
    """
    regex_half_map = {1: "[A-M]", 2: "[N-Z]"}

    if half is None:
        regex = "[A-Z]"
    else:
        assert half in [1, 2], "half must be one or two"
        regex = regex_half_map[half]

    path = dataset_root_dir
    categories = get_total_features(sample_hdf5_file_path)
    data = []
    file_paths = glob.glob(f"{path}/{letter}/{regex}/*/*.h5")

    for file_path in tqdm(file_paths, total=len(file_paths)):
        # for file_path in file_paths:
        h5file = hdf5_getters.open_h5_file_read(file_path)
        datapoint = {}
        for cat in categories:
            attr = getattr(hdf5_getters, "get_" + cat)(h5file)
            datapoint[cat] = attr
        h5file.close()

        # Only include tracks with song_hotttnesss, artist_latitude, artist_longitude,
        # year, artist_mbtags and artist_mbtags_count
        if all(
            [
                pd.notna(datapoint.get("song_hotttnesss")),
                pd.notna(datapoint.get("artist_hotttnesss")),
                pd.notna(datapoint.get("artist_familiarity")),
                pd.notna(datapoint.get("artist_latitude")),
                pd.notna(datapoint.get("artist_longitude")),
                datapoint.get("year") != 0,
                len(datapoint.get("artist_mbtags")) > 0,
                len(datapoint.get("artist_mbtags_count")) > 0,
            ]
        ):
            data.append(datapoint)
            if max_songs is not None and len(data) >= max_songs:
                break

    df = pd.DataFrame(data)
    return df
