"""
extraction_helpers.py

Yuri Avanci Laguardia e Henrique Laguardia Heringer Faria

Módulo destinado a implementar a operação de extração de músicas da base completa
do "Million Song Dataset" (http://millionsongdataset.com/)
"""
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from . import hdf5_getters

# Permite mostrar uma barra de progresso durante a operação
tqdm.pandas()


def get_total_features(path_hdf5: str):
    """Extrai o nome das features dos metadados e agrega em uma lista

    Para que não precisemos inseri-los manualmente

    Args:
        path_hdf5 (str): caminho para um arquivo de exemplo em formato hdf5 que será
        usado como modelo para extrair os nomes da colunas/features
        - Ex: "../sample_data/TRAXLZU12903D05F94.h5"

    Returns:
        list[str]: lista com os nomes de todas as colunas/features.
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
    """Extrai dados de músicas de arquivos HDF5 para um pandas dataframe

    Args:
        - dataset_root_dir (str): Caminho para a raiz do diretório que contém a base
        - sample_hdf5_file_path (str): Caminho para um arquivo HDF5 de exemplo
        - letter (str, optional): o nome do diretório onde queremos realizar a busca
            (na base original, todos os diretório são denominados "A", "B", "C" etc)
            - Por padrão, "*" (que significa todos os diretórios e subdiretórios).
        - half (int, optional): Qual metade do diretório queremos explorar.
            Definimos a 1ª metade como sendo as letras de A até M, 2ª metade sendo N a Z
            - Por padrão, None (significa: procurar diretório inteiro, em vez de metade)
        - max_songs (int, optional): O número máximo de músicas que queremos extrair.
            - Por padrão, None (que significa TODAS as músicas de todos diretórios)

    Returns:
        pd.DataFrame: um dataframe contendo informações sobre as músicas extraídas
    """
    # Definição do que é 1ª metade e o que é 2ª metade.
    regex_half_map = {1: "[A-M]", 2: "[N-Z]"}

    # Por padrão, busca-se em todos os diretórios (todas as letras)
    if half is None:
        regex = "[A-Z]"
    else:
        assert half in [1, 2], "'half' deve ser 1 ou 2"
        regex = regex_half_map[half]

    path = dataset_root_dir
    categories = get_total_features(sample_hdf5_file_path)
    data = []
    file_paths = glob.glob(f"{path}/{letter}/{regex}/*/*.h5")

    # tdqm aqui é utilizado para mostrar barra de progresso durante a operação
    count = 0
    for file_path in tqdm(file_paths, total=len(file_paths)):
        h5file = hdf5_getters.open_h5_file_read(file_path)
        datapoint = {}
        for cat in categories:
            attr = getattr(hdf5_getters, "get_" + cat)(h5file)
            datapoint[cat] = attr
        h5file.close()

        # Queremos tracks que necessariamente contenham as features abaixo não nulas
        # A depender do tipo de dado, temos diferentes definições para que é "nulo"
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
