import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from joblib import load
from utils import standardize
from typing import List, Tuple

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop: List[str] = \
        ['ch1_peak', 'ch1_width', 'ch1_area', 'ch2_peak', 'ch2_width',
         'ch2_area', 'ch3_peak', 'ch3_width', 'ch3_area', 'image_id',
         'src_x', 'src_y', 'src_image', 'camera', 'cal_image', 'elapsed_time',
         'sphere_count', 'raw_sphere_volume', 'raw_sphere_complement',
         'raw_sphere_unknown', 'cal_const', 'fringe_size',
         'intensity_calimage', 'timestamp', 'image_h', 'image_w', 'image_x',
         'image_y', 'collage_file', 'feret_max_angle', 'feret_max_angle', 'id',
         'month', 'season', 'width', 'length', 'raw_legendre_minor',
         'raw_legendre_major', 'raw_feret_min', 'abd_area']

    for col in cols_to_drop:
        if col in df:
            df = df.drop(col, axis=1)

    return df

def prepare_training_data(
    df: pd.DataFrame,
    images: np.array,
    min_samples: int,
    path_to_store) -> Tuple[np.array, np.array, np.array,
                               np.array, np.array, np.array,
                               np.array, np.array, np.array]:
   
    targets = df['_target'].value_counts().keys().tolist()
    counts = df['_target'].value_counts().tolist()
    for (target, count) in zip(targets, counts):
        if count < min_samples:
            idx = list(np.where(df["_target"] == target)[0])
            df = df[df._target != target]
            images = np.delete(images, idx, axis=0)

    # Split into testing,validation and training data
    trainAttrX, testAttrX, trainImagesX, testImagesX = \
        train_test_split(df, images, test_size=0.28, random_state=42,
                         stratify=df['_target'])

    testAttrX, valAttrX, testImagesX, valImagesX = \
        train_test_split(testAttrX, testImagesX, test_size=0.5,
                         random_state=42,  stratify=testAttrX['_target'])

    # Get the train, val and test targets, then drop the _target column
    y_train = trainAttrX['_target']
    y_val = valAttrX['_target']
    y_test = testAttrX['_target']
    del trainAttrX['_target']
    del valAttrX['_target']
    del testAttrX['_target']

    # The labels need to be in an array of one-hot vectors that
    # neural networks will use
    y_train = np_utils.to_categorical(y_train, num_classes=len(targets))
    y_val = np_utils.to_categorical(y_val, num_classes=len(targets))
    y_test = np_utils.to_categorical(y_test, num_classes=len(targets))

    # Convert everything to pure numpy arrays (images already are)
    trainAttrX = trainAttrX.to_numpy()
    valAttrX = valAttrX.to_numpy()
    testAttrX = testAttrX.to_numpy()

    # standardize the data (0 mean, 1 std)
    '''path to store standard scaler binary'''
    trainAttrX, valAttrX, testAttrX = standardize(
        trainAttrX, valAttrX, testAttrX, path_to_store)

    return (trainAttrX, valAttrX, testAttrX, trainImagesX,
            valImagesX, testImagesX, y_train, y_val, y_test)


def process_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering

    Arguments:
        df {pd.DataFrame} -- [description]

    Returns:
        pd.DataFrame -- [description]
    """
    seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
    month_to_season = dict(zip(range(1, 13), seasons))
    # # get season sample was collected from timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df.timestamp.dt.month - 1
    df['season'] = df.timestamp.dt.month.map(month_to_season)

    df['season_sin'] = np.sin((df.season-1)*(2.*np.pi/4))
    df['season_cos'] = np.cos((df.season-1)*(2.*np.pi/4))

    # handle missing values
    df = df.apply(pd.to_numeric, errors='coerce')
    df['raw_legendre_major'].replace(0, np.nan, inplace=True)
    df['raw_legendre_minor'].replace(0, np.nan, inplace=True)

    df['raw_legendre_major'].fillna(
        df['raw_legendre_major'].mean(), inplace=True)
    df['raw_legendre_minor'].fillna(
        df['raw_legendre_minor'].mean(), inplace=True)

    # log transform
    df['width_log+1'] = (df['width'].astype(float)+1).transform(np.log)
    df['length_log+1'] = (df['length'].astype(float)+1).transform(np.log)
    df['raw_legendre_minor_log+1'] = (
        df['raw_legendre_minor'].astype(float)+1).transform(np.log)
    df['raw_legendre_major_log+1'] = (
        df['raw_legendre_major'].astype(float)+1).transform(np.log)
    df['abd_area_log+1'] = (df['abd_area'].astype(float)+1).transform(np.log)
    df['raw_feret_min_log+1'] = (df['raw_feret_min'].astype(float) +
                                 1).transform(np.log)
    # ratio
    df['wh_ratio'] = df['width'].astype(float) / df['length'].astype(float)

    # based on \cite{embleton2003automated}
    df['perimeter_area_ratio'] = df['perimeter'].astype(
        float) / df['abd_area'].astype(float)
    df['area_length_ratio'] = df['abd_area'].astype(
        float) / df['length'].astype(float)
    df['length_maxferret_ratio'] = df['length'].astype(
        float) / df['raw_feret_max'].astype(float)
    df['feret_ratio'] = df['raw_feret_min'].astype(
        float) / df['raw_feret_max'].astype(float)
    df['perimeter_sqrt_area_ratio'] = df['perimeter'].astype(
        float) / np.sqrt(df['abd_area'].astype(float))
    df['perimeter_length_ratio'] = df['perimeter'].astype(
        float) / df['length'].astype(float)

    return df
