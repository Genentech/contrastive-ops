import sys, os
sys.path.append("..")  # Adds higher directory to python modules path.

from constants import Column
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics, preprocessing
from scipy.stats import median_abs_deviation
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

def fit(df, metadata, column, average=None):
    df = df.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    non_null = df.iloc[:,0].notnull() & metadata[column].notnull()
    df = df[non_null].copy()
    metadata = metadata[non_null].copy()

    if column == Column.batch.value or column == Column.function.value:
        eval_metrics = _fit_one(df, metadata, column, average=average)
        return eval_metrics, set_values_to_zero(eval_metrics)
    else:
        median_metrics, mad_metrics = _fit_many(df, metadata, column, average=average)
        return median_metrics, mad_metrics

def _fit_linear_regression(df, metadata, column):
    '''
    fit linear regression model to predict metadata[column] from df
    '''
    print(f"Applying Linear Regression to {column}")
    X_train, X_test, y_train, y_test = train_test_split(df, metadata[column], test_size=0.3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute metrics for linear regression
    eval_metrics = {}
    eval_metrics['r2'] = metrics.r2_score(y_test, y_pred)
    eval_metrics['mse'] = metrics.mean_squared_error(y_test, y_pred)
    eval_metrics['mae'] = metrics.mean_absolute_error(y_test, y_pred)
    eval_metrics['explained_variance'] = metrics.explained_variance_score(y_test, y_pred)
    return eval_metrics

def _fit_many(df, metadata, column, average):
    #compute metrics for each batch, output should be a dictionary of dictionaries, 
    # where the outer dictionary has keys corresponding to the batch names and the 
    # inner dictionary has keys corresponding to the metrics
    eval_metrics = {}
    nunique_label = metadata[column].nunique()
    for batch in metadata[Column.batch.value].unique():
        isbatch = (metadata[Column.batch.value] == batch)
        if nunique_label < 100:
            eval_metrics[batch] = _fit_one(df[isbatch], metadata[isbatch], column, average)
        else:
            eval_metrics[batch] = _fit_linear_regression(df[isbatch], metadata[isbatch], column)
    #compute mean and standard deviation for each metric across the batches, output should be two dictionaries, one with mean and one with std computed across batches
    median_metrics = {}
    mad_metrics = {}
    all_metric_names = next(iter(eval_metrics.values())).keys()
    for metric_name in all_metric_names:
        median_metrics[metric_name] = median_arrays([eval_metrics[batch][metric_name] for batch in eval_metrics.keys()])
        mad_metrics[metric_name] = mad_arrays([eval_metrics[batch][metric_name] for batch in eval_metrics.keys()])
    return median_metrics, mad_metrics

def _fit_one(df, metadata, column, average):
    y = metadata[column]
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4, stratify=y)
    unique_targets = y_train.nunique()
    
    if unique_targets > 2:
        print(f"Applying Multinomial Logistic Regression to {column}")
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')
    else:
        print(f"Applying Logistic Regression to {column}")
        model = LogisticRegression(class_weight='balanced')

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    if unique_targets == 2:
        y_pred_proba = y_pred_proba[:, 1]
    # Compute metrics
    return get_metrics(y_test, y_pred, y_pred_proba, average)

def get_cellprofiler_features(file_path, features):
    chunk = pd.read_parquet(file_path)
    if chunk.shape[0] > 0:
        return chunk[features].replace([np.inf, -np.inf], np.nan).dropna(how="any")

def parget_cellprofiler_features(features):
    DATA_DIR = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22-phenotype-profiles'

    # Get all parquet files
    folder_path = f'{DATA_DIR}/mitotic-reclassified_cp_phenotype_normalized.parquet'
    mitotic_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]
    folder_path = f'{DATA_DIR}/interphase-reclassified_cp_phenotype_normalized.parquet'
    parquet_files = mitotic_files + [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.parquet')]

    tasks = [delayed(get_cellprofiler_features)(file_path, features) for file_path in parquet_files]
    results = Parallel(n_jobs=6)(tqdm(tasks))

    return pd.concat(results)


def get_metrics(y_test, y_pred, y_pred_proba, average):
    '''
    compute metrics given a model prediction
    '''
    le = preprocessing.LabelEncoder()
    y_test_label = le.fit_transform(y_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average=average)
    auroc = metrics.roc_auc_score(y_test_label, y_pred_proba, average=average, multi_class='ovr')
    precision = metrics.precision_score(y_test, y_pred, average=average)
    recall = metrics.recall_score(y_test, y_pred, average=average)

    return {'auroc':auroc, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':acc, 'confusion_matrix':cm}

def add_batch(metadata):
    if Column.batch.value not in metadata.columns:
        metadata[Column.batch.value] = metadata[Column.plate.value] + metadata[Column.well.value]
    return metadata

def add_well_position(metadata):
    disk = get_grid('6W_ph')
    edge_tiles, center_tiles = find_tile_adjacent_to_nan(disk)
    #for all rows in medadata with tile number in edge_tiles, set tile_position to edge
    metadata.loc[metadata[Column.tile.value].astype(str).isin(edge_tiles), 'well_position'] = 'edge'
    #for all rows in medadata with tile number in center_tiles, set tile_position to center
    metadata.loc[metadata[Column.tile.value].astype(str).isin(center_tiles), 'well_position'] = 'center'
    return metadata

def add_fov_position(metadata):
    # Add on new column describing distance of cell to tile center
    fov_position = np.abs(metadata[['cell_i','cell_j']]-metadata[['cell_i','cell_j']].mean(axis=0)).max(axis=1)
    metadata.loc[fov_position<450,'fov_position'] = 'center'
    metadata.loc[fov_position>=1200,'fov_position'] = 'edge'

    print(metadata['fov_position'].value_counts())
    return metadata

def add_covariates(metadata):
    return add_well_position(add_fov_position(add_batch(metadata)))

def reformat_dict(df, metric_name):
    reformated_dict = {key:{key_2:[] for key_2 in df.keys()} for key in metric_name}
    for space_key, space_val in df.items():
        for _, covariate_val in space_val.items():
            for metric_key, metric_val in covariate_val.items():
                reformated_dict[metric_key][space_key].append(metric_val)
    return reformated_dict

def get_grid(shape: str) -> np.ndarray:
    """
    Generate an array with tile index.

    :param shape: a string of grid layout
    :return:
    An array where entries are tile index. An entry's value being NaN means no image was taken at that location.
    """
    if shape == '6W_ph':
        rows = [7, 13, 17, 21, 25, 27, 29, 31, 33, 33, 35, 35, 37, 37, 39, 39, 39, 41, 41, 41, 41,
                41, 41, 41, 39, 39, 39, 37, 37, 35, 35, 33, 33, 31, 29, 27, 25, 21, 17, 13, 7]
    elif shape == '6W_sbs':
        rows = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21, 19, 19, 17, 17, 15, 13, 9, 5]
    elif isinstance(shape, list):
        rows = shape
    else:
        raise ValueError('{} shape not implemented, can pass custom shape as a'
                         'list specifying number of sites per row'.format(shape))

    c, r = len(rows), max(rows)
    grid = np.empty((r, c))
    grid[:] = np.NaN

    next_site = 0
    for col, row_sites in enumerate(rows):
        start = int((r - row_sites) / 2)
        if col % 2 == 0:
            grid[start:start + row_sites, col] = range(next_site, next_site + row_sites)
        else:
            grid[start:start + row_sites, col] = range(next_site, next_site + row_sites)[::-1]
        next_site += row_sites
    return grid
    

def set_values_to_zero(dictionary):
    dict_copy = deepcopy(dictionary)
    for key in dict_copy.keys():
        if isinstance(dict_copy[key], dict):
            dict_copy[key] = set_values_to_zero(dict_copy[key])
        else:
            dict_copy[key] = 0
    return dict_copy

def median_arrays(arr_list):
    # Check if the elements in the list are numbers (scalars)
    if np.isscalar(arr_list[0]):
        # Calculate and return median for list of numbers
        return np.median(arr_list)
    else:
        # Stack arrays along new axis
        stacked_arrs = np.stack(arr_list)
        # Calculate and return median for list of arrays
        return np.median(stacked_arrs, axis=0)

def mad_arrays(arr_list):
    # Check if the elements in the list are numbers (scalars)
    if np.isscalar(arr_list[0]):
        # Calculate and return MAD for list of numbers
        return median_abs_deviation(arr_list)
    else:
        # Stack arrays along new axis
        stacked_arrs = np.stack(arr_list)
        # Calculate and return MAD for list of arrays
        return median_abs_deviation(stacked_arrs, axis=0)
    

def select_center_coordinates(disk, center, num_points):
    # Calculate the distance of each point from the center
    distances = np.sqrt((np.indices(disk.shape).T - center)**2).sum(-1)

    # Create a mask to exclude NaN values
    mask = np.isnan(disk)

    # Exclude the NaN values from the distance array
    masked_distances = distances[~mask]

    # Sort the distances
    sorted_distances = np.sort(masked_distances)

    # Find the radius that includes num_points
    radius = sorted_distances[num_points]

    # Create a circular mask
    y,x = np.ogrid[-center[0]:disk.shape[0]-center[0], -center[1]:disk.shape[1]-center[1]]
    mask = x*x + y*y <= radius*radius

    # Get the coordinates of the points within the circular mask and inside the disk
    center_coordinates = np.column_stack(np.where(np.logical_and(mask, ~np.isnan(disk))))
    
    return center_coordinates


def find_tile_adjacent_to_nan(disk):
    # Find the disk center
    center = np.array(disk.shape) // 2

    # Placeholder lists for edge and center coordinates
    edge_coordinates = []
    center_coordinates = []

    # Define the kernel to find the neighbors
    kernel = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    # Go over the disk array
    for i in range(disk.shape[0]):
        for j in range(disk.shape[1]):
            # If the point is a number
            if not np.isnan(disk[i, j]):
                # If the point is on the edge of the array
                if i == 0 or i == disk.shape[0] - 1 or j == 0 or j == disk.shape[1] - 1:
                    edge_coordinates.append([i, j])
                else:
                    # Check all the neighbors
                    for k in range(8):
                        ni, nj = np.array([i, j]) + kernel[k]
                        # If the neighbor is NaN, it means this point is at the edge
                        if np.isnan(disk[ni, nj]):
                            edge_coordinates.append([i, j])
                            break

    # Select roughly similar number of coordinates from the center
    center_coordinates = np.array(select_center_coordinates(disk, center, len(edge_coordinates)))
    edge_coordinates = np.array(edge_coordinates)
    center_tile = disk[center_coordinates[:,0], center_coordinates[:,1]].astype('int').astype('str')
    edge_tile = disk[edge_coordinates[:,0], edge_coordinates[:,1]].astype('int').astype('str')

    return edge_tile, center_tile


from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_confusion_matrices(eval_metrics, classes, normalize=False, cmap=plt.cm.Blues):
    if isinstance(eval_metrics, dict):
        n_matrices = len(eval_metrics)
        is_dict = True
    else:
        n_matrices = 1
        is_dict = False
        eval_metrics = {'': eval_metrics}
        
    fig, axes = plt.subplots(1, n_matrices, figsize=(n_matrices * 4, 5), dpi=200, sharex=True, sharey=True)
    if n_matrices == 1:
        axes = [axes]
    vmax = 0

    for key in eval_metrics:
        if normalize:
            cm = eval_metrics[key].astype('float') / eval_metrics[key].sum(axis=1)[:, np.newaxis]
        else:
            cm = eval_metrics[key]
        vmax = max(vmax, cm.max())

    for ax, (key, cm) in zip(axes, eval_metrics.items()):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=vmax)
        title = key
        if not is_dict:
            title = title.replace(' ()', '')
        ax.set_title(title, fontsize=14)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel('Predicted label', fontsize=13)
        if key == 'salient':
            ax.set_ylabel('True label', fontsize=13)

        # thresh = vmax / 2.
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     ax.text(j, i, np.round(cm[i, j]*100, 2), fontsize=2.5,
        #             horizontalalignment="center",
        #             color="white" if cm[i, j] > thresh else "black")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig('figure/confusion_matrix.svg', bbox_inches='tight')
    plt.show()

def remove_highly_correlated_features(correlation_matrix, threshold=0.9, max_removal_proportion=0.8):
    # Get the shape of the correlation matrix
    n_features = correlation_matrix.shape[0]

    # Maximum number of features to remove
    max_removals = int(n_features * max_removal_proportion)
    
    # Create a set to store the indices of the features to drop
    to_drop = set()

    # Iterate through the correlation matrix
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # If we have already reached the maximum number of removals, break
            if len(to_drop) >= max_removals:
                break
            
            # If the feature is already marked to drop, skip the iteration
            if i in to_drop or j in to_drop:
                continue

            # If the correlation is above the threshold
            if np.abs(correlation_matrix[i, j]) > threshold:
                # Get the mean absolute correlation for both features
                mean_corr_i = np.mean(np.abs(correlation_matrix[i, [k for k in range(n_features) if k != i and k not in to_drop]]))
                mean_corr_j = np.mean(np.abs(correlation_matrix[j, [k for k in range(n_features) if k != j and k not in to_drop]]))

                # If both features have low mean correlation with other features, skip this pair
                if mean_corr_i < threshold and mean_corr_j < threshold:
                    continue
                
                # Drop the feature with the higher mean absolute correlation with other features
                if mean_corr_i > mean_corr_j:
                    to_drop.add(i)
                else:
                    to_drop.add(j)

        # If we have already reached the maximum number of removals, break
        if len(to_drop) >= max_removals:
            break

    # Create a list of the features to keep
    features_to_keep = [i for i in range(n_features) if i not in to_drop]

    return features_to_keep
