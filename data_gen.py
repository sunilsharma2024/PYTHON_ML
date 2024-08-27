import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from feature_extraction_model import cnn_feature_extraction, GNN, resnet_50

def modified_smote(data, minority_class_indices, k=5, num_synthetic=1):
    synthetic_data = []
    neighbors = NearestNeighbors(n_neighbors=k).fit(data)

    for idx in minority_class_indices:
        instance = data[idx]
        nearest_neighbors = neighbors.kneighbors([instance], return_distance=False)[0]

        for _ in range(num_synthetic):
            alpha = np.random.rand()
            synthetic_instance = instance + alpha * (data[nearest_neighbors[np.random.choice(k)]] - instance)
            synthetic_data.append(synthetic_instance)

    if synthetic_data:
        return np.vstack([data, np.array(synthetic_data)])
    else:
        return data  # No synthetic instances generated, return original data

def datagen():
    # Load dataset from Excel file
    dataset1 = pd.read_excel('./Dataset/CIC-IDS2018.xlsx')

    # Assign column names based on index
    dataset1.columns = [str(i) for i in range(len(dataset1.columns))]

    # Replace label strings with numerical values
    label_mapping = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-Bruteforce': 2}
    label = dataset1.columns[-1]
    dataset1[label].replace(label_mapping, inplace=True)

    # Drop rows with missing values and duplicates
    dataset1.dropna(inplace=True)
    columns_to_remove = ['2']
    dataset1 = dataset1.drop(columns=columns_to_remove, axis=1)
    dataset1 = pd.DataFrame.to_numpy(dataset1)
    labels1 = dataset1[:, -1].astype('int16')
    dataset1 = np.delete(dataset1, -1, axis=1)  # Assuming the last column is the label

    # Perform K-nearest neighbors (KNN) based data imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_features1 = knn_imputer.fit_transform(dataset1)

    # Combine imputed features with labels
    dataset_imputed = np.column_stack((imputed_features1, labels1))

    # Identify minority class indices
    minority_class_indices = np.where(labels1 != 0)[0]

    # Perform modified SMOTE-based data augmentation
    augmented_data = modified_smote(dataset_imputed, minority_class_indices, k=5, num_synthetic=1)
    labels1 = (augmented_data[:, -1].astype('int16'))
    dataset1 = np.delete(augmented_data, -1, axis=1)

    # Initialize StandardScaler (Z-score normalization)
    scaler = StandardScaler()

    # Perform Z-score normalization on the features
    dataset1 = scaler.fit_transform(dataset1)

    # Feature extraction
    mean = dataset1.mean(axis=1)
    median = np.median(dataset1, axis=1)
    std = dataset1.std(axis=1)
    var = dataset1.var(axis=1)
    dataset = pd.DataFrame(dataset1)
    mode = dataset.mode(axis=1).iloc[:, 0]
    skew = dataset.skew(axis=1)
    kurt = dataset.kurt(axis=1)

    # Create a DataFrame for statistical features
    feat1 = pd.DataFrame({
        'mean': mean,
        'median': median,
        'var': var,
        'std': std,
        'skew': skew,
        'kurt': kurt,
        'mode': mode
    })

    # Deep learning based features (CNN, GNN, ResNet50)
    X_train, X_test, y_train, y_test = train_test_split(dataset1, labels1, test_size=0.2, random_state=42)

    feat2 = cnn_feature_extraction(X_train, y_train, X_test, y_test)
    feat2 = pd.DataFrame(feat2)

    feat3 = GNN(X_train, y_train, X_test, y_test)
    feat3 = pd.DataFrame(feat3)

    feat4 = resnet_50(X_train, y_train, X_test, y_test)
    feat4 = pd.DataFrame(feat4)

    # Concatenate the DataFrames horizontally (along columns) to create 'df_extra'
    feature1 = pd.concat([pd.DataFrame(dataset1), feat1, feat2, feat3, feat4], axis=1)

    # Normalization
    feat_2 = feature1 / np.max(feature1, axis=0)
    feat_2 = np.nan_to_num(feat_2)

    # Split the combined data into training and testing sets
    train_data, test_data, train_lab, test_lab = train_test_split(feat_2, labels1, test_size=0.3, random_state=42)

    # Save data
    save('X_train', train_data)
    save('X_test', test_data)
    save('y_train', train_lab)
    save('y_test', test_lab)

datagen()
