import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

COLUMNS_TO_DROP = ['Pos', 'Tm', 'G', 'FG%', '3P%', '2P%', 'FT%', 'eFG%']

def drop_columns(train_data, COLUMNS_TO_DROP):
    X_train, X_validation, y_train, y_validation = split_train_validation(train_data, COLUMNS_TO_DROP)
    return COLUMNS_TO_DROP + select_features(X_train, X_validation, y_train, y_validation)

def load_data(filename):
    ''' Load data from a CSV file. '''
    return pd.read_csv(filename)

def split_train_validation(data, drop_cols, train_size=0.8):
    ''' Split the dataset into training and validation sets. '''
    if train_size == 1:
        return data.drop(drop_cols, axis=1), data['Pos']
    return train_test_split(data.drop(drop_cols, axis=1), data['Pos'], train_size=train_size, random_state=0)

def select_features(X_train, X_val, y_train, y_val):
    ''' Identify low-importance features using permutation importance. '''
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    importance_scores = permutation_importance(tree, X_val, y_val, random_state=0).importances_mean
    low_importance_features = pd.Series(importance_scores, index=X_train.columns)
    return list(low_importance_features[low_importance_features.round(2) <= 0].index)

def train_neural_network(X_train, y_train):
    ''' Train a multi-layer neural network on the training data. '''
    nn_model = MLPClassifier(
        solver='adam',
        alpha=1e-4,
        hidden_layer_sizes=(40, 15),
        max_iter=250,
        activation='tanh',
        random_state=0
    )
    nn_model.fit(X_train, y_train)
    return nn_model

def evaluate_model(model, X, y, dataset_type=""):
    ''' Evaluate model accuracy and confusion matrix for a given dataset. '''
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f'\n{dataset_type} Accuracy: {accuracy * 100:.2f}%')
    print(f'\n{dataset_type} Confusion Matrix:\n{confusion_matrix(y, predictions)}')

def cross_validate_model(model, X, y):
    ''' Perform 10-fold stratified cross-validation and display fold-wise accuracy. '''
    
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cv_results = cross_validate(model, X, y, cv=stratified_kfold, scoring='accuracy')

    print('10-Fold Cross-Validation Accuracy:\n')
    for i, acc in enumerate(cv_results['test_score']):
        print(f'Fold {i + 1}: {acc * 100:.2f}%')
    print(f'\nAverage Accuracy: {cv_results["test_score"].mean() * 100:.2f}%')

if __name__ == '__main__':
    dash_count = 30
    
    # Load data
    train_data = load_data('nba_stats.csv')
    test_data = load_data('dummy_test.csv')
    
    # Feature selection
    selected_columns_to_drop = drop_columns(train_data, COLUMNS_TO_DROP)
    
    # Finalize training/validation sets after feature selection
    X_train, X_val, y_train, y_val = split_train_validation(train_data, selected_columns_to_drop)
    X_test, y_test = test_data.drop(selected_columns_to_drop, axis=1), test_data['Pos']
    
    # Train neural network
    nn_model = train_neural_network(X_train, y_train)

    # Task 1: Evaluate on training and validation sets
    print(f'\n{"-" * dash_count} Task 1 {"-" * dash_count}\n')

    evaluate_model(nn_model, X_train, y_train, dataset_type="Training")

    evaluate_model(nn_model, X_val, y_val, dataset_type="Validation")

    # Task 2: Evaluate on test set
    print(f'\n{"-" * dash_count} Task 2 {"-" * dash_count}\n')

    evaluate_model(nn_model, X_test, y_test, dataset_type="Test")

    # Task 3: Cross-validation
    print(f'\n{"-" * dash_count} Task 3 {"-" * dash_count}\n')
    COLUMNS_TO_DROP = ['Pos', 'Tm', 'G', 'PF', 'FG%', '3P%', '2P%', 'FT%', 'eFG%']
    drop_col = drop_columns(train_data, COLUMNS_TO_DROP)

    X_train_full, y_train_full = split_train_validation(train_data, drop_col, train_size=1.0)

    cross_validate_model(nn_model, X_train_full, y_train_full)

    print(f'\n{"-" * 2 * (dash_count + 4)}\n')



