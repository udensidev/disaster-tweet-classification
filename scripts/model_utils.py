import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """
    Train a Bernoulli Naive Bayes model.
    Args:
        X_train: sparse matrix, training feature vectors
        y_train: array-like, training labels
    Returns:
        model: trained Bernoulli Naive Bayes model
    """
    model = BernoulliNB().fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate a trained model on a validation set.
    Args:
        model: trained machine learning model
        X_val: sparse matrix, validation feature vectors
        y_val: array-like, validation labels
    Returns:
        accuracy: float, validation accuracy
    """
    pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, pred_val)
    print(f"\nValidation accuracy: {accuracy:.4f}")
   
    

def make_predictions(model, X_test, test_ids, output_path='data/processed/submission.csv', file_name='submission.csv'):
    """
    Make predictions with a trained model and save them to a CSV file.
    Args:
        model: trained machine learning model
        X_test: sparse matrix, test feature vectors
        test_ids: array-like, IDs for the test data
        output_file: str, path to save the submission file
    """
    # Make predictions on the test set
    pred_test = model.predict(X_test)
    
    # Create a DataFrame
    submission = pd.DataFrame({'id': test_ids, 'target': pred_test})
    
    # Save the DataFrame to a CSV file
    submission.to_csv(output_path, index=False)
    print(f"\nPredictions made and saved to file: {file_name}")