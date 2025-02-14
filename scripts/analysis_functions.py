import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Bernoulli Naive Bayes model.
    Args:
        X_train: sparse matrix, training feature vectors
        y_train: array-like, training labels
        X_val: sparse matrix, validation feature vectors
        y_val: array-like, validation labels
    Returns:
        model: trained Bernoulli Naive Bayes model
        accuracy: float, validation accuracy
    """
    model = BernoulliNB().fit(X_train, y_train)
    pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, pred_val)
    print(f"Model trained. Validation accuracy: {accuracy}")
    return model, accuracy

def make_predictions(model, X_test, test_ids, output_file='submission.csv'):
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
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions made and saved to file: {output_file}")