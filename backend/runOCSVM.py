import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score

class OCSVMExperiment:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

    def run(self):
        # Prepare data
        X_train = self.train_data.drop(columns=['label'])
        X_test = self.test_data.drop(columns=['label'])
        y_test = self.test_data['label'].values

        # Initialize and fit the One-Class SVM
        ocsvm = OneClassSVM(gamma='auto', nu=0.5)
        ocsvm.fit(X_train)

        # Predict test data
        predictions = ocsvm.predict(X_test)
        # Convert inliers (1) to 0, and outliers (-1) to 1 for evaluation
        predicted_labels = [0 if x == 1 else 1 for x in predictions]
        # Evaluate the results
        return self.evaluate_results(predicted_labels, y_test)

    def evaluate_results(self, predicted_labels, true_labels):
        f1 = f1_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
    
    

        metrics = {
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }
        return metrics

def run(train_path, test_path):
    experiment = OCSVMExperiment(train_path, test_path)
    results = experiment.run()
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python runOCSVM.py <train_path> <test_path>")
    else:
        run(sys.argv[1], sys.argv[2])
