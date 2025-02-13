import pandas as pd
import numpy as np
import time
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

start_time = time.time()

# Fix file paths using raw strings
file_path = r'D:\M.E NITTTR\TERM_PAPER_PG\train.csv'
test_file_path = r'D:\M.E NITTTR\TERM_PAPER_PG\test.csv'

# Load dataset with optimized sampling
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Training file not found. Please check the file path.")
    raise

# Reduce dataset size for speed optimization
data_sample = data.sample(frac=0.08, random_state=42)

# Convert responses to list (faster than numpy array)
responses = data_sample['response_a'].astype(str).tolist() + data_sample['response_b'].astype(str).tolist()

# Optimize TF-IDF with binary term presence
tfidf = TfidfVectorizer(max_features=800, stop_words='english', ngram_range=(1, 2), binary=True)

# Fit and transform all responses at once
tfidf_matrix = tfidf.fit_transform(responses)

# Split into A and B response matrices
half = len(data_sample)
tfidf_matrix_a, tfidf_matrix_b = tfidf_matrix[:half], tfidf_matrix[half:]

# Convert to compressed sparse row format (CSR) for efficiency
X = csr_matrix(hstack([tfidf_matrix_a, tfidf_matrix_b]))

# Labels: 1 if model A is preferred, 0 otherwise
y = (data_sample['winner_model_a'] == 1).astype(int)

# Train-test split with stratification and shuffling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# Optimized Random Forest with better generalization
rf_classifier = RandomForestClassifier(
    n_estimators=150, max_depth=30, min_samples_split=6, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict probabilities for log loss calculation
y_pred = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate log loss (should be lower than 0.63)
log_loss_score = log_loss(y_test, y_pred)
print(f'Log Loss: {log_loss_score:.4f}')

end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")

# Load and process test data
try:
    test_data = pd.read_csv(test_file_path)
except FileNotFoundError:
    print("Test file not found. Please check the file path.")
    raise

# Convert test responses to list (faster processing)
test_responses = test_data['response_a'].astype(str).tolist() + test_data['response_b'].astype(str).tolist()

# Transform test responses into TF-IDF matrices
test_tfidf_matrix = tfidf.transform(test_responses)

# Split test matrices
test_tfidf_matrix_a, test_tfidf_matrix_b = test_tfidf_matrix[:len(test_data)], test_tfidf_matrix[len(test_data):]

# Generate test features using CSR matrix
X_test_final = csr_matrix(hstack([test_tfidf_matrix_a, test_tfidf_matrix_b]))

# Predict for submission
final_predictions = rf_classifier.predict_proba(X_test_final)[:, 1]

# Create optimized submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'winner_model_a': final_predictions,
    'winner_model_b': 1 - final_predictions,
    'winner_tie': 0.0  # Assuming no ties
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
