import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV data
loan_data = pd.read_csv('data/loan_data.csv')

# Categorize data
gender_map = {
    'male': 0,
    'female': 1
}

education_map = {
    'Master': 0,
    'High School': 1,
    'Bachelor': 2
}

home_ownership_map = {
    'RENT': 0,
    'OWN': 1,
    'MORTGAGE': 2
}

loan_intent_map = {
    'PERSONAL': 0, 
    'EDUCATION': 1,
    'MEDICAL': 2,
    'VENTURE': 3,
    'HOMEIMPROVEMENT': 4,
    'DEBTCONSOLIDATION': 5
}

previous_loan_defaults_map = {
    'No': 0, 
    'Yes': 1,
}

loan_data = loan_data[loan_data['person_age'] <= 100]
loan_data['person_gender'] = loan_data['person_gender'].map(gender_map)
loan_data['person_education'] = loan_data['person_education'].map(education_map)
loan_data['person_home_ownership'] = loan_data['person_home_ownership'].map(home_ownership_map)
loan_data['loan_intent'] = loan_data['loan_intent'].map(loan_intent_map)
loan_data['previous_loan_defaults_on_file'] = loan_data['previous_loan_defaults_on_file'].map(previous_loan_defaults_map)
loan_data.dropna(inplace=True)


# Split features and target
X = loan_data.drop(columns=['loan_status'])  # Features
y = loan_data['loan_status']  # Target label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
#random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
#random_forest_model.fit(X_train, y_train)

# Make predictions
#y_pred = random_forest_model.predict(X_test)


# Train the KNN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust 'n_neighbors' based on your preference
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)



# Compute confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
true_negative, false_positive, false_negative, true_positive = confusion_mat.ravel()

accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive)

# Print results
print("Confusion Matrix:")
print(confusion_mat)
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")

# Export model to the pickle file
with open('loan_approval.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)
    #pickle.dump(random_forest_model, model_file)

print("Model has been saved to 'loan_approval.pkl'.")