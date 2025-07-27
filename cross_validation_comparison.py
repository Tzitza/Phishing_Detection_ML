from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.io import arff

# Φόρτωση δεδομένων ARFF
file_path = "C:/Users/ilias/OneDrive/Desktop/ΤΖΙΤΖΑ_2589_ΕΡΓΑΣΙΑ1/Training Dataset.arff"
data, meta = arff.loadarff(file_path)

# Δημιουργία DataFrame
df = pd.DataFrame(data)

# Διαχωρισμός των χαρακτηριστικών (X) και των labels (y)
X = df.drop(columns=['Result'])
y = df['Result']

# Αν τα labels είναι σε byte μορφή, μετατροπή σε κανονική μορφή
y = y.str.decode('utf-8')

# Εκτέλεση 10-Fold Cross-Validation για Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
scores_dt = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')

print("Αποτελέσματα 10-Fold Cross-Validation για Decision Tree:")
print(f"Ακρίβεια ανά fold: {scores_dt}")
print(f"Μέση ακρίβεια: {scores_dt.mean():.4f}")
print(f"Τυπική απόκλιση: {scores_dt.std():.4f}")

# Κανονικοποίηση των δεδομένων για k-NN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Μετατροπή των labels σε ακέραιες τιμές για k-NN
y_knn = y.astype(int)

# Δημιουργία του μοντέλου k-NN
knn = KNeighborsClassifier(n_neighbors=10)  # k = 10 για το παράδειγμα

# Εκτέλεση 10-Fold Cross-Validation για k-NN
scores_knn = cross_val_score(knn, X_scaled, y_knn, cv=10, scoring='accuracy')

print("\nΑποτελέσματα 10-Fold Cross-Validation για k-NN:")
print(f"Ακρίβεια ανά fold: {scores_knn}")
print(f"Μέση ακρίβεια: {scores_knn.mean():.4f}")
print(f"Τυπική απόκλιση: {scores_knn.std():.4f}")
