from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
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

# Μετατροπή σε ακέραιες τιμές (1 για Νόμιμο και -1 για Phishing)
y = y.astype(int)

# Διαχωρισμός σε εκπαίδευση και δοκιμή
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Κανονικοποίηση μόνο για το k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Κανονικοποίηση δεδομένων εκπαίδευσης
X_test_scaled = scaler.transform(X_test)       # Κανονικοποίηση δεδομένων δοκιμής

# Δημιουργία μοντέλων
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)  # Επιλέξαμε k=10 για το παράδειγμα

# Εκπαίδευση Decision Tree (χωρίς κανονικοποίηση)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

# Εκπαίδευση k-NN (με κανονικοποίηση)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Υπολογισμός μετρικών για το Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt, pos_label=-1)  # Χρησιμοποιούμε το -1 για phishing
f1_dt = f1_score(y_test, y_pred_dt, pos_label=-1)

# Υπολογισμός μετρικών για το k-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn, pos_label=-1)  # Χρησιμοποιούμε το -1 για phishing
f1_knn = f1_score(y_test, y_pred_knn, pos_label=-1)

# Εμφάνιση αποτελεσμάτων
print(f"Ακρίβεια Decision Tree: {accuracy_dt:.4f}")
print(f"Ανάκληση Decision Tree: {recall_dt:.4f}")
print(f"F1 Score Decision Tree: {f1_dt:.4f}")
print(f"--------------------------------------------")
print(f"Ακρίβεια k-NN: {accuracy_knn:.4f}")
print(f"Ανάκληση k-NN: {recall_knn:.4f}")
print(f"F1 Score k-NN: {f1_knn:.4f}")
