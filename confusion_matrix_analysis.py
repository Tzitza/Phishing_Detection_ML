from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import arff

# Φόρτωση δεδομένων ARFF
file_path = "C:/Your/file/path"
data, meta = arff.loadarff(file_path)

# Δημιουργία DataFrame
df = pd.DataFrame(data)

# Διαχωρισμός των χαρακτηριστικών (X) και των labels (y)
X = df.drop(columns=['Result'])
y = df['Result']

# Αν τα labels είναι σε byte μορφή, μετατροπή σε κανονική μορφή
y = y.str.decode('utf-8')
y = y.astype(int)  # Μετατροπή σε ακέραιες τιμές (-1 και 1)

# Διαχωρισμός σε εκπαίδευση και δοκιμή
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Κανονικοποίηση για το k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Κανονικοποίηση των δεδομένων εκπαίδευσης
X_test_scaled = scaler.transform(X_test)       # Κανονικοποίηση των δεδομένων δοκιμής

# Δημιουργία μοντέλων
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)

# Εκπαίδευση Decision Tree (χωρίς κανονικοποίηση)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

# Εκπαίδευση k-NN (με κανονικοποίηση)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Δημιουργία πίνακα σύγχυσης για Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=[-1, 1])
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["Phishing", "Legitimate"])

# Δημιουργία πίνακα σύγχυσης για k-NN
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=[-1, 1])
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Phishing", "Legitimate"])

# Παρουσίαση πινάκων σύγχυσης
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp_dt.plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Πίνακας Σύγχυσης για Decision Tree")

disp_knn.plot(ax=axes[1], cmap="Blues", values_format="d")
axes[1].set_title("Πίνακας Σύγχυσης για k-NN")

plt.show()
