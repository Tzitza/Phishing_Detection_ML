from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

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
y = y.astype(int)  # Μετατροπή σε ακέραιες τιμές (-1 και 1)

# Κανονικοποίηση των δεδομένων
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Τιμές για k που θα δοκιμάσουμε
k_values = range(1, 21)

# Λίστα για αποθήκευση της ακρίβειας για κάθε τιμή του k
accuracies = []

# Εφαρμογή του k-NN για διαφορετικές τιμές του k
for k in k_values:
    # Δημιουργία του μοντέλου k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Εκτέλεση 10-fold cross-validation με κανονικοποιημένα δεδομένα
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    
    # Αποθήκευση της μέσης ακρίβειας
    accuracies.append(scores.mean())

# Εμφάνιση των αποτελεσμάτων
plt.plot(k_values, accuracies, marker='o')
plt.title("Ακρίβεια του k-NN για διαφορετικές τιμές του k")
plt.xlabel("Τιμή k")
plt.ylabel("Μέση Ακρίβεια")
plt.grid(True)
plt.show()

# Εκτύπωση του καλύτερου k
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Η καλύτερη τιμή για το k είναι: {best_k} με μέση ακρίβεια {max(accuracies):.4f}")
