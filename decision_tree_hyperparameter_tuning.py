from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
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

# Δοκιμή διαφορετικών τιμών για max_leaf_nodes
max_leaf_nodes_values = [10, 20, 30, 50, 100, 200, None]

for max_leaf_nodes in max_leaf_nodes_values:
    # Δημιουργία του Decision Tree με max_leaf_nodes
    decision_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    
    # Εκτέλεση 10-fold cross-validation
    scores = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')
    
    # Εκτύπωση του μέσου όρου της ακρίβειας
    print(f"Max Leaf Nodes: {max_leaf_nodes} -> Μέση Ακρίβεια: {scores.mean():.4f}")
