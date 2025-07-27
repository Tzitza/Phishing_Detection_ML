import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

# 1. Φόρτωση δεδομένων
file_path = "C:/Users/ilias/OneDrive/Desktop/ΤΖΙΤΖΑ_2589_ΕΡΓΑΣΙΑ1/Training Dataset.arff"
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# 2. Μετατροπή δυαδικών δεδομένων σε ακέραιους
df = df.applymap(lambda x: int(x.decode('utf-8')))

# 3. Στατιστική περιγραφή χαρακτηριστικών
feature_summary = df.describe().transpose()

# 4. Υπολογισμός συχνοτήτων της στήλης "Result"
category_counts = df['Result'].value_counts()

# 5. Εμφάνιση αριθμού αποτελεσμάτων ανά κατηγορία
print("Αριθμός αποτελεσμάτων ανά κατηγορία:")
print(f"Νόμιμο (1): {category_counts.get(1, 0)}")
print(f"Phishing (-1): {category_counts.get(-1, 0)}")

# 6. Δημιουργία γραφήματος
plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Κατανομή Κατηγοριών στο Dataset', fontsize=14)
plt.xlabel('Κατηγορία (1 = Νόμιμο, -1 = Phishing)', fontsize=12)
plt.ylabel('Πλήθος Εγγραφών', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 7. Εκτύπωση στατιστικής περιγραφής
print("Στατιστική Περιγραφή Χαρακτηριστικών:")
print(feature_summary)
