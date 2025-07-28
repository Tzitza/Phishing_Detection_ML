# Phishing Detection ML

Machine learning algorithms (Decision Trees & k-NN) for detecting phishing websites using URL features

## Overview

This project implements and compares two machine learning algorithms for detecting phishing websites:
- **Decision Trees**: Creates a tree-based decision model using website features
- **k-Nearest Neighbors (k-NN)**: Classifies websites based on similarity to neighboring data points

## Dataset

The project uses the [Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites) from UCI Machine Learning Repository, which contains various URL and website characteristics that help identify phishing attempts.

### Key Features Analyzed:
- URL length and structure
- Use of IP addresses instead of domains
- HTTPS certificate validation
- Number of subdomains
- Favicon source verification
- Redirect patterns
- Special characters in URLs (e.g., "@" symbol)

## Project Structure

```
phishing-detection-ml/
├── data_exploration_and_visualization.py      # Dataset analysis and visualization
├── cross_validation_comparison.py             # 10-fold cross-validation for both algorithms
├── decision_tree_hyperparameter_tuning.py    # Optimization of max_leaf_nodes parameter
├── knn_k_optimization.py                      # Finding optimal k value for k-NN
├── model_performance_evaluation.py            # Performance metrics calculation
├── confusion_matrix_analysis.py               # Confusion matrix visualization
├── Training Dataset.arff                      # Dataset file
└── README.md
```

## Requirements

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- scipy

## Installation

```bash
pip install scikit-learn pandas matplotlib scipy
```

## Usage

### 1. Data Exploration
```bash
python data_exploration_and_visualization.py
```
- Loads and analyzes the dataset
- Shows class distribution (phishing vs legitimate)
- Displays statistical summary of features

### 2. Cross-Validation Comparison
```bash
python cross_validation_comparison.py
```
- Performs 10-fold cross-validation on both algorithms
- Compares baseline performance

### 3. Hyperparameter Optimization

#### Decision Tree:
```bash
python decision_tree_hyperparameter_tuning.py
```
- Tests different `max_leaf_nodes` values: [10, 20, 30, 50, 100, 200, None]
- Finds optimal tree complexity

#### k-NN:
```bash
python knn_k_optimization.py
```
- Tests k values from 1 to 20
- Visualizes accuracy vs k value
- Identifies optimal k parameter

### 4. Performance Evaluation
```bash
python model_performance_evaluation.py
```
- Calculates accuracy, recall, and F1-score
- Compares final model performance

### 5. Confusion Matrix Analysis
```bash
python confusion_matrix_analysis.py
```
- Generates confusion matrices for both algorithms
- Visualizes classification results

## Methodology

### Data Preprocessing
- Conversion of binary features to integers
- Feature-label separation
- Data normalization for k-NN (StandardScaler)
- No normalization for Decision Trees

### Training Strategy
- **Cross-Validation**: 10-fold cross-validation for robust evaluation
- **Train-Test Split**: 70-30 split for final performance assessment
- **Random State**: Set to 42 for reproducible results

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Recall**: Ability to identify phishing websites (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Key Findings

### Dataset Characteristics
- Total samples: 11055
- Class distribution: Legitimate vs Phishing
- Feature count: 30 URL and website characteristics

### Algorithm Performance
- **Decision Tree**:
  - Optimal max_leaf_nodes: 100
  - Final accuracy: 0.9526
  - Advantages: Interpretable, handles non-linear relationships
  
- **k-NN**:
  - Optimal k value: 1
  - Final accuracy: 0.9644
  - Advantages: Simple, effective for pattern recognition

### Comparison Results
- Performance metrics comparison
- When to use each algorithm
- Computational complexity considerations

## Technical Details

### Decision Tree Configuration
```python
DecisionTreeClassifier(
    max_leaf_nodes=optimal_value,
    random_state=42
)
```

### k-NN Configuration
```python
KNeighborsClassifier(
    n_neighbors=optimal_k,
    metric='euclidean'
)
```

### Data Normalization
- Applied only to k-NN algorithm using StandardScaler
- Decision Trees work directly with original feature scales

## Results Interpretation

### Confusion Matrix Analysis
- **True Positives**: Correctly identified phishing sites
- **True Negatives**: Correctly identified legitimate sites
- **False Positives**: Legitimate sites misclassified as phishing
- **False Negatives**: Phishing sites misclassified as legitimate

### Performance Implications
- High recall is crucial for phishing detection (minimize false negatives)
- Balance between accuracy and computational efficiency
- Consider deployment constraints and real-time requirements

## Future Improvements

- Feature engineering and selection
- Ensemble methods combination
- Deep learning approaches
- Real-time URL analysis pipeline
- Integration with web browsers or security tools

## References

- [Phishing Websites Dataset - UCI ML Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- [Scikit-learn Documentation](https://scikit-learn.org/dev/index.html)
- Course: Data Mining - University of Ioannina, Department of Computer Science and Telecommunications

## License

This project is developed for educational purposes as part of a Data Mining course assignment.
