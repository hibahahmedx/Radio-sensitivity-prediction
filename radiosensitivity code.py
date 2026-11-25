
import pandas as pd\n
from sklearn.model_selection import train_test_split\n
from sklearn.ensemble import RandomForestClassifier\n
from sklearn.ensemble import BaggingClassifier\n
from sklearn.feature_selection import SelectKBest, chi2\n
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n
import seaborn as sns\n
import matplotlib.pyplot as plt\n
import numpy as np\n
"\n"

# Load the dataset\n"
df = pd.read_csv(r\"E:\\breast cancer genomic-clinical dataset.csv\")\n
df.head()

#Seperating features: radio sensitivity in cancer patients\n"
X = df.drop(\"Radio_Sensitivity\", axis=1)\n"
y = df[\"Radio_Sensitivity\"]\n"
"\n"
#Spliting the data into training and testing sets\n"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n

#1: feature selection filter method: Chi-squared test\n"
\n
X_train_selected_chi2 = SelectKBest(chi2, k=5).fit_transform(X_train, y_train)\n
X_test_selected_chi2 = SelectKBest(chi2, k=5).fit_transform(X_test, y_test)\n
chi2_feature_names = X.columns[SelectKBest(chi2, k=5).fit(X_train, y_train).get_support()]\n

# 2: Feature Importance from Random Forest\n",
initial_rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
initial_rf.fit(X_train, y_train)\n",
feature_importances = initial_rf.feature_importances_\n",
features = X.columns\n",
important_features = pd.DataFrame({'Feature': features, 'Importance': feature_importances})\n,
important_features = important_features.sort_values(by='Importance', ascending=False)\n,ArithmeticError
# Select the top 5 most important features\n,
rf_top_features = important_features['Feature'].head(5).values

#Combine the features selected by Chi-squared and Random Forest\n",
combined_top_features = np.unique(np.concatenate((chi2_feature_names, rf_top_features)))

# Creating new training and testing sets with the combined selected features\n",
X_train_combined = X_train[combined_top_features]\n,
X_test_combined = X_test[combined_top_features]

# Define a Bagging Classifier with Random Forest using selected features\n",
bagging_rf_model = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),\n",
n_estimators=3,  # Train 3 different Random Forest models\n",
random_state=42,\n",
n_jobs=-1\n)
bagging_rf_model.fit(X_train_combined, y_train)\n
















