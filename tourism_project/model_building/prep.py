import pandas as pd
from sklearn.model_selection import train_test_split

#registered tourism.csv inside the data folder
df = pd.read_csv("tourism_project/data/tourism.csv") 

#complete the code: drop the customer identifier column, it is not a predictive feature
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)   

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

# complete the code to set the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0
target = "ProdTaken" 
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # complete the code: which variable should stay balanced across the splits?
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
