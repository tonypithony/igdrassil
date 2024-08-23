# source VISTREE/bin/activate

# pip install ipython scikit-learn pydotplus pandas numpy==1.26
# sudo apt-get install graphviz

# Load libraries
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
import pandas as pd
import numpy as np

# Load data
taxi_csv = 'taxis.csv'

dataframe = pd.read_csv(taxi_csv)

print(dataframe.head(), '\n')
# print(dataframe.shape, '\n')
print(dataframe.info(), '\n')
print(dataframe.describe(), '\n')

categorical_features = ['pickup',
						'dropoff',
						'color',
						'pickup_zone',
						'dropoff_zone',
						'pickup_borough',
						'dropoff_borough',]

dataframe = dataframe.drop(categorical_features, axis=1)

dataframe = dataframe.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))

target = 'payment'
features = dataframe.drop(target, axis=1)
target = dataframe[target]

# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)
# Train model
model = decisiontree.fit(features, target)
# Create DOT data
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=features.columns.tolist(),
                                # class_names=target.index.to_list()
                                class_names=np.array(sorted(target.index.unique())).astype('str').tolist())
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
Image(graph.create_png())

# Create PNG
graph.write_png("taxi-tree.png")

# Create PDF
graph.write_pdf("taxi-tree.pdf")
