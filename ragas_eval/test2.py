# Load necessary libraries
from pymongo import MongoClient
from datasets import Dataset, DatasetDict, load_dataset
import json

# Connect to MongoDB and load data
client = MongoClient('mongodb://localhost:27017/')  
db = client['h2ogpt'] 
collection = db['Eval'] 

data = list(collection.find())

# Remove MongoDB's _id field
for item in data:
    item.pop('_id', None)

# Write data to a JSON file
with open('eval.json', 'w') as f:
    json.dump(data[0], f, indent=4)

# Load the JSON file into a dataset
dataset = Dataset.from_json('eval.json')

# Wrap the dataset in a DatasetDict with key 'eval'
dataset_dict = DatasetDict({'eval': dataset})

# Print the updated dataset
print(dataset_dict)
