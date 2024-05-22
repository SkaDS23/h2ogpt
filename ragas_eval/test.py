from pymongo import MongoClient
from datasets import Dataset, load_dataset
import json

client = MongoClient('mongodb://localhost:27017/')  
db = client['h2ogpt'] 
collection = db['Eval'] 

document_from_mongo = collection.find_one({}, {"_id": 0})  

python_dict = document_from_mongo

dataset_sam = Dataset.from_dict(python_dict)

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")

print(dataset_sam.features)
print("------")
print(amnesty_qa['eval'].features)
