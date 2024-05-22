from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from datasets import Dataset
from ragas.evaluation import evaluate
from pymongo import MongoClient
from ragas.metrics import (
    answer_relevancy,
    faithfulness, 
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)
from ragas.metrics.critique import harmfulness

client = MongoClient('mongodb://localhost:27017/')  
db = client['h2ogpt'] 
collection = db['Eval'] 

data1 = collection.find_one({}, {"_id": 0})
data2 = data1
dataset_sam = Dataset.from_dict(data2)

llm = Ollama(model = "phi3")
embeddings = HuggingFaceHubEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token="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"
)

result = evaluate(
    dataset_sam,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_similarity,
        answer_correctness,
        harmfulness
    ],
    llm=llm,
    embeddings=embeddings
)

eval_df = result.to_pandas()
eval_df.to_csv("Eval.csv")