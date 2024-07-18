from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from datasets import Dataset
from ragas.evaluation import evaluate
from pymongo import MongoClient
import json
from typing import Literal
from datetime import datetime
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    answer_relevancy,
    faithfulness, 
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)

def ragas_eval(llm:str, 
               embedding_model:str, 
               hf_token:str,
               model: Literal["phi3", "mistral-7b", "llama3"]):
    
    """
    Evaluates the performance of RAG on a given Dataset.
    Args:
        llm : The language model that will evaluate the results.
        embedding_model : The name of the embedding model to be used.
        hf_token : The Hugging Face token for accessing the embedding model.

    Returns:
        Evaluation Dataset inserted into mongodb
    """
    
    client = MongoClient('mongodb://localhost:27017/')  
    db = client['h2ogpt'] 
    collection = db['Eval'] 
    post_eval_collection = db["PostEval"]

    eval_data = collection.find_one({model: {'$exists': True}}, {"_id": 0})
    dataset_sam = Dataset.from_dict(eval_data[model])

    llm = Ollama(model=llm)
    embeddings = HuggingFaceHubEmbeddings(
        model = embedding_model,
        huggingfacehub_api_token=hf_token
    )

    result = evaluate(
        dataset_sam, #.select(range(1))
        metrics=[context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
                answer_similarity,
                answer_correctness,
                harmfulness],
        llm=llm,
        embeddings=embeddings
    )

    eval_df = result.to_pandas()
    eval_df["Datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    post_eval = eval_df.to_json(orient="index")
    post_eval_dict = json.loads(post_eval)
    
    document = {f"Eval {model}" : post_eval_dict}
    
    #Ajouter ici le management des datasets (update ou insert)
    post_eval_collection.insert_one(document)
        
ragas_eval("phi3","sentence-transformers/all-MiniLM-L6-v2","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","mistral-7b")

