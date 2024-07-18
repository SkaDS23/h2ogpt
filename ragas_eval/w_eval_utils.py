from gradio_client import Client
import pandas as pd
from enum import Enum
import ast
import json
from pymongo import MongoClient
from typing import Literal
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from datasets import Dataset
from datetime import datetime
from ragas.evaluation import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    answer_relevancy,
    faithfulness, 
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness,
)

''' This Module Contains all the necessary functions to evaluate the RAG Pipeline Using Ragas Library 

    There is a total of 2 main functions : 
    
    1) Gen_Eval_Dataset (generates the evaluation dataset dynamically through API calls)
    2) Ragas_Eval (Evaluates the rag pipeline based on the chosen dataset)
    
    And a Total of 6 Helper functions '''

def get_client():
    H2OGPT_URL = "http://localhost:7862/"
    h2oclient = Client(H2OGPT_URL, auth=("Expert", "Expert"))
    
    return h2oclient

client = MongoClient('mongodb://localhost:27017/')  
db = client['h2ogpt'] 
PreEvalCollection = db['PreEval'] 
EvalCollection = db['Eval'] 
PostEvalCollection = db["PostEval"]

class PromptTemplates(Enum):
    '''Prompt templates'''
    
    PRE_PROMPT_QUERY = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends."
    PROMPT_QUERY = "According to only the information in the document sources provided within the context above, write an insightful and well-structured response only and only in french to:"
    SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions only and only in french even if the question is in another langage"

class LangChainAction(Enum):
    '''LangChain action'''
    
    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"
    
def print_full_model_response(response):
    '''
    function to print full response from the h2oGPT call, including all parameters.
    '''
    
    print("Model Response with Parameters:\n")
    save_dict = ast.literal_eval(response)['save_dict']
    print(save_dict)
    print("\n")
    try:
        sources = ast.literal_eval(response)['sources']
        print("Sources:\n")
        print(sources)
        print("\n")
    except:
        print("No sources\n")

def fetch_db_data(questions:list, ground_truths:list):
    ''' Helper Function That fetches data from the Pre Eval collection (Questions and Ground Truths).
        This Function is called in gen_eval_dataset'''
    
    documents = PreEvalCollection.find()
    
    for document in documents:
        questions.append(document.get('Question'))
        ground_truths.append(document.get('Ground_truth'))
        
def get_model_name(response):
    ''' Helper Function that fetches the model name from the Api Call.
        Thus function is used in the gen_eval_dataset function.
    
        endpoint : /submit_nochat_api '''
        
    Model_name = ast.literal_eval(response)['save_dict']['base_model']
    return Model_name

def get_model_name_h2ogpt():
    return "phi3"
def get_model_name_ragas():
    ''' Helper Function that fetches the model name from the Api Call for dynamic model name generation for the ragas dataset. 
        It uses another api call for time optimization.
        This function is used in the ragas_eval function.
        
        Endpoint : /modl_names '''
        
    h2oclient = get_client()
    
    res = h2oclient.predict(api_name='/model_names')
    Model = ast.literal_eval(res)[0]['base_model']
    return Model
        
def gen_eval_dataset():
    ''' Function that generates the evaluation dataset for the ragas evaluation.
        This function can take several params (same Expert params as in the main function in H2O) '''
    
    h2oclient = get_client()
    
    questions = []
    ground_truths = []
    
    fetch_db_data(questions, ground_truths)
    
    i = 1
    for question, ground_truth in zip(questions, ground_truths):
        prompt = question
        ground_truth = ground_truth
        
        kwargs = dict(instruction_nochat=prompt,
                    langchain_mode="UserData",
                    langchain_action = LangChainAction.QUERY.value,
                    top_k_docs=3,
                    pre_prompt_query=PromptTemplates.PRE_PROMPT_QUERY.value,
                    prompt_query=PromptTemplates.PROMPT_QUERY.value,
                    system_prompt=PromptTemplates.SYSTEM_PROMPT.value,
                    document_subset="Relevant",
                    temperature=0.1, 
                    chunk=True,
                    chunk_size=512     
                    )
        try:
            res = h2oclient.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
            save_data_db(prompt=prompt, response=res, ground_truth=ground_truth, model="phi3")  
            #print(f"Question {i} generated successfully")
            #i = i+1
        except Exception:
            print("Please ensure that the app is running with a given model.")
            break
        
        print("Dataset Generated Successfully  (if no exception occured)")
def save_data_db(prompt: str, response, ground_truth: str, model: str):
    '''
    Helper function that Saves question / Gorund_truth / answer and context to the Eval Collection in the database, used in gen_eval_dataset.
    '''
    
    prompt = prompt
    Answer = ast.literal_eval(response)['save_dict']['output'].replace('"""','')
    sources = ast.literal_eval(response)['save_dict']['sources']
    context=[]
    for source in sources:
        content = source['content']
        start_marker = "Document Contents:"
        end_marker = "End Document"
        
        start_index = content.find(start_marker) + len(start_marker)
        end_index = content.find(end_marker)
        
        extracted_text = content[start_index:end_index].replace('"""','').replace('\n', ' ').replace('\t', ' ').strip()
        context.append(extracted_text)

    update_query = {
        "$push": {
            f"{model}.question": prompt,
            f"{model}.ground_truth": ground_truth,
            f"{model}.answer": Answer,
            f"{model}.contexts": context
        }
    }
    
    existing_document = EvalCollection.find_one({model: {"$exists": True}})
    
    if existing_document:
        EvalCollection.update_one({model: {"$exists": True}}, update_query)
    else:
        new_document = {
            model: {
                "question": [prompt],
                "ground_truth": [ground_truth],
                "answer": [Answer],
                "contexts": [context]
            }
        }
        EvalCollection.insert_one(new_document)
    
def ragas_eval(llm:str, 
               embedding_model:str, 
               hf_token:str):
    
    '''
    Function that Evaluates the performance of RAG on a given Dataset.
    Args:
        llm : The language model that will evaluate the results.
        embedding_model : The name of the embedding model to be used.
        hf_token : The Hugging Face token for accessing the embedding model.

    Returns:
        Evaluation Dataset inserted into mongodb
    '''

    model = get_model_name_ragas()
    
    
    try:
        eval_data = EvalCollection.find_one({model: {'$exists': True}}, {"_id": 0})
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
        save_ragas_eval_dataset(eval_df, model)
        print("Evaluation Finished Succefully !")
        
    except Exception:
        print("Please check the Database collection of the given model, ensure that the model is running with Ollama and th HF Token is Valid.")
    
def save_ragas_eval_dataset(eval_df, model):
    '''Helper Function that formats and saves the final evaluation dataset to the PostEval Collection in the database.
       This function is called in the ragas_eval function. '''
    
    eval_df["Datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    post_eval = eval_df.to_json(orient="index")
    post_eval_dict = json.loads(post_eval)
    
    document = {model : post_eval_dict}
        
    existing_document = PostEvalCollection.find_one({f"Eval {model}" : {'$exists': True}})
    
    if existing_document:
        PostEvalCollection.update_one(
            {f"Eval {model}": {'$exists': True}},
            {"$set": document}
        )
    else:
        PostEvalCollection.insert_one(document)
    

def fetch_data_eval():
    """ Function to display eval dataset in the Eval UI Section """
    
    documents = PreEvalCollection.find({})
    
    data = []
    for doc in documents:
        question = doc.get('Question')
        ground_truth = doc.get('Ground_truth')
        data.append({"question": question, "ground_truth": ground_truth})

    df = pd.DataFrame(data)
    return df

def fetch_data_eval2():
    """ Helper section for the CRUD Operations in Eval UI Section """
    
    cursor = PreEvalCollection.find({}, {"_id": 0})  
    users_df = pd.DataFrame(list(cursor))
    return users_df
def clear_data_eval():
        return pd.DataFrame(columns=['Questions', 'Ground Truths'])
    
def update_gr_eval(question, ground_truth):
    PreEvalCollection.update_one({"Question": question}, {"$set": {"Ground_truth": ground_truth}})
    
    text_message = "Ground truth has been updated succesfully !"
    return text_message

def insert_qs_gr_eval(question, ground_truth):
    PreEvalCollection.insert_one({"Question": question, "Ground_truth": ground_truth})
    
    text_message = "Data has been inserted succesfully !"
    return text_message

def del_qs_gr_eval(question):
    PreEvalCollection.delete_one({"Question": question})
    
    text_message = "Data has been deleted succesfully !"
    return text_message