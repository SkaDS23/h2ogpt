from enum import Enum
import ast
from pprint import pprint
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')  
db = client['h2ogpt'] 
collection = db['Eval'] 

pre_prompt_query="Pay attention and remember the information below, which will help to answer the question or imperative after the context ends."
prompt_query="According to only the information in the document sources provided within the context above, write an insightful and well-structured response only and only in french to:"
system_prompt="A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions only and only in french even if the question is in another langage"

class LangChainAction(Enum):
    """LangChain action"""
    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"

def save_data_db1(prompt, response, ground_truth):
    '''
    Helper function that Saves question / answer and context to database.
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
        
        extracted_text = content[start_index:end_index].replace('"""','').strip()
        context.append(extracted_text)

    collection.insert_one({
    'question': [prompt],
    'ground_truth': [[ground_truth]],
    'answer': [Answer],
    'contexts': [context]})

def save_data_db2(prompt, response, ground_truth):
    '''
    Helper function that Saves question / answer and context to database.
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
        
    existing_document = collection.find_one({})
    
    if existing_document:
        collection.update_one({}, {"$push": {
        "question": prompt,
        "ground_truth" : ground_truth,
        "answer" : Answer,
        'contexts' : context 
         }})
        
    else:
        collection.insert_one({
        'question': [prompt],
        'ground_truth': [ground_truth],
        'answer': [Answer],
        'contexts': [context]
         })
    
    
def print_full_model_response(response):
    '''
    Helper function to print full response from the h2oGPT call, including all parameters.
    '''
    print("Model Response with Parameters:\n")
    save_dict = ast.literal_eval(response)['save_dict']
    pprint(save_dict)
    print("\n")
    try:
        sources = ast.literal_eval(response)['sources']
        print("Sources:\n")
        pprint(sources)
        print("\n")
    except:
        print("No sources\n")



