from pymongo import MongoClient

def store_parameters_in_mongodb(parameters):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['h2ogpt']
    collection = db['params_config']

    collection.insert_one(parameters)

parameters = {
    'system_prompt' : 'auto',
    'context' : None,
    'chat_conversation' : None,
    'iinput' : '',
    'pre_prompt_query' : None,
    'prompt_query' : None,
    'pre_prompt_summary' : None,
    'prompt_summary' : None,
    'hyde_llm_prompt' : None,
    'llava_prompt' : None,
    'top_k_docs' : None,
    'chunk' : True,
    'chunk_size' : 512,
    'docs_ordering_type' : "best_near_prompt",
    'hyde_level' : 0,
    'hyde_template' : None,
    'hyde_show_only_final' : False,
    'docs_token_handling' : "split_or_merge",
    'doc_json_mode' : False,
    'stream_output' : True,
    'max_time': None,
    'temperature': None,
    'top_p': None,
    'top_k': None,
    'penalty_alpha': None,
    'max_output_seq_len' : None,
    'repetition_penalty': None,
}

store_parameters_in_mongodb(parameters)