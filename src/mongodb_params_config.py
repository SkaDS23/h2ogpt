from pymongo import MongoClient

def store_parameters_in_mongodb(parameters):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['h2ogpt']
    collection = db['params_config']

    collection.insert_one(parameters)

parameters = {
    'temperature': None,
    'top_p': None,
    'top_k': None,
    'penalty_alpha': None,
    'num_beams': None,
    'repetition_penalty': None,
    'num_return_sequences': None,
    'do_sample': None,
    'seed': None,
    'max_new_tokens': None,
    'min_new_tokens': None,
    'early_stopping': None,
    'max_time': None,
    'max_seq_len' : None,
    'max_output_seq_len' : None,
    "hyde_level" : 0,
    "pre_prompt_query" : "Test Query for testing database",
    "prompt_query" : None,
    "pre_prompt_summary" : None,
    "prompt_summary" : None,
    "hyde_llm_prompt" : None,
    "prompt_type" : None,
    "system_prompt" : None,
}

store_parameters_in_mongodb(parameters)