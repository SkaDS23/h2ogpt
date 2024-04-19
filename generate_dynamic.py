from pymongo import MongoClient
from src.utils_sys import protect_stdout_stderr
from src.gen import main

def get_parameters_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['h2ogpt']
    collection = db['params_config']
    parameters_document = collection.find_one()
    
    if parameters_document:
        parameters_document.pop('_id', None)
    return parameters_document

def entrypoint_main():
    parameters_document = get_parameters_from_mongodb()
    parameters = {}  
    if parameters_document:
        parameters = parameters_document  

    main(**parameters)

if __name__ == "__main__":
    protect_stdout_stderr()
    entrypoint_main()
