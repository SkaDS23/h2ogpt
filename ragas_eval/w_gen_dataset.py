from gradio_client import Client
from w_eval_utils import *

"""
-----This script generates a dataset with Quesiton / Answer and context and exports it to a mongodb database.----

-Launch h2ogpt on your server (and change server address in HOST variable below)
-Assume to create the collection and load the documents from UI (shared collection) and change the name of the collection below (langchain_mode)
-Configure the mongodb parameters below according to the mongo credentials (server_name/ port ..)
-This is a doc/qa program over all of the documents in the collection (cannot chose document)
-Do not forget to adjust params with your needs (pre-prompt, prompt, chunking, top_k_docs ....)
-Do not forget to annotate ground_truth manually in the database
-Some params (like pre-prompt query and prompt query are defined in the w_eva_utils.py)
"""

HOST = "http://localhost:7860/"
client = Client(HOST)

#NOTE: list of all params : https://github.com/h2oai/h2ogpt/blob/main/gradio_utils/grclient.py#L40

def main():
    prompt = ""
    ground_truth = ""
    i = 1
    while True:
        prompt = input(f"{i}) Ask (enter q to quit): ")
        if prompt == "q":
            break
        else:
            ground_truth = input("Ground Truth : ")
            kwargs = dict(instruction_nochat=prompt,
                        langchain_mode="UserData",
                        langchain_action = LangChainAction.QUERY.value,
                        top_k_docs=3,
                        pre_prompt_query=pre_prompt_query,
                        prompt_query=prompt_query,
                        system_prompt=system_prompt,
                        document_subset="Relevant",
                        temperature=0.1, 
                        chunk=True,
                        chunk_size=512     
                        )

            res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
            save_data_db2(prompt=prompt, response=res, ground_truth = ground_truth)
            i = i+1
        
    
if __name__ == "__main__":
    main()