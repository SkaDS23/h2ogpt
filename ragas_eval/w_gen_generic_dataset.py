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
    Questions = []
    Ground_truthts = []
    
    fetch_db_data(Questions, Ground_truthts)
    
    i = 1
    for question, ground_truth in zip(Questions, Ground_truthts):
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

        res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        model = get_model_name(res)
        save_data_db(prompt=prompt, response=res, ground_truth=ground_truth, model=model)
        print(f"Question {i} generated successfully")
        i = i+1
    
if __name__ == "__main__":
    main()