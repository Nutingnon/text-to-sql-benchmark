from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select
)
from omegaconf import OmegaConf
import os
import jsonlines
import json
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.retrievers import SQLRetriever
from typing import List, Dict, Callable
from llama_index.core.query_pipeline import FnComponent

from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.llms import ChatResponse
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch

# https://github.com/huggingface/transformers/blob/main/docs/source/zh/main_classes/quantization.md
# bf16
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
#                                         bnb_4bit_compute_dtype=torch.bfloat16,
#                                         )
# from official colab: https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf#scrollTo=WQ-BLtJG9b38
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#     # load_in_8bit=True
# )



quantization_dict = dict()
quantization_dict['nf4'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    # load_in_8bit=True   
)
quantization_dict['8bit'] = BitsAndBytesConfig(
    load_in_8bit=True
)



def get_llm_model(model_name, init_method='huggingface', quantization=None):
    # llm =Ollama(model=LLM_MODEL, request_timeout=360.0, device='cuda')
    if init_method == 'ollama':
        model = Ollama(model=model_name, request_timeout=360.0, device='cuda')

    elif init_method in ['huggingface', 'hugging_face']:
        if "GLM" in model_name:
            model = HuggingFaceLLM(
                model_name=model_name,
                tokenizer_name=model_name,
                context_window=10_000, # Default is set to 3900 ,chatglm-4 10k, mistral-7b 5k
                max_new_tokens=100,
                # generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95}
                device_map = "auto",
                model_kwargs={"trust_remote_code":True,
                            }
            )
        else:
            model = HuggingFaceLLM(
                model_name=model_name,
                tokenizer_name=model_name,
                context_window=5000,
                max_new_tokens=100,
                device_map="auto",
                model_kwargs={"quantization_config":quantization_dict.get(quantization, None)}
            )
    return model

def get_embedding_model(model_path, init_method='huggingface'):
    # Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH, device="cuda")

    if init_method=='huggingface':
        embed_model = HuggingFaceEmbedding(model_name=model_path, device='cuda')
    else:
        pass 
    return embed_model

# Query -> [obj_index] 
# to retrive the matched objects
def llama_index_pipeline_obj_retriever(db_engine, table_names:List[str], retrieve_top_k=3):
    "DB engine is the engine object initialized by SQLAlchemy"
    sql_database = SQLDatabase(db_engine)
    table_node_mapping = SQLTableNodeMapping(sql_database)

    table_schema_objs = [
        SQLTableSchema(table_name=t)
        for t in table_names
    ]
    # add a SQLTableSchema for each table

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=retrieve_top_k)
    return obj_retriever


def llama_index_pipeline_query_executor(sql_database:SQLDatabase):
    """
    This function is responsible for retrieving results from database according to the input query.
    Args:
        sql_database (SQLDatabase): the one initalized with engine.
        For example:
            sql_database = SQLDatabase(engine)
    """
    sql_retriever = SQLRetriever(sql_database)
    return sql_retriever


#  table_parser_component = FnComponent(fn=get_table_context_str)
def llama_index_pipeline_get_table_context_str(table_schema_objs: List[SQLTableSchema], sql_database:SQLDatabase):
    """Get table context string."""
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(
            table_schema_obj.table_name
        )
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context
        context_strs.append(table_info)
    return "\n\n".join(context_strs)

# def llama_index_pipeline_get_table_context_str():
#     return FnComponent(fn=_llama_index_pipeline_get_table_context_str)


# sql_parser_component = FnComponent(fn=parse_response_to_sql)
def llama_index_pipeline_parse_response_to_sql(response: ChatResponse, model_init='huggingface') -> str:
    """Parse response from LLM to excutable SQL query."""
    if model_init == "huggingface":
        response=response.text
    elif model_init == 'ollama':
        response=response.message.content

    # print(response)
    # print("==="*50)
    
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
            
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip()

# def llama_index_pipeline_parse_response_to_sql():
#     return FnComponent(fn=_llama_index_pipeline_parse_response_to_sql)



def llama_index_pipeline_response_prompt() -> PromptTemplate:
    # Response Synthesis Prompt
    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt = PromptTemplate(
        response_synthesis_prompt_str,
    )
    return response_synthesis_prompt


def read_json_jsonl_file(file_path: str)-> List[Dict]:
    if file_path.endswith(".jsonl"):
        with jsonlines.open(file_path, 'r') as jsonl_f:
            file = [obj for obj in jsonl_f]
    elif file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            file = json.load(f)
    else:
        raise TypeError(f"Unknown format of input file: {file_path}")
    
    return file