import os
from pathlib import Path
import argparse
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)
from omegaconf import OmegaConf

from tqdm import tqdm
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import SQLDatabase, VectorStoreIndex, PromptTemplate, SimpleDirectoryReader, Settings
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select
)
from functools import partial
import jsonlines
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from testing.base_test import BaseTest
from preprocessing.preprocess import TextToSQLDataSet
from utils import (
    get_llm_model, get_embedding_model, 
    llama_index_pipeline_obj_retriever, 
    llama_index_pipeline_query_executor, 
    llama_index_pipeline_get_table_context_str,
    llama_index_pipeline_parse_response_to_sql,
    llama_index_pipeline_response_prompt
)
from typing import List, Dict, Callable



class TxextToSQLTest(BaseTest):
    def __init__(self, database_path: str, llm_model: str, embed_model: str, 
                llm_init_method: str, 
                dev_file, 
                schema_file, 
                pandas_transform, 
                middle_name,
                embed_init_method='huggingface', output_file="",
                quantization=None
                 ):
        # Key attributes
        self.llm = None
        self.schema = None
        self.embed_model = None
        self.datasets = None
        self.test_data = None
        self.schema = None
        self.model_init = llm_init_method

        self.dev_file = dev_file
        self.schema_file = schema_file
        self.pandas_transform = pandas_transform
        self.middle_name=middle_name
        self.quantization=quantization
        
        self.load_data(database_path)
        self.database_path = database_path
        self.output_file = output_file

        self.load_models(llm_model_name=llm_model,
                        embed_model_path=embed_model, 
                        llm_init_method=llm_init_method,
                        embed_init_method=embed_init_method)

    def load_models(self, llm_model_name, embed_model_path, llm_init_method='huggingface', embed_init_method='huggingface'):
        llm_model = get_llm_model(llm_model_name, llm_init_method, quantization=self.quantization)
        embed_model = get_embedding_model(embed_model_path, embed_init_method)
        Settings.llm = llm_model
        Settings.embed_model = embed_model
        self.llm = llm_model
    
    def load_data(self, database_path):
        self.datasets = TextToSQLDataSet(root_dir=database_path, 
                                    dev_file=self.dev_file, 
                                    schema_file=self.schema_file, 
                                    pandas_transform=self.pandas_transform)
        # test_data, list of dictionary
        self.test_data: List[Dict] = self.datasets.dev
        # schema, dictionary, key is db_id
        self.schema: Dict = self.datasets.schema


    def _pipline_setup(self, 
                        obj_retriever:Callable, 
                        table_schema_parser:Callable, 
                        text2sql_prompt:str, 
                        llm_response_to_sql_parser: Callable
                        ):
        qp = QP(
            modules={
                "input": InputComponent(),
                "table_retriever": obj_retriever,
                "table_output_parser": table_schema_parser,
                "text2sql_prompt": text2sql_prompt,
                "text2sql_llm": self.llm,
                "sql_output_parser": llm_response_to_sql_parser,
                # "sql_retriever": sql_retriever,
                # "response_synthesis_prompt": response_synthesis_prompt,
                # "response_synthesis_llm": llm,
            },
            verbose=False,
        )

        qp.add_chain(["input", "table_retriever", "table_output_parser"])
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser"])
        return qp

    def pipeline_setup(self, db_id, top_k=3):
        # 1. prepare engine
        # engine = create_engine(f'sqlite:///{os.path.join(self.database_path, "database", db_id, db_id+".sqlite")}')
        # (db_schema, db_to_df, engine, table2cols, col2types)
        engine = self.datasets.get_db(db_id=db_id, middle_name=self.middle_name)[2]

        # 2. prepare table names
        table_names: str = self.schema[db_id]['table_names_original']


        DEFAULT_TEXT_TO_SQL_TMPL = (
            "Given an input question, first create a syntactically correct {dialect} "
            "query to run, then look at the results of the query and return the answer. "
            "You can order the results by a relevant column to return the most "
            "interesting examples in the database.\n\n"
            "Never query for all the columns from a specific table, only ask for a "
            "few relevant columns given the question.\n\n"
            "Pay attention to use only the column names that you can see in the schema "
            "description. "
            "Be careful to not query for columns that do not exist. "
            "Pay attention to which column is in which table. "
            "Also, qualify column names with the table name when needed. "
            "You are required to use the following format, each taking one line:\n\n"
            "Question: Directly put the input Question here, any paraphrase and any changes on Question is not allowed\n"
            "SQLQuery: SQL Query to run\n"
            "SQLResult: Result of the SQLQuery\n"
            "Answer: Final answer here\n\n"
            "Only use tables listed below.\n"
            "{schema}\n\n"
            "Question: {query_str}\n"
            "SQLQuery: "
        )
        DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
            DEFAULT_TEXT_TO_SQL_TMPL,
            prompt_type=PromptType.TEXT_TO_SQL,
            )

        # 3. prepare components
        sql_database = SQLDatabase(engine)
        obj_retriever: Callable = llama_index_pipeline_obj_retriever(db_engine=engine, table_names=table_names, retrieve_top_k=top_k)
        table_schema_parser: Callable = FnComponent(fn=partial(llama_index_pipeline_get_table_context_str, sql_database=sql_database)) # table_schema_objs: List[SQLTableSchema], sql_database:SQLDatabase
        text2sql_prompt: PromptTemplate =  DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name, )
        llm_response_to_sql_parser: Callable = FnComponent(fn=partial(llama_index_pipeline_parse_response_to_sql, model_init=self.model_init) ) # response: ChatResponse

        # return QP
        pipeline = self._pipline_setup(obj_retriever=obj_retriever, 
                                table_schema_parser=table_schema_parser,
                                text2sql_prompt=text2sql_prompt,
                                llm_response_to_sql_parser=llm_response_to_sql_parser)
        return pipeline, sql_database

    def run_test_single_db(self, row:Dict, pipeline, sql_database, execute_on_db=False):
        # code 999 N/A
        # code -999 syntax error
        # code 1, executable
        predict_query:str = pipeline.run(query=row['question'])
        execute_output=None
        execute_status = 999 # not applicable
    
        if execute_on_db:
            sql_retriever: Callable = llama_index_pipeline_query_executor(sql_database)
            try:
                execute_output = sql_retriever(predict_query)
                execute_status = 1
            except:
                execute_output = None
                execute_status = -999 # Syntax Error            
        return predict_query, execute_status, execute_output


        
    def run_test(self, dump=True, execute=False, limit=None):
        # Run test on wiki dataset
        # self.test_data
        pre_db_id = ""
        output_query_records = []
        qp = None
        if not execute:
            self.row_cnt = -999
            self.syntax_error_cnt = -999
        else:
            self.row_cnt = 0
            self.syntax_error_cnt = 0  
        
        if limit:
            self.test_data = self.test_data[:limit]

        for index, row in enumerate(tqdm(self.test_data)):
            if execute:
                self.row_cnt += 1

            db_id = row['db_id']
            if index == 0:
                pre_db_id = db_id
                qp, sql_database = self.pipeline_setup(db_id)
            else:
                if pre_db_id != db_id:
                    # time to setup a new pipeline
                    pre_db_id=db_id
                    qp, sql_database = self.pipeline_setup(db_id) # TBD

            predict_query, execute_status, execute_output = self.run_test_single_db(row, qp, sql_database, execute_on_db=execute)
            if execute:
                if execute_status == -999:
                    self.syntax_error_cnt += 1
            _record = {"db_id": db_id, 'predict_query':predict_query, 'question':row['question']}
            output_query_records.append(_record)
        output_query_records.insert(0, {'row_cnt':self.row_cnt, 'syntax_error_cnt': self.syntax_error_cnt})
        if dump:
            with jsonlines.open(self.output_file, "w") as writter:
                writter.write_all(output_query_records) 
        return output_query_records

    def report_ex(self):
        pass

def unit_test_on_bird_dev():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_on_birddev.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

def unit_test_on_wiki_sql():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_on_wikisql.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

def unit_test_on_dbqa():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_on_dbqa.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

def main(args):
    # mkdirs
    dir_name = os.path.dirname(args.predictions)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    sql_test = TxextToSQLTest(database_path=args.database_path, 
                        llm_model=args.llm_model, 
                        embed_model=args.embedding_model, 
                        llm_init_method=args.llm_init_method, 
                        dev_file=args.dev_file, 
                        schema_file=args.schema_file, 
                        middle_name=args.middle_name,
                        pandas_transform=args.pandas_transform,
                        embed_init_method='huggingface',
                        output_file=args.predictions,
                        quantization=args.llm_quantization
                        )
    
    output_query_records = sql_test.run_test(limit=args.limit_lines)

if __name__ == "__main__":
    # unit_test_on_bird_dev()
    # unit_test_on_wiki_sql()
    # unit_test_on_dbqa()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_on_birddev.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
        

        
