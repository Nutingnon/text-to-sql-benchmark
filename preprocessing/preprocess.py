import pandas as pd
import json
import jsonlines
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    Numeric,
    Float,
    Text,
    REAL,
    DATE,
    DATETIME
)
from typing import List, Tuple, Dict
import sqlalchemy as sa
import os


# DEV_FILE='mini_dev_sqlite.json'
# SCHEMA_FILE = 'dev_tables.json'
# PANDAS_TRANSFORM=False



class TextToSQLDataSet(object):
    def __init__(self, root_dir, dev_file, schema_file, pandas_transform) -> None:
        self.root_dir = root_dir
        self.dev_file = dev_file
        self.schema_file=schema_file
        self.pandas_transform=pandas_transform
        self._parse_structure()

    def _parse_structure(self):
        if self.dev_file.endswith(".jsonl"):
            print(self.root_dir, self.dev_file)
            with jsonlines.open(os.path.join(self.root_dir , self.dev_file), "r") as jsonl_f:
                self.dev = [obj for obj in jsonl_f]
                # sort
                # self.dev = sorted(self.dev, key=lambda x:x['db_id'])
        elif self.dev_file.endswith(".json"):
            with open(os.path.join(self.root_dir, self.dev_file), "r") as jsonfile:
                self.dev = jsonfile.read()
                self.dev = json.loads(self.dev)

        with open(os.path.join(self.root_dir , self.schema_file), 'r') as jsonfile:
            schema_file = jsonfile.read()
        
        schema = json.loads(schema_file)
        self.schema = dict()
        for s in schema:
            db_id = s["db_id"]
            self.schema[db_id] = s
    

    def get_db(self, db_id, middle_name=''):
        # get schema of db
        db_schema, table_names, col2types, table2cols, table_pk, table_fk = self._get_schema_of_db(db_id)
        if self.pandas_transform:
            db_to_df: Dict[str, pd.DataFrame] = self._get_dataframe_from_db(db_id, table_names=table_names)
            engine = TextToSQLDataSet.create_table_from_database(table_name_to_df=db_to_df, table_names=table_names, column_names_dict=table2cols, column_types_dict=col2types)
        else:
            db_path = os.path.join(self.root_dir, middle_name, db_id, db_id+".sqlite")
            engine = create_engine(f"sqlite:///{db_path}")
            metadata = MetaData()
            metadata.reflect(bind=engine)
            db_to_df = None
        return (db_schema, db_to_df, engine, table2cols, col2types)
    

    def _get_schema_of_db(self, db_id):
        db_schema = self.schema.get(db_id, None)
        table_names: List[str] = db_schema['table_names_original'] # wikisql only have one table in each db
        col_types:List[str] = db_schema['column_types']
        primary_keys: List = db_schema['primary_keys']
        foreign_keys: List = db_schema["foreign_keys"]
        table2cols: Dict = dict()
        col2types: Dict = dict()
        col_idx_to_tables_idx: Dict = dict()
        # Table2cols, col2type
        pre_table_id = 0
        col_cnt = 0
        for table_id, col_name in db_schema['column_names_original']:
            if int(table_id) == -1: # -1 colname: *, coltype: text, currently I don't know what it is used for
                col2types[col_name] = col_types[col_cnt]
                col_cnt += 1
                continue

            elif table_id !=  pre_table_id:
                pre_table_id = table_id
                table2cols[table_names[table_id]] = [col_name]

            else:
                if table_names[table_id] in table2cols.keys():
                    table2cols[table_names[table_id]].append(col_name)
                else:
                    table2cols[table_names[table_id]] =[col_name]
                
            col2types[col_name] = col_types[col_cnt]
            col_idx_to_tables_idx[col_cnt]=table_id
            col_cnt += 1

        # Table to FK PK
        table_pk: Dict[str, List] = dict()
        table_fk: Dict[str, List] = dict()

        for idx, pk_col_idx in enumerate(primary_keys):
            if type(pk_col_idx)==int:
                table_pk[table_names[idx]] = [db_schema['column_names_original'][pk_col_idx][1]]
            # more than one pk
            elif len(pk_col_idx)>1:
                table_pk[table_names[idx]] = [db_schema['column_names_original'][sub_pk_idx][1] for sub_pk_idx in pk_col_idx]
        
        for fk_col_table_a_idx, link_col_table_b_idx in foreign_keys:
            # table_fk['table_a'] = [("col1", 'table_b', 'col1'),
            #   ("col2","table_c", "col2"),
            #   ("col3", "table_d", "col3")
            #]
            try:
                fk_info = (db_schema['column_names_original'][fk_col_table_a_idx][1],
                            table_names[col_idx_to_tables_idx[link_col_table_b_idx]],
                            db_schema['column_names_original'][link_col_table_b_idx][1]
                            )
                if table_names[col_idx_to_tables_idx[fk_col_table_a_idx]] not in table_fk:
                    table_fk[table_names[col_idx_to_tables_idx[fk_col_table_a_idx]]] = [fk_info]
                else:
                    table_fk[table_names[col_idx_to_tables_idx[fk_col_table_a_idx]]].append(fk_info)

            except:
                print("DB ID:", db_id, 
                    "\nIDx:", idx,
                    "\nTable Names:", table_names,
                    "\nfk_col_idx:", [fk_col_table_a_idx, link_col_table_b_idx],
                    "\ncol_idx_to_tables:", col_idx_to_tables_idx,
                    "\nTable_FK:", table_fk)
                raise IndexError("list index out of range")
                

        # TBD: Add Table, Column description
        return db_schema, table_names, col2types, table2cols, table_pk, table_fk

    def _get_dataframe_from_db(self, db_id: str, table_names: List[str], middle_name="dev_databases") -> Dict[str, pd.DataFrame]:
        db_dir = os.path.join(self.root_dir, middle_name, db_id, db_id+".sqlite")
        # Create an engine that connects to the SQLite database
        engine = create_engine(f'sqlite:///{db_dir}')

        # change column type if necessary
        table_to_df_dict: Dict[str, pd.DataFrame] = dict()
        with engine.begin() as conn:
            for table_name in table_names:
                # You must add the backslash ` ` to table name
                # or it will raise SyntaxError once the table name is 
                # same as the Keyword in SQL
                qry = sa.text(f"SELECT * FROM `{table_name}`;")
                resultset = conn.execute(qry)
                results_as_dict = resultset.mappings().all()
                df = pd.DataFrame(results_as_dict, dtype=str)
                table_to_df_dict[table_name] = df
        return table_to_df_dict


    # @staticmethod
    def create_table_from_database(
        table_name_to_df: Dict[str, pd.DataFrame], 
        table_names: List[str], 
        column_names_dict: Dict[str, List],
        column_types_dict: Dict[str, str]
    ):

        metadata_obj = MetaData()
        engine = create_engine("sqlite:///:memory:")
        for table_name in table_names:
            df: pd.DataFrame = table_name_to_df[table_name]
            column_types = [column_types_dict[col] for col in column_names_dict[table_name]]
            column_names = column_names_dict[table_name]
            create_table_from_dataframe(df, table_name, engine, metadata_obj, column_names, column_types)

        return engine


def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj, column_names, column_types
):
    # Dynamically create columns based on DataFrame columns and data types
    dtype_transformer = {"text":Text, "number":Integer, "float":Float, 
                        "numeric":Numeric, "real":REAL, "integer":Integer, 
                        "date":DATE, "datetime": DATETIME}

    columns = [
        Column(col, dtype_transformer[dtype.lower()])
        for col, dtype in zip(column_names, column_types)
    ]
    
    # Correct data if date type in it
    for idx, col in enumerate(column_names):
        if column_types[idx] in ['date','datetime']:
            try:
                if len( (df[col][pd.notnull(df[col])]).iloc[0] ) == 10:
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

                elif len( (df[col][pd.notnull(df[col])]).iloc[0] )>=19:
                    df[col]=df[col].apply(lambda x: x[:19] if pd.notnull(x) else None)
                    df[col]=pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')

            except:
                print(df[col])
                raise TypeError

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)    

    # Create the table in the database
    metadata_obj.create_all(engine)
    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():

            # convert nan to None
            for rk in row.keys():
                if pd.isnull(row[rk]):
                    row[rk]=None

            try:
                insert_stmt = table.insert().values(**row.to_dict())
                conn.execute(insert_stmt)
            except:
                print(row.to_dict())
                raise ValueError("Problem on this row")
        conn.commit()

def unit_test_bird():
    db_id1 = "california_schools"
    db_id2 = "card_games"
    root_dir = "/home/yixin/work/msxf/Text2SQL_Exp/datasets/minidev/MINIDEV"
    bird_dataset = TextToSQLDataSet(root_dir=root_dir)

    # (db_schema, db_to_df, engine, table2cols, col2types)
    db1_result = bird_dataset.get_db(db_id1, middle_name='dev_databases')
    db2_result = bird_dataset.get_db(db_id2, middle_name='dev_databases')
    
    engine1 = db1_result[2]
    engine2 = db2_result[2]

    with engine1.connect() as conn1:
        qry = sa.text(f"SELECT * FROM `{db1_result[0]['table_names_original'][0]}` limit 5;")
        resultset = conn1.execute(qry)
        results_as_dict = resultset.mappings().all()
    print(results_as_dict)

    with engine2.connect() as conn2:
        qry = sa.text(f"SELECT * FROM `{db2_result[0]['table_names_original'][0]}` limit 5;")
        resultset2 = conn2.execute(qry)
        results_as_dict2 = resultset2.mappings().all()
    print(results_as_dict2)


def unit_test_wiki_sql():
    pass 
def unit_test_dbqa():
    pass

if __name__ == "__main__":
    unit_test_bird()

