import os
from threading import main_thread
from pyspark.context import SparkContext
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.session import SparkSession
import pandas as pd
import numpy as np
from pyarrow import parquet as pq

def spark_session() -> SparkSession:
    os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'
    return SparkSession.builder.master('local[4]') \
                               .config("spark.sql.execution.arrow.enabled", "true") \
                               .getOrCreate()

names = pd.array(['Tom', 'Pete', 'Elsa', 'Ben', 'Judy'])

def create_user_info() -> pd.DataFrame:
    num_of_users = len(names)
    return pd.DataFrame({
        'id': np.arange(1, num_of_users + 1),
        'user': names,
        'age': np.random.randint(10,50,size=num_of_users)
    })

def create_dataset(num_of_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        'transaction_group': np.random.randint(0,100,size=num_of_rows),
        'user': pd.array(np.random.choice(names, num_of_rows), dtype='string'),
        'cost': np.random.randint(0,1000,size=num_of_rows)/10.0,
    })


if __name__ == '__main__':
    transaction_data = create_dataset(1000)
    user_info = create_user_info()

    print(transaction_data.merge(user_info, how='left'))

    spark = spark_session()

    sc: SparkContext = spark._sc
    user_info_bc = sc.broadcast(user_info)

    idf = spark.createDataFrame(user_info)

    tdf = spark.createDataFrame(transaction_data) \
               .repartition(16, 'transaction_group')

    return_type = {}
    
    for field in tdf.schema.jsonValue()['fields']:
        name = field['name']
        if name not in return_type:
            return_type[name] = field['type']
    
    for field in idf.schema.jsonValue()['fields']:
        name = field['name']
        if name not in return_type:
            return_type[name] = field['type']
        
    return_type_string = ', '.join('{name} {type_name}'.format(name=field, type_name=return_type[field]) for field in return_type) 
    print(return_type_string)

    @pandas_udf(return_type_string, PandasUDFType.GROUPED_MAP)
    def merge_dataframe(pdf: pd.DataFrame) -> pd.DataFrame:
        info_df: pd.DataFrame = user_info_bc.value
        result = pdf.merge(info_df, how='left')
        print(result)
        return result
    
    tdf.groupBy('transaction_group').apply(merge_dataframe).write.parquet('file:///tmp/transaction_group.parquet')
    print(pq.read_table('file:///tmp/transaction_group.parquet').to_pandas())