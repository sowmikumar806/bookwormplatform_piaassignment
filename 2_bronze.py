# Databricks notebook source
# MAGIC %md
# MAGIC Bronze notebook
# MAGIC
# MAGIC -Add _ingestion_timestamp
# MAGIC
# MAGIC -Add _source_file name
# MAGIC
# MAGIC -Store in Delta format
# MAGIC
# MAGIC Auto Loader :checkpoint to detect new file and metadata to check file change
# MAGIC   -Incremental file detection
# MAGIC   -schema inference and timetravel
# MAGIC   -retry with checkpoint
# MAGIC   -audit column for montiroing
# MAGIC
# MAGIC Delta Lake time travel  : For data summary

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta
import json
import boto3

# COMMAND ----------

AWS_ACCESS_KEY = "AKIAXCB7QKFFGNOP2BGI"
AWS_SECRET_KEY = "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"
AWS_REGION     = "eu-north-1"
S3_BUCKET      = "bookworm-datalake"

# S3 paths
RAW_BOOKS        = f"s3://{S3_BUCKET}/raw/books/"
RAW_REVIEWS      = f"s3://{S3_BUCKET}/raw/reviews/"
RAW_INTERACTIONS = f"s3://{S3_BUCKET}/raw/interactions/"
BRONZE_BOOKS     = f"s3://{S3_BUCKET}/bronze/books/"
BRONZE_REVIEWS   = f"s3://{S3_BUCKET}/bronze/reviews/"
BRONZE_INTERACT  = f"s3://{S3_BUCKET}/bronze/interactions/"
CHECKPOINT_BOOKS = f"s3://{S3_BUCKET}/checkpoints/bronze/books/"
CHECKPOINT_REV   = f"s3://{S3_BUCKET}/checkpoints/bronze/reviews/"
CHECKPOINT_INT   = f"s3://{S3_BUCKET}/checkpoints/bronze/interactions/"

print("paths of raw & bronze folder")

# COMMAND ----------

try:
    files = dbutils.fs.ls(RAW_BOOKS)
    print(f"S3 accessible")
    print(f"   Found {len(files)} book files")
    for f in files[:5]:
        print(f"{f.name} ({f.size/(1024*1024):.1f} MB)")
except Exception as e:
    print(f"{str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Autoloader Ingestion

# COMMAND ----------

# DBTITLE 1,Cell 3
# reads all json files from raw/books/
# Detects new files automatically on each run
# Writes to Bronze as Delta table

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", CHECKPOINT_BOOKS + "schema/")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(RAW_BOOKS)
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_BOOKS)
    .option("mergeSchema", "true")
    .outputMode("append")
    .trigger(availableNow=True) #this take care of batch processing
    .start(BRONZE_BOOKS)
    .awaitTermination()
)

count = spark.read.format("delta").load(BRONZE_BOOKS).count()
print(f"Bronze book files created— {count:,} rows")

# COMMAND ----------

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", CHECKPOINT_INT + "schema/")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(RAW_INTERACTIONS)
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file",F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_INT)
    .option("mergeSchema", "true")
    .outputMode("append")
    .trigger(availableNow=True)
    .start(BRONZE_INTERACT)
    .awaitTermination()
)

count = spark.read.format("delta").load(BRONZE_INTERACT).count()
print(f"Bronze interaction files created — {count:,} rows")

# COMMAND ----------

(spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", CHECKPOINT_REV + "schema/")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(RAW_REVIEWS)
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file",F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_REV)
    .option("mergeSchema", "true")
    .outputMode("append")
    .trigger(availableNow=True)
    .start(BRONZE_REVIEWS)
    .awaitTermination()
)

count = spark.read.format("delta").load(BRONZE_REVIEWS).count()
print(f"Bronze review files created — {count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC Monitoring 
# MAGIC - Row counts — today vs yesterday (Delta time travel)
# MAGIC - New files added — from Auto Loader checkpoint
# MAGIC - File metadata changes — size changes detected

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta

def bronze_summary_table(bronze_path, label):

    # Row count comparison 
    print(f"\nRow Count Comparison:")
    try:
        today     = spark.read.format("delta").load(bronze_path).count()
        yesterday = spark.read \
            .format("delta") \
            .option("timestampAsOf",
                (datetime.now() - timedelta(days=1))
                .strftime("%Y-%m-%d %H:%M:%S")) \
            .load(bronze_path) \
            .count()

        diff   = today - yesterday
        display(spark.createDataFrame([{
            "yesterday"  : yesterday,
            "today"      : today,
            "difference" : diff,
        }]))

    except:
        today = spark.read.format("delta").load(bronze_path).count()
        display(spark.createDataFrame([{
            "yesterday"  : "N/A — first run",
            "today"      : today,
            "difference" : "N/A",
            "status"     : "First load"
        }]))

    # Delta operation history
    print(f"\nDelta History (last 5 operations):")
    display(
        spark.sql(f"""
            DESCRIBE HISTORY delta.`{bronze_path}`
        """).select(
            "version",
            "timestamp",
            "operation",
            "operationMetrics"
        ).limit(5)
    )

    # Raw file metadata
    print(f"\nTable 3 — Raw File Metadata:")
    raw_path = bronze_path.replace(
        f"s3://{S3_BUCKET}/bronze/",
        f"s3://{S3_BUCKET}/raw/"
    )
    files = dbutils.fs.ls(raw_path)
    display(spark.createDataFrame([{
        "file_name"     : f.name,
        "size_mb"       : round(f.size / (1024*1024), 2),
        "last_modified" : datetime.fromtimestamp(
            f.modificationTime / 1000
        ).strftime("%Y-%m-%d %H:%M:%S")
    } for f in files]))


# all three Bronze tables 
bronze_summary_table(BRONZE_BOOKS,    "Bronze Books")
bronze_summary_table(BRONZE_INTERACT, "Bronze Interactions")
bronze_summary_table(BRONZE_REVIEWS,  "Bronze Reviews")