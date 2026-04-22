# Databricks notebook source
# MAGIC %md
# MAGIC ###Download the Data 
# MAGIC        
# MAGIC        
# MAGIC        2 ways to store : DBFS and Cloud storage(Azure ADLS ,AWS S3,Google Cloud Storage) 

# COMMAND ----------

# MAGIC %md
# MAGIC Downloads the complete GoodReads dataset from the official source 
# MAGIC and stores raw files in AWS S3 — no transformation, no cleaning
# MAGIC
# MAGIC key functions 
# MAGIC - Dynamic cataloging -if new record add this to csv file
# MAGIC - s3.header_object()-skips file exist and download only new files

# COMMAND ----------

import requests
import os
import time
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

# S3 bucket configuration
S3_BUCKET        = "bookworm-datalake"

# COMMAND ----------

# All paths from s3
PATHS = {
    "books"        : f"s3://{S3_BUCKET}/raw/books/",
    "reviews"      : f"s3://{S3_BUCKET}/raw/reviews/",
    "interactions" : f"s3://{S3_BUCKET}/raw/interactions/",
    "complete"     : f"s3://{S3_BUCKET}/raw/complete/",
    "bronze"       : f"s3://{S3_BUCKET}/bronze/",
    "silver"       : f"s3://{S3_BUCKET}/silver/",
    "gold"         : f"s3://{S3_BUCKET}/gold/",
    "checkpoints"  : f"s3://{S3_BUCKET}/checkpoints/",
}

for name, path in PATHS.items():
    dbutils.fs.mkdirs(path)
    print(f"{name:<15} → {path}")

print(f"\nFolders created successfully")

# COMMAND ----------


# he original dataset url
CATALOGUE_URL = (
    "https://raw.githubusercontent.com/"
    "MengtingWan/goodreads/master/dataset_names.csv"
)

response = requests.get(CATALOGUE_URL, timeout=30)
response.raise_for_status()

# Parse CSV in memory
lines  = response.text.strip().split("\n")
header = [col.strip() for col in lines[0].split(",")]

rows = []
for line in lines[1:]:
    values = [v.strip() for v in line.split(",")]
    if len(values) == len(header):
        rows.append(dict(zip(header, values)))

# Create PySpark DataFrame
schema = StructType([
    StructField("type", StringType(), True),
    StructField("name", StringType(), True),
])

catalogue = spark.createDataFrame(rows, schema=schema)

assert "name" in catalogue.columns, "Expected 'name' column missing"
assert "type" in catalogue.columns, "Expected 'type' column missing"

print(f"Catalogue loaded from goodreads GitHub")
print(f"   Total datasets : {catalogue.count()}")
print(f"   Complete       : {catalogue.filter(F.col('type') == 'complete').count()}")
print(f"   By Genre       : {catalogue.filter(F.col('type') == 'byGenre').count()}")
print(f"\n📋 Full catalogue:")
display(catalogue)

# COMMAND ----------

from pyspark.sql.functions import col, when, concat, lit

#base URLs 
BASE_URLS = {
    "complete" : "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/",
    "byGenre"  : "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/",
}

def build_url(fname, ftype):
    if ftype not in BASE_URLS:
        raise ValueError(
            f"Unknown dataset type: {ftype}. "
            f"Expected one of {list(BASE_URLS.keys())}"
        )
    return BASE_URLS[ftype] + fname

print("Base URLs configured:")
for ftype, url in BASE_URLS.items():
    print(f"   {ftype:<12} → {url}")

# COMMAND ----------

def build_s3_path(fname, ftype):
    """
    Route each file to correct S3 folder
    based on filename content — no hardcoding
    """
    if ftype == "complete":
        return PATHS["complete"] + fname

    for keyword, path_key in [
        ("books",        "books"),
        ("reviews",      "reviews"),
        ("interactions", "interactions"),
    ]:
        if keyword in fname:
            return PATHS[path_key] + fname

    raise ValueError(
        f"Cannot determine S3 path for: {fname}. "
        f"Filename does not contain expected keyword."
    )

# Preview routing
print("S3 routing preview:")
for row in catalogue.limit(6).collect():
    s3 = build_s3_path(row["name"], row["type"])
    print(f"   {row['name']:<55} → {s3.replace(f's3://{S3_BUCKET}/','')}")
print(f"   ... and {catalogue.count() - 6} more")

# COMMAND ----------

# Build manifest as PySpark DataFrame
manifest_rows = []

for row in catalogue.collect():
    fname = row["name"]
    ftype = row["type"]
    manifest_rows.append({
        "name"    : fname,
        "type"    : ftype,
        "url"     : build_url(fname, ftype),
        "s3_path" : build_s3_path(fname, ftype),
    })

manifest = spark.createDataFrame(manifest_rows)

# Validate no nulls
assert manifest.filter(col("url").isNull()).count()     == 0, "Null URLs found"
assert manifest.filter(col("s3_path").isNull()).count() == 0, "Null S3 paths found"

print(f"Manifest built — {manifest.count()} files")
print(f"\n📋 Full manifest:")
display(manifest.select("name", "type", "url", "s3_path"))

# COMMAND ----------

import boto3
import requests

AWS_ACCESS_KEY = "AKIAXCB7QKFFGNOP2BGI"
AWS_SECRET_KEY = "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"
AWS_REGION     = "eu-north-1"
S3_BUCKET      = "bookworm-datalake"

# boto3 only — no Spark config needed for download
s3 = boto3.client(
    "s3",
    aws_access_key_id     = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_KEY,
    region_name           = AWS_REGION
)

# Test connection
try:
    s3.head_bucket(Bucket=S3_BUCKET)
    print(f"S3 connected — {S3_BUCKET}")
    print(f"Region — {AWS_REGION}")
except Exception as e:
    print(f"Failed: {str(e)}")

# COMMAND ----------

def stream_to_s3(url, s3_path):
    """
    Stream file directly from URL to S3 using boto3
    No local file needed — works on Serverless
    """
    bucket = s3_path.replace("s3://", "").split("/")[0]
    key    = s3_path.replace(f"s3://{bucket}/", "")
    fname  = s3_path.split("/")[-1]

    # Skip if already exists
    try:
        size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"] / (1024*1024)
        print(f"⏭️  {fname} ({size:.1f} MB) — skipping")
        return {"name": fname, "status": "skipped", "size_mb": size}
    except:
        pass

    # Stream URL → S3
    try:
        print(f"⬇️  {fname}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        s3.upload_fileobj(response.raw, bucket, key)
        size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"] / (1024*1024)
        print(f"   {size:.1f} MB")
        return {"name": fname, "status": "success", "size_mb": size}
    except Exception as e:
        print(f"   {fname} failed: {str(e)}")
        return {"name": fname, "status": "failed", "size_mb": 0, "error": str(e)}

# Download all files from manifest
print("=" * 60)
print("🚀 Starting download of all files")
print("=" * 60)

results = []
for row in manifest.collect():
    result = stream_to_s3(row["url"], row["s3_path"])
    results.append(result)

# Summary
results_df = spark.createDataFrame(results)
print("\n📊 Download Summary:")
display(
    results_df.groupBy("status")
    .agg(
        F.count("*").alias("count"),
        F.round(F.sum("size_mb"), 1).alias("total_mb")
    )
)