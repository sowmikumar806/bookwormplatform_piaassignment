# bookwormplatform_piaassignment

A cloud-native data platform built on Databricks 
for audiobook title prioritisation at BookWorm Publishing.

## Architecture
GoodReads JSON → AWS S3 → Bronze (Auto Loader) 
→ Silver (Star Schema + SCD Type 2) 
→ Gold (Popularity Score) 
→ Unity Catalog → Power BI

## Notebooks
- 01_download_data — Downloads GoodReads dataset to S3
- 02_bronze_autoloader — Auto Loader ingestion to Bronze Delta
- 03_silver — Star Schema, SCD Type 2, staging layer
- 04_gold — Popularity score, audiobook priority flag

## Setup
Replace in each notebook before running:
  AWS_ACCESS_KEY → your AWS access key
  AWS_SECRET_KEY → your AWS secret key
  S3_BUCKET      → your S3 bucket name

## Data Source
GoodReads Graph Dataset
https://mengtingwan.github.io/data/goodreads
