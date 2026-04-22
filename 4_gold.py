# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Notebook
# MAGIC 1. Reads only current rows from Silver (WHERE is_current = True)
# MAGIC 2. Aggregates reviews and interactions per book — counts, averages, sentiment
# MAGIC 3. Extracts shelf signals — to_read_count, currently_reading_count, read_count
# MAGIC 4. Calculates popularity_score — weighted blend of rating, volume, demand and sentiment
# MAGIC 5. Applies audiobook_priority flag — High / Medium / Low based on page count and rating
# MAGIC 6. Writes two Gold tables — gold_audiobook_priority (BI team) and gold_book_features (ML team) — both with is_current = True so Power BI always loads latest data only

# COMMAND ----------

from pyspark.sql import functions as F
from delta.tables import DeltaTable

S3_BUCKET = "bookworm-datalake"

SILVER_DIM_BOOKS     = f"s3://{S3_BUCKET}/silver/dim_books/"
SILVER_DIM_AUTHORS   = f"s3://{S3_BUCKET}/silver/dim_authors/"
SILVER_DIM_SHELVES   = f"s3://{S3_BUCKET}/silver/dim_shelves/"
SILVER_BRIDGE        = f"s3://{S3_BUCKET}/silver/bridge_book_authors/"
SILVER_FACT_ACTIVITY = f"s3://{S3_BUCKET}/silver/fact_book_activity/"
SILVER_REVIEWS      = f"s3://{S3_BUCKET}/silver/reviews/"
SILVER_INTERACT     = f"s3://{S3_BUCKET}/silver/interactions/"
GOLD_AUDIOBOOK  = f"s3://{S3_BUCKET}/gold/audiobook_priority/"
GOLD_ML_FEATURES= f"s3://{S3_BUCKET}/gold/book_features/"

print(" Gold paths configured")

# COMMAND ----------

# Read all Silver tables
books = spark.read.format("delta").load(SILVER_DIM_BOOKS) \
             .filter(F.col("is_current") == True)

authors = spark.read.format("delta").load(SILVER_DIM_AUTHORS) \
               .filter(F.col("is_current") == True)

bridge = spark.read.format("delta").load(SILVER_BRIDGE)

shelves = spark.read.format("delta").load(SILVER_DIM_SHELVES)

reviews = spark.read.format("delta").load(SILVER_REVIEWS)

interact = spark.read.format("delta").load(SILVER_INTERACT)

print(f" books     : {books.count():,}")
print(f" authors   : {authors.count():,}")
print(f"bridge    : {bridge.count():,}")
print(f"shelves   : {shelves.count():,}")
print(f"reviews   : {reviews.count():,}")
print(f"interact  : {interact.count():,}")

# COMMAND ----------

# Reviews aggregation 
reviews_agg = reviews.groupBy("book_id").agg(
    F.count("*")                                         .alias("total_reviews"),
    F.round(F.avg("rating"), 2)                          .alias("avg_review_rating"),
    F.sum(F.when(F.col("rating") >= 4, 1).otherwise(0)) .alias("positive_reviews"),
    F.sum(F.when(F.col("rating") <= 2, 1).otherwise(0)) .alias("negative_reviews"),
    F.round(F.avg(F.col("n_votes")), 2)                  .alias("avg_votes"),
)
print(f"reviews_agg: {reviews_agg.count():,} books")

# COMMAND ----------

# Interactions aggregation 
interact_agg = interact.groupBy("book_id").agg(
    F.count("*")                          .alias("total_interactions"),
    F.sum("is_read")                      .alias("total_read"),
    F.sum("has_review")                   .alias("total_reviewed"),
    F.round(F.avg("rating"), 2)           .alias("avg_interaction_rating"),
)
print(f"interact_agg: {interact_agg.count():,} books")

# COMMAND ----------

# Shelf signals
to_read = shelves \
    .filter(F.col("shelf_name") == "to-read") \
    .select("book_id", F.col("shelf_count").alias("to_read_count"))

currently_reading = shelves \
    .filter(F.col("shelf_name") == "currently-reading") \
    .select("book_id", F.col("shelf_count").alias("currently_reading_count"))

read_shelf = shelves \
    .filter(F.col("shelf_name") == "read") \
    .select("book_id", F.col("shelf_count").alias("shelf_read_count"))

print("Shelf signals ready")

# COMMAND ----------

# Author mapping
book_authors = bridge \
    .filter(F.col("role") == "Author") \
    .groupBy("book_id") \
    .agg(
        F.concat_ws(", ",
            F.collect_list("author_id")
        ).alias("author_ids")
    )

print(f"book_authors: {book_authors.count():,} books")

# COMMAND ----------


# BUILD GOLD — AUDIOBOOK PRIORITY (BI Team)


df_gold_bi = books \
    .join(reviews_agg,       "book_id", "left") \
    .join(interact_agg,      "book_id", "left") \
    .join(to_read,           "book_id", "left") \
    .join(currently_reading, "book_id", "left") \
    .join(read_shelf,        "book_id", "left") \
    .join(book_authors,      "book_id", "left") \
    .withColumn("positive_review_pct",
        F.when(F.col("total_reviews") > 0,
            F.round(F.col("positive_reviews") / F.col("total_reviews") * 100, 1)
        ).otherwise(0)
    ) \
    .withColumn("demand_score",
        F.round(F.log1p(F.coalesce(F.col("to_read_count"), F.lit(0))), 3)
    ) \
    .withColumn("popularity_score",
        F.round(
            (F.col("average_rating")                                       * 0.4) +
            (F.log1p(F.col("ratings_count"))                               * 0.2) +
            (F.col("demand_score")                                         * 0.25) +
            (F.coalesce(F.col("positive_review_pct"), F.lit(0)) / 100     * 0.15),
            3
        )
    ) \
    .withColumn("audiobook_priority",
        F.when(
            (F.col("num_pages").between(50, 400)) &
            (F.col("average_rating") >= 4.0),
            "High"
        ).when(
            (F.col("num_pages").between(50, 600)) &
            (F.col("average_rating") >= 3.5),
            "Medium"
        ).otherwise("Low")
    ) \
    .withColumn("is_current",        F.lit(True)) \
    .withColumn("valid_from",        F.current_date()) \
    .withColumn("valid_to",          F.lit("9999-12-31").cast("date")) \
    .withColumn("_gold_timestamp",   F.current_timestamp()) \
    .select(
        "book_id", "title", "author_ids",
        "language_code", "publication_year",
        "num_pages", "format", "is_ebook",
        "average_rating", "ratings_count",
        "total_reviews", "positive_reviews",
        "negative_reviews", "avg_review_rating",
        "positive_review_pct",
        "to_read_count", "currently_reading_count",
        "shelf_read_count", "total_read",
        "total_interactions", "demand_score",
        "popularity_score", "audiobook_priority",
        "is_current", "valid_from", "valid_to",
        "_gold_timestamp"
    ) \
    .orderBy(F.col("popularity_score").desc())

# Expire previous gold rows
try:
    gold_tbl = DeltaTable.forPath(spark, GOLD_AUDIOBOOK)
    gold_tbl.update(
        condition = "is_current = true",
        set = {
            "is_current": "false",
            "valid_to"  : "date_sub(current_date(), 1)"
        }
    )
except:
    pass

df_gold_bi.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save(GOLD_AUDIOBOOK)

current = spark.read.format("delta").load(GOLD_AUDIOBOOK) \
              .filter(F.col("is_current") == True).count()
print(f"gold_audiobook_priority: {current:,} current rows")
display(
    spark.read.format("delta").load(GOLD_AUDIOBOOK)
    .filter(F.col("is_current") == True)
    .filter(F.col("audiobook_priority") == "High")
    .orderBy(F.col("popularity_score").desc())
    .limit(10)
)

# COMMAND ----------


# BUILD GOLD — BOOK FEATURES (ML Team)
# Keeps description + all numeric features for ML


df_gold_ml = books \
    .join(reviews_agg,       "book_id", "left") \
    .join(interact_agg,      "book_id", "left") \
    .join(to_read,           "book_id", "left") \
    .join(currently_reading, "book_id", "left") \
    .join(read_shelf,        "book_id", "left") \
    .withColumn("positive_review_pct",
        F.when(F.col("total_reviews") > 0,
            F.round(F.col("positive_reviews") / F.col("total_reviews") * 100, 1)
        ).otherwise(0)
    ) \
    .withColumn("popularity_score",
        F.round(
            (F.col("average_rating")                                       * 0.4) +
            (F.log1p(F.col("ratings_count"))                               * 0.2) +
            (F.log1p(F.coalesce(F.col("to_read_count"), F.lit(0)))         * 0.25) +
            (F.coalesce(F.col("positive_review_pct"), F.lit(0)) / 100     * 0.15),
            3
        )
    ) \
    .withColumn("is_current",      F.lit(True)) \
    .withColumn("valid_from",      F.current_date()) \
    .withColumn("valid_to",        F.lit("9999-12-31").cast("date")) \
    .withColumn("_gold_timestamp", F.current_timestamp()) \
    .select(
        "book_id", "title", "description",
        "language_code", "num_pages",
        "publication_year", "publication_month",
        "is_ebook", "format",
        "average_rating", "ratings_count",
        "text_reviews_count",
        "total_reviews", "positive_reviews",
        "negative_reviews", "avg_review_rating",
        "positive_review_pct", "avg_votes",
        "to_read_count", "currently_reading_count",
        "shelf_read_count", "total_read",
        "total_interactions", "avg_interaction_rating",
        "popularity_score",
        "is_current", "valid_from", "valid_to",
        "_gold_timestamp"
    )

# Expire previous
try:
    ml_tbl = DeltaTable.forPath(spark, GOLD_ML)
    ml_tbl.update(
        condition = "is_current = true",
        set = {
            "is_current": "false",
            "valid_to"  : "date_sub(current_date(), 1)"
        }
    )
except:
    pass

df_gold_ml.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save(GOLD_ML_FEATURES)

current = spark.read.format("delta").load(GOLD_ML_FEATURES) \
              .filter(F.col("is_current") == True).count()
print(f"✅ gold_book_features: {current:,} current rows")
display(
    spark.read.format("delta").load(GOLD_ML_FEATURES)
    .filter(F.col("is_current") == True)
    .limit(10)
)

# COMMAND ----------


# GOLD SUMMARY


print("=" * 55)
print(" GOLD COMPLETE SUMMARY")
print("=" * 55)

ga = spark.read.format("delta").load(GOLD_AUDIOBOOK)
gm = spark.read.format("delta").load(GOLD_ML_FEATURES)

display(spark.createDataFrame([
    {
        "table"  : "gold_audiobook_priority",
        "total"  : ga.count(),
        "current": ga.filter(F.col("is_current")==True).count(),
        "for"    : "BI team / Power BI"
    },
    {
        "table"  : "gold_book_features",
        "total"  : gm.count(),
        "current": gm.filter(F.col("is_current")==True).count(),
        "for"    : "ML team"
    },
]))

print("\n Top 10 High Priority Audiobooks:")
display(
    spark.read.format("delta").load(GOLD_AUDIOBOOK)
    .filter(F.col("is_current") == True)
    .filter(F.col("audiobook_priority") == "High")
    .orderBy(F.col("popularity_score").desc())
    .select(
        "title", "author_ids", "average_rating",
        "positive_review_pct", "to_read_count",
        "popularity_score", "audiobook_priority"
    )
    .limit(10)
)

# COMMAND ----------

# COMMAND ----------
# Step 1 — Make sure catalog and schema exist

spark.sql("CREATE CATALOG IF NOT EXISTS bookworm")
spark.sql("CREATE SCHEMA IF NOT EXISTS bookworm.gold")

# COMMAND ----------
# Step 2 — Drop and recreate gold tables pointing to S3

spark.sql("DROP TABLE IF EXISTS bookworm.gold.audiobook_priority")
spark.sql("DROP TABLE IF EXISTS bookworm.gold.book_features")

# COMMAND ----------
# Step 3 — Register with correct S3 path

spark.sql("""
    CREATE TABLE bookworm.gold.audiobook_priority
    USING DELTA
    LOCATION 's3://bookworm-datalake/gold/audiobook_priority/'
""")

spark.sql("""
    CREATE TABLE bookworm.gold.book_features
    USING DELTA
    LOCATION 's3://bookworm-datalake/gold/book_features/'
""")

# COMMAND ----------
# Step 4 — Verify data is visible

df = spark.sql("SELECT COUNT(*) AS total_rows FROM bookworm.gold.audiobook_priority")
df.show()