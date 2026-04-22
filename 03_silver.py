# Databricks notebook source
# MAGIC %md
# MAGIC Silver notebook
# MAGIC
# MAGIC - -Transform data from bronze 
# MAGIC - -Staging (intermediate)-to store today fresh data (each for comparison to apply SCD)
# MAGIC
# MAGIC Handled three data
# MAGIC -Book
# MAGIC -Review
# MAGIC -Interactions
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from delta.tables import DeltaTable

S3_BUCKET           = "bookworm-datalake"
BRONZE_BOOKS        = f"s3://{S3_BUCKET}/bronze/books/"
BRONZE_REVIEWS      = f"s3://{S3_BUCKET}/bronze/reviews/"
BRONZE_INTERACT     = f"s3://{S3_BUCKET}/bronze/interactions/"
STAGING_BOOKS       = f"s3://{S3_BUCKET}/staging/books/"
STAGING_REVIEWS     = f"s3://{S3_BUCKET}/staging/reviews/"
STAGING_INTERACT    = f"s3://{S3_BUCKET}/staging/interactions/"
DIM_BOOKS           = f"s3://{S3_BUCKET}/silver/dim_books/"
DIM_AUTHORS         = f"s3://{S3_BUCKET}/silver/dim_authors/"
DIM_SHELVES         = f"s3://{S3_BUCKET}/silver/dim_shelves/"
DIM_DATE            = f"s3://{S3_BUCKET}/silver/dim_date/"
BRIDGE_BOOK_AUTHORS = f"s3://{S3_BUCKET}/silver/bridge_book_authors/"
SILVER_REVIEWS      = f"s3://{S3_BUCKET}/silver/reviews/"
SILVER_INTERACT     = f"s3://{S3_BUCKET}/silver/interactions/"

today      = F.current_date()
far_future = F.lit("9999-12-31").cast("date")

print("Paths configured")

# COMMAND ----------

# MAGIC %md
# MAGIC # BRONZE

# COMMAND ----------

# MAGIC %md
# MAGIC Cleaning & Transformation 
# MAGIC 1.check duplciate or null on key column(book id)
# MAGIC 2.count the rows (data mismatch)
# MAGIC 3.cast the datatype
# MAGIC 4.drop column not required for business usecase
# MAGIC 5.create dimension(dim) tables 

# COMMAND ----------


# BOOKS — READ BRONZE
df_raw = spark.read.format("delta").load(BRONZE_BOOKS)
print(f"Bronze books: {df_raw.count():,} rows")
df_raw.printSchema()

# COMMAND ----------

df_staging = df_raw.select(
    F.col("book_id")                       .cast("string"),
    F.trim(F.col("title"))                 .cast("string").alias("title"),
    F.trim(F.col("title_without_series"))  .cast("string").alias("title_without_series"),
    F.expr("try_cast(average_rating    AS FLOAT)")  .alias("average_rating"),
    F.expr("try_cast(ratings_count     AS INT)")    .alias("ratings_count"),
    F.expr("try_cast(text_reviews_count AS INT)")   .alias("text_reviews_count"),
    F.upper(F.trim(F.col("language_code"))).cast("string").alias("language_code"),
    F.expr("try_cast(num_pages          AS INT)")   .alias("num_pages"),
    F.expr("try_cast(publication_year   AS INT)")   .alias("publication_year"),
    F.expr("try_cast(publication_month  AS INT)")   .alias("publication_month"),
    F.trim(F.col("publisher"))             .cast("string").alias("publisher"),
    F.trim(F.col("format"))                .cast("string").alias("format"),
    F.trim(F.col("description"))           .cast("string").alias("description"),
    F.col("isbn")                          .cast("string"),
    F.col("isbn13")                        .cast("string"),
    F.col("country_code")                  .cast("string"),
    F.when(F.lower(F.col("is_ebook")) == "true",  1)
     .when(F.lower(F.col("is_ebook")) == "false", 0)
     .otherwise(None).cast("integer").alias("is_ebook"),
) \
.filter(F.col("book_id").isNotNull()) \
.filter(F.col("title").isNotNull()) \
.filter(F.col("average_rating").between(0.0, 5.0)) \
.filter(F.col("ratings_count") > 0) \
.dropDuplicates(["book_id"]) \
.withColumn("last_updated_at",    F.current_timestamp()) \
.withColumn("_staging_timestamp", F.current_timestamp())

# COMMAND ----------

df_staging.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(STAGING_BOOKS)

print(f"staging/books: {df_staging.count():,} rows")
display(df_staging.limit(5))

# COMMAND ----------


# DIM_BOOKS — SCD Type 2
# New version created when rating/counts/pages change
# is_current = True → latest data

df_new_books = spark.read.format("delta").load(STAGING_BOOKS) \
    .withColumn("is_current",        F.lit(True)) \
    .withColumn("valid_from",        today) \
    .withColumn("valid_to",          far_future) \
    .withColumn("_silver_timestamp", F.current_timestamp())

try:
    dim_books_tbl = DeltaTable.forPath(spark, DIM_BOOKS)

    # Expire changed rows
    dim_books_tbl.alias("t").merge(
        df_new_books.alias("s"),
        "t.book_id = s.book_id AND t.is_current = true"
    ).whenMatchedUpdate(
        condition = """
            t.average_rating     != s.average_rating     OR
            t.ratings_count      != s.ratings_count      OR
            t.text_reviews_count != s.text_reviews_count OR
            t.num_pages          != s.num_pages          OR
            t.publisher          != s.publisher          OR
            t.format             != s.format
        """,
        set = {
            "is_current": "false",
            "valid_to"  : "date_sub(current_date(), 1)"
        }
    ).execute()

    # Find rows to insert — new books + changed books
    df_current_books = spark.read.format("delta").load(DIM_BOOKS) \
        .filter(F.col("is_current") == True) \
        .select("book_id", "average_rating", "ratings_count",
                "text_reviews_count", "num_pages", "publisher", "format")

    df_to_insert = df_new_books.alias("s") \
        .join(df_current_books.alias("c"), "book_id", "left") \
        .filter(
            F.col("c.book_id").isNull() |
            (F.col("s.average_rating")     != F.col("c.average_rating"))     |
            (F.col("s.ratings_count")      != F.col("c.ratings_count"))      |
            (F.col("s.text_reviews_count") != F.col("c.text_reviews_count")) |
            (F.col("s.num_pages")          != F.col("c.num_pages"))          |
            (F.col("s.publisher")          != F.col("c.publisher"))          |
            (F.col("s.format")             != F.col("c.format"))
        ).select("s.*")

    if df_to_insert.count() > 0:
        df_to_insert.write.format("delta").mode("append").save(DIM_BOOKS)

except Exception as e:
    if "Path does not exist" in str(e) or "is not a Delta table" in str(e):
        df_new_books.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(DIM_BOOKS)
    else:
        raise e

total   = spark.read.format("delta").load(DIM_BOOKS).count()
current = spark.read.format("delta").load(DIM_BOOKS) \
              .filter(F.col("is_current") == True).count()
print(f" dim_books — total: {total:,} | current: {current:,} | history: {total-current:,}")
display(spark.read.format("delta").load(DIM_BOOKS).filter(F.col("is_current") == True).limit(5))

# COMMAND ----------

# DIM_AUTHORS — SCD Type 2
# One row per author_id + role
# Explode authors array directly 

df_new_authors = df_raw \
    .select(F.explode(F.col("authors")).alias("a")) \
    .select(
        F.col("a.author_id").cast("string").alias("author_id"),
        F.when(
            F.col("a.role").isNull() | (F.trim(F.col("a.role")) == ""),
            "Author"
        ).otherwise(F.trim(F.col("a.role"))).alias("role")
    ) \
    .filter(F.col("author_id").isNotNull()) \
    .filter(F.col("author_id") != "") \
    .dropDuplicates(["author_id"]) \
    .withColumn("is_current",        F.lit(True)) \
    .withColumn("valid_from",        today) \
    .withColumn("valid_to",          far_future) \
    .withColumn("_silver_timestamp", F.current_timestamp())

try:
    dim_authors_tbl = DeltaTable.forPath(spark, DIM_AUTHORS)

    dim_authors_tbl.alias("t").merge(
        df_new_authors.alias("s"),
        "t.author_id = s.author_id AND t.is_current = true"
    ).whenMatchedUpdate(
        condition = "t.role != s.role",
        set = {
            "is_current": "false",
            "valid_to"  : "date_sub(current_date(), 1)"
        }
    ).execute()

    df_current_authors = spark.read.format("delta").load(DIM_AUTHORS) \
        .filter(F.col("is_current") == True) \
        .select("author_id", "role")

    df_to_insert = df_new_authors.alias("s") \
        .join(df_current_authors.alias("c"), "author_id", "left") \
        .filter(
            F.col("c.author_id").isNull() |
            (F.col("s.role") != F.col("c.role"))
        ).select("s.*")

    if df_to_insert.count() > 0:
        df_to_insert.write.format("delta").mode("append").save(DIM_AUTHORS)

except Exception as e:
    if "Path does not exist" in str(e) or "is not a Delta table" in str(e):
        df_new_authors.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(DIM_AUTHORS)
    else:
        raise e

total   = spark.read.format("delta").load(DIM_AUTHORS).count()
current = spark.read.format("delta").load(DIM_AUTHORS) \
              .filter(F.col("is_current") == True).count()
print(f"dim_authors — total: {total:,} | current: {current:,}")
display(spark.read.format("delta").load(DIM_AUTHORS).filter(F.col("is_current") == True).limit(5))

# COMMAND ----------


# BRIDGE_BOOK_AUTHORS — SCD Type 1 overwrite

df_bridge = df_raw \
    .select(
        F.col("book_id").cast("string"),
        F.explode(F.col("authors")).alias("a")
    ) \
    .select(
        F.col("book_id"),
        F.col("a.author_id").cast("string").alias("author_id"),
        F.when(
            F.col("a.role").isNull() | (F.trim(F.col("a.role")) == ""),
            "Author"
        ).otherwise(F.trim(F.col("a.role"))).alias("role")
    ) \
    .filter(F.col("book_id").isNotNull()) \
    .filter(F.col("author_id").isNotNull()) \
    .filter(F.col("author_id") != "") \
    .dropDuplicates(["book_id", "author_id"]) \
    .withColumn("_silver_timestamp", F.current_timestamp())

df_bridge.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(BRIDGE_BOOK_AUTHORS)

print(f"✅ bridge_book_authors: {df_bridge.count():,} rows")
display(df_bridge.limit(5))

# COMMAND ----------

# DIM_SHELVES — SCD Type 1 overwrite
# One row per book + standardised shelf name
# Synonyms mapped to canonical values

df_shelves_raw = df_raw \
    .select(
        F.col("book_id").cast("string"),
        F.explode(F.col("popular_shelves")).alias("s")
    ) \
    .select(
        F.col("book_id"),
        F.lower(F.trim(F.col("s.name"))).alias("shelf_name_raw"),
        F.col("s.count").cast("integer").alias("shelf_count"),
    ) \
    .filter(F.col("book_id").isNotNull()) \
    .filter(F.col("shelf_name_raw").isNotNull()) \
    .filter(F.col("shelf_count") > 0)

# Standardise shelf names
# Map synonyms to canonical categories
df_shelves = df_shelves_raw \
    .withColumn("shelf_name",
        F.when(F.col("shelf_name_raw").isin(
            "to-read", "want-to-read", "watchlist",
            "wish-list", "to-buy", "next-to-read",
            "tbr", "my-to-read", "to-read-fiction",
            "want-to-buy", "to-read-soon"
        ), "to-read")
        .when(F.col("shelf_name_raw").isin(
            "currently-reading", "reading",
            "in-progress", "currently-reading-2023",
            "now-reading", "currently"
        ), "currently-reading")
        .when(F.col("shelf_name_raw").isin(
            "read", "already-read", "finished",
            "books-read", "have-read", "read-2023",
            "read-books", "done"
        ), "read")
        .when(F.col("shelf_name_raw").isin(
            "favorites", "favourite", "favourites",
            "faves", "all-time-favorites", "my-favorites"
        ), "favorites")
        .when(F.col("shelf_name_raw").isin(
            "owned", "own-it", "i-own",
            "owned-books", "to-buy", "my-library"
        ), "owned")
        .otherwise(F.col("shelf_name_raw"))
    ) \
    .groupBy("book_id", "shelf_name") \
    .agg(F.sum("shelf_count").alias("shelf_count")) \
    .withColumn("_silver_timestamp", F.current_timestamp())

df_shelves.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(DIM_SHELVES)

print(f"dim_shelves: {df_shelves.count():,} rows")

print("\nTop shelves by total count:")
display(
    spark.read.format("delta").load(DIM_SHELVES)
    .groupBy("shelf_name")
    .agg(F.sum("shelf_count").alias("total"))
    .orderBy(F.col("total").desc())
    .limit(10)
)

# COMMAND ----------


# DIM_DATE 
try:
    existing = spark.read.format("delta").load(DIM_DATE)
    print(f"✅ dim_date already exists — {existing.count():,} rows")
except:
    df_date = spark.sql("""
        SELECT
            date_format(d, 'yyyyMMdd') AS date_id,
            d                          AS full_date,
            year(d)                    AS year,
            month(d)                   AS month,
            dayofmonth(d)              AS day,
            quarter(d)                 AS quarter,
            date_format(d, 'MMMM')     AS month_name,
            date_format(d, 'EEEE')     AS day_of_week
        FROM (
            SELECT explode(sequence(
                to_date('1990-01-01'),
                to_date('2030-12-31'),
                interval 1 day
            )) AS d
        )
    """)
    df_date.write.format("delta").mode("overwrite").save(DIM_DATE)
    print(f"dim_date created — {df_date.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC # REVIEW TABLE
# MAGIC
# MAGIC - check for duplicate(review_id)
# MAGIC - format date columns
# MAGIC - cast datatype
# MAGIC - add last_updated

# COMMAND ----------


# REVIEWS — STAGING


df_reviews = spark.read.format("delta").load(BRONZE_REVIEWS)
print(f"Bronze reviews: {df_reviews.count():,} rows")

# Set legacy time parser — handles "Tue Jun 12 08:59:04 -0700 2012" format
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
ts_fmt = "EEE MMM dd HH:mm:ss Z yyyy"

df_staging_reviews = df_reviews.select(
    F.col("review_id")  .cast("string"),
    F.col("book_id")    .cast("string"),
    F.sha2(F.col("user_id").cast("string"), 256).alias("user_id"),
    F.col("rating")     .cast("integer"),
    F.col("review_text").cast("string"),
    F.col("n_votes")    .cast("integer"),
    F.col("n_comments") .cast("integer"),
    F.try_to_timestamp(F.col("date_added"),  F.lit(ts_fmt)).alias("date_added"),
    F.try_to_timestamp(F.col("date_updated"),F.lit(ts_fmt)).alias("date_updated"),
    F.try_to_timestamp(F.col("read_at"),     F.lit(ts_fmt)).alias("read_at"),
    F.try_to_timestamp(F.col("started_at"),  F.lit(ts_fmt)).alias("started_at"),
) \
.filter(F.col("review_id").isNotNull()) \
.filter(F.col("book_id").isNotNull()) \
.filter(F.col("rating").between(1, 5)) \
.dropDuplicates(["review_id"]) \
.withColumn("is_positive",
    F.when(F.col("rating") >= 4, 1).otherwise(0)) \
.withColumn("is_negative",
    F.when(F.col("rating") <= 2, 1).otherwise(0)) \
.withColumn("date_id",
    F.date_format(F.col("date_added"), "yyyyMMdd")) \
.withColumn("last_updated_at",    F.current_timestamp()) \
.withColumn("_staging_timestamp", F.current_timestamp())

df_staging_reviews.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(STAGING_REVIEWS)

print(f"staging/reviews: {df_staging_reviews.count():,} rows")
display(df_staging_reviews.limit(5))

# COMMAND ----------


# REVIEWS — SILVER FINAL (Append only)
# Reviews are immutable — only add new ones


df_staging_reviews = spark.read.format("delta").load(STAGING_REVIEWS)

try:
    existing_ids = spark.read.format("delta").load(SILVER_REVIEWS) \
        .select("review_id")

    df_new = df_staging_reviews \
        .join(existing_ids, "review_id", "left_anti")

    if df_new.count() > 0:
        df_new.write.format("delta").mode("append").save(SILVER_REVIEWS)
        print(f"Appended {df_new.count():,} new reviews")
    else:
        print("No new reviews")

except Exception as e:
    if "Path does not exist" in str(e) or "is not a Delta table" in str(e):
        df_staging_reviews.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true").save(SILVER_REVIEWS)
        print(f"First load — silver/reviews created")
    else:
        raise e

print(f"Total silver/reviews: {spark.read.format('delta').load(SILVER_REVIEWS).count():,}")
display(spark.read.format("delta").load(SILVER_REVIEWS).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Interactions

# COMMAND ----------


# INTERACTIONS — STAGING


df_interact = spark.read.format("delta").load(BRONZE_INTERACT)
print(f"Bronze interactions: {df_interact.count():,} rows")

# Set legacy time parser — handles "Tue Jun 12 08:59:04 -0700 2012" format
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

ts_fmt = "EEE MMM dd HH:mm:ss Z yyyy"

w = Window.partitionBy("user_id", "book_id") \
          .orderBy(F.col("date_updated").desc())

df_staging_interact = df_interact.select(
    F.sha2(F.col("user_id").cast("string"), 256).alias("user_id"),
    F.col("book_id")   .cast("string"),
    F.col("review_id") .cast("string"),
    F.when(F.lower(F.col("is_read")) == "true",  1)
     .when(F.lower(F.col("is_read")) == "false", 0)
     .otherwise(None).cast("integer").alias("is_read"),
    F.when(F.col("rating").cast("integer") == 0, None)
     .otherwise(F.col("rating").cast("integer")).alias("rating"),
    F.try_to_timestamp(F.col("date_added"),  F.lit(ts_fmt)).alias("date_added"),
    F.try_to_timestamp(F.col("date_updated"),F.lit(ts_fmt)).alias("date_updated"),
    F.try_to_timestamp(F.col("read_at"),     F.lit(ts_fmt)).alias("read_at"),
    F.try_to_timestamp(F.col("started_at"),  F.lit(ts_fmt)).alias("started_at"),
) \
.filter(F.col("user_id").isNotNull()) \
.filter(F.col("book_id").isNotNull()) \
.withColumn("rn", F.row_number().over(w)) \
.filter(F.col("rn") == 1) \
.drop("rn") \
.withColumn("has_review",
    F.when(F.col("review_id").isNotNull(), 1).otherwise(0)) \
.withColumn("date_id",
    F.date_format(F.col("date_added"), "yyyyMMdd")) \
.withColumn("last_updated_at",    F.col("date_updated")) \
.withColumn("_staging_timestamp", F.current_timestamp())

df_staging_interact.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(STAGING_INTERACT)

print(f"✅ staging/interactions: {df_staging_interact.count():,} rows")
display(df_staging_interact.limit(5))

# COMMAND ----------

df_staging_interact = spark.read.format("delta").load(STAGING_INTERACT)

# Check if silver interactions exists
silver_interact_exists = False
try:
    spark.read.format("delta").load(SILVER_INTERACT).limit(1).collect()
    silver_interact_exists = True
except:
    silver_interact_exists = False

print(f"Silver interactions exists: {silver_interact_exists}")

if not silver_interact_exists:
    # First load — just write directly
    print("First load — writing silver/interactions...")
    df_staging_interact.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(SILVER_INTERACT)
    print(f"Created silver/interactions")

else:
    # Subsequent runs — merge
    print("Merging into silver/interactions...")
    silver_interact = DeltaTable.forPath(spark, SILVER_INTERACT)

    silver_interact.alias("t").merge(
        df_staging_interact.alias("s"),
        "t.user_id = s.user_id AND t.book_id = s.book_id"
    ).whenMatchedUpdate(
        condition = """
            s.date_updated > t.date_updated
            OR t.date_updated IS NULL
        """,
        set = {
            "review_id"      : "s.review_id",
            "is_read"        : "s.is_read",
            "rating"         : "s.rating",
            "date_updated"   : "s.date_updated",
            "read_at"        : "s.read_at",
            "started_at"     : "s.started_at",
            "has_review"     : "s.has_review",
            "last_updated_at": "s.last_updated_at",
        }
    ).whenNotMatchedInsertAll() \
     .execute()
    print("Merge complete")

total = spark.read.format("delta").load(SILVER_INTERACT).count()
print(f"Total silver/interactions: {total:,} rows")
display(spark.read.format("delta").load(SILVER_INTERACT).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary

# COMMAND ----------


# SILVER SUMMARY


print("=" * 55)
print("SILVER COMPLETE SUMMARY")


sb  = spark.read.format("delta").load(DIM_BOOKS)
sa  = spark.read.format("delta").load(DIM_AUTHORS)
ss  = spark.read.format("delta").load(DIM_SHELVES)
sbr = spark.read.format("delta").load(BRIDGE_BOOK_AUTHORS)
sr  = spark.read.format("delta").load(SILVER_REVIEWS)
si  = spark.read.format("delta").load(SILVER_INTERACT)

display(spark.createDataFrame([
    {"table":"dim_books",           "rows":sb.count(),  "current":sb.filter(F.col("is_current")==True).count(),  "strategy":"SCD Type 2"},
    {"table":"dim_authors",         "rows":sa.count(),  "current":sa.filter(F.col("is_current")==True).count(),  "strategy":"SCD Type 2"},
    {"table":"dim_shelves",         "rows":ss.count(),  "current":ss.count(),  "strategy":"Overwrite"},
    {"table":"bridge_book_authors", "rows":sbr.count(), "current":sbr.count(), "strategy":"Overwrite"},
    {"table":"silver/reviews",      "rows":sr.count(),  "current":sr.count(),  "strategy":"Append only"},
    {"table":"silver/interactions", "rows":si.count(),  "current":si.count(),  "strategy":"Merge"},
]))