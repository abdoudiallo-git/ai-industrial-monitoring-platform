import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"

import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pathlib import Path
import math

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/ai4i2020.csv"
PROCESSED_SPARK_PATH = "data/processed/ai4i_spark"

# ── Spark Session ────────────────────────────────────────────────────────────
def create_spark_session() -> SparkSession:
    """Create and return a Spark session."""
    spark = SparkSession.builder \
        .appName("AI Industrial Monitoring - PySpark Pipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark session created")
    return spark

# ── Loading ──────────────────────────────────────────────────────────────────
def load_data(spark: SparkSession, path: str):
    """Load raw CSV data into a Spark DataFrame."""
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"✓ Data loaded : {df.count()} rows, {len(df.columns)} columns")
    return df

# ── Cleaning ─────────────────────────────────────────────────────────────────
def clean_data(df):
    """Drop useless columns and rename."""
    df = df.drop("UDI", "Product ID")

    df = df.withColumnRenamed("Type", "type") \
           .withColumnRenamed("Air temperature [K]", "air_temp") \
           .withColumnRenamed("Process temperature [K]", "process_temp") \
           .withColumnRenamed("Rotational speed [rpm]", "rotational_speed") \
           .withColumnRenamed("Torque [Nm]", "torque") \
           .withColumnRenamed("Tool wear [min]", "tool_wear") \
           .withColumnRenamed("Machine failure", "machine_failure") \
           .withColumnRenamed("TWF", "twf") \
           .withColumnRenamed("HDF", "hdf") \
           .withColumnRenamed("PWF", "pwf") \
           .withColumnRenamed("OSF", "osf") \
           .withColumnRenamed("RNF", "rnf")

    print("✓ Columns renamed and cleaned")
    return df

# ── Encoding ─────────────────────────────────────────────────────────────────
def encode_type(df):
    """Encode machine type : L→0, M→1, H→2."""
    df = df.withColumn("type",
        F.when(F.col("type") == "L", 0)
         .when(F.col("type") == "M", 1)
         .when(F.col("type") == "H", 2)
         .otherwise(None).cast(IntegerType())
    )
    print("✓ Column 'type' encoded : L→0, M→1, H→2")
    return df

# ── Feature Engineering ───────────────────────────────────────────────────────
def add_features(df):
    """Create new features from existing ones."""
    df = df.withColumn("delta_temp",
        F.col("process_temp") - F.col("air_temp")
    )
    df = df.withColumn("power",
        F.col("torque") * F.col("rotational_speed") * F.lit(2 * math.pi / 60)
    )
    df = df.withColumn("tool_wear_torque",
        F.col("tool_wear") * F.col("torque")
    )
    print("✓ New features added : delta_temp, power, tool_wear_torque")
    return df

# ── Analysis ──────────────────────────────────────────────────────────────────
def run_analysis(df) -> None:
    """Run key analyses using Spark aggregations."""
    print("\n── Global Failure Rate ──")
    df.select(
        F.count("*").alias("total"),
        F.sum("machine_failure").alias("failures"),
        (F.sum("machine_failure") / F.count("*") * 100).alias("failure_rate_%")
    ).show()

    print("── Failure Rate by Machine Type ──")
    df.groupBy("type") \
      .agg(
          F.count("*").alias("count"),
          F.sum("machine_failure").alias("failures"),
          (F.sum("machine_failure") / F.count("*") * 100).alias("failure_rate_%")
      ) \
      .orderBy("type") \
      .show()

    print("── Failure Modes Distribution ──")
    failure_modes = ["twf", "hdf", "pwf", "osf", "rnf"]
    for mode in failure_modes:
        df.select(F.sum(mode).alias(f"{mode}_count")).show()

    print("── Descriptive Statistics ──")
    df.select(
        "air_temp", "process_temp", "rotational_speed",
        "torque", "tool_wear", "delta_temp", "power"
    ).describe().show()

# ── Saving ────────────────────────────────────────────────────────────────────
def save_data(df, path: str) -> None:
    """Convert to Pandas and save as CSV."""
    df_pandas = df.toPandas()
    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    df_pandas.to_csv(f"{path}/ai4i_spark.csv", index=False)
    print(f"✓ Data saved to {path}/ai4i_spark.csv")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = create_spark_session()

    df = load_data(spark, RAW_PATH)
    df = clean_data(df)
    df = encode_type(df)
    df = add_features(df)
    run_analysis(df)
    save_data(df, PROCESSED_SPARK_PATH)

    print("\n✓ Spark pipeline complete")
    spark.stop()
    print("✓ Spark session stopped")