"""Main script for generating output.csv."""
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark import SparkContext


def generate_df_for_csv(df, subject, stat):
    df = df.select('SubjectId', 'Split', coalesce(stat).alias('Value')).withColumn('Stat', lit(stat)).withColumn(
        'Subject', lit(subject))
    return df.select('SubjectId', 'Stat', 'Split', 'Subject', 'Value')


def generate_df(pitch_data, subject, split_val):
    split_arr = [None] * 2
    if split_val == "PitcherSide":
        split_arr[0] = "vs RHP"
        split_arr[1] = "vs LHP"
    else:
        split_arr[0] = "vs RHH"
        split_arr[1] = "vs LHH"

    hit_vs_pitside_df = pitch_data.groupBy(subject, split_val).agg(
        {'PA': 'sum', 'AB': 'sum', 'H': 'sum', 'TB': 'sum', 'BB': 'sum', 'SF': 'sum', 'HBP': 'sum'}).filter(
        coalesce('sum(PA)') >= 25).withColumn('SubjectId', coalesce(subject)).withColumn('Split', (when(
        coalesce(split_val) == 'R', split_arr[0]).otherwise(split_arr[1]))).withColumn(
        'AVG',
        format_number(
            coalesce('sum(H)') / coalesce(
                'sum(AB)'), 3)).withColumn(
        'OBP',
        format_number((coalesce('sum(H)') + coalesce('sum(BB)') + coalesce('sum(HBP)')) / (
                coalesce('sum(AB)') + coalesce('sum(BB)') + coalesce('sum(HBP)') + coalesce('sum(SF)')),
                      3)).withColumn('SLG',
                                     format_number(
                                         coalesce('sum(TB)') / coalesce(
                                             'sum(AB)'), 3)).withColumn('OPS',
                                                                        format_number(coalesce('OBP') + coalesce('SLG'),
                                                                                      3))

    hit_vs_pitside_avg = generate_df_for_csv(hit_vs_pitside_df, subject, "AVG")
    hit_vs_pitside_obp = generate_df_for_csv(hit_vs_pitside_df, subject, "OBP")
    hit_vs_pitside_slg = generate_df_for_csv(hit_vs_pitside_df, subject, "SLG")
    hit_vs_pitside_ops = generate_df_for_csv(hit_vs_pitside_df, subject, "OPS")

    return hit_vs_pitside_avg.union(hit_vs_pitside_obp).union(hit_vs_pitside_slg).union(hit_vs_pitside_ops)


def main():
    # add basic program logic here
    sc = SparkContext("local", "Baseball Stats")
    sqlContext = SQLContext(sc)
    # lines = sc.textFile("./data/raw/pitchdata.csv", 16)
    pitch_data = sqlContext.read.load(path='./data/raw/pitchdata.csv', format='com.databricks.spark.csv',
                                      header='true',
                                      inferSchema='true')

    hitter_vs_pitside = generate_df(pitch_data, "HitterId", "PitcherSide")
    hitter_team_vs_pitside = generate_df(pitch_data, "HitterTeamId", "PitcherSide")
    pitcher_vs_hitside = generate_df(pitch_data, "PitcherId", "HitterSide")
    pitcher_team_vs_hitside = generate_df(pitch_data, "PitcherTeamId", "HitterSide")

    final_df = hitter_vs_pitside.union(hitter_team_vs_pitside).union(pitcher_vs_hitside).union(pitcher_team_vs_hitside)
    final_df = final_df.sort('SubjectID', 'Stat', 'Split', 'Subject')
    final_df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        "./data/processed/output.csv")


if __name__ == '__main__':
    main()
