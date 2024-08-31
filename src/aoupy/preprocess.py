import polars as pl
from polars import DataFrame
from typing import Tuple, Optional
from functools import reduce

def clean_fitbit(fitbit: DataFrame, wear_time: DataFrame, date_of_birth: DataFrame, 
                min_wear_hours: int = 10, step_count_minmax:Tuple[int] = (100, 45_000), age_min: int = 18) -> DataFrame:
    """
    Cleans the fitbit query DataFrame with given parameters.

    Parameters:
    -----------
    fitbit: DataFrame
        DataFrame to clean
    wear_time: DataFrame
        Fitbit wear time at each day. Needs to have at least the columns 'person_id', 'date' and 'wear_time'.
    date_of_birth: DataFrame
        DataFrame that contains date of birth information for each person_id. Needs to have at least the columns 'person_id' and 'date_of_birth'.
    min_wear_hours: int
        Minimum hours a day the fitbit needs to be worn for each day.
    step_count_minmax: Tuple[int]
        Minimum and maximum step counts limits.
    age_min: int
        Minimum age to be included.
    
    Returns:
    --------
    Combined dataframes (fitbit, wear_time, date_of_birth) after subsetting according to the given limits.

    Example:
    --------
    clean_fitbit(fitbit, wear_time, demographics)
    """
    
    return (fitbit.join(wear_time, on=["person_id", "date"], how="inner")
            .filter(pl.col("wear_time")>=min_wear_hours)
            .filter(pl.col("steps")>step_count_minmax[0], pl.col("steps")<=step_count_minmax[1])
            .join(date_of_birth, on="person_id", how="inner")
            .filter((pl.col("date").str.to_date(format="%Y-%m-%d")-
               pl.col("date_of_birth").str.to_date(format="%Y-%m-%d %H:%M:%S %Z")).dt.total_hours()/24/365.25>=age_min))

def combine_events(df:DataFrame, start_date_col:str, threshold:int, group_by:Optional[str|None]=None, end_date_col:Optional[str|None]=None):
    """
    Takes a dataframe with start date and end dates (or only start date assuming end dates are the same) and combines the events if they occured closer than the threshold.

    Parameters:
    -----------
    df: DataFrame
        DataFrame containing start (and end dates and grouping variable)
    start_date_col: str
        Column name for the start dates (or event dates if no end date is provided)
    threshold: int
        Minimum number of days to combine events together
    group_by: str
        Optional - Column name to group the events 
    end_date_col: str
        Optional - Column name for the end dates of events
    
    Returns:
    --------
    Returns a dataframe of events with start date, end date grouping variable and duration of events. If end_date_col is given the start and end column names saved otherwise start_date and end_date are used for these columns 

    Example:
    --------
    date_to_ranges(hospital_visits, "start_date", group_by = "person_id", end_date_col="end_date", threshold=2)
    """
    if end_date_col == None:
        df = (df.with_columns(pl.col(start_date_col).alias("start_date"),
                              pl.col(start_date_col).alias("end_date")))
        df = df.drop(start_date_col) if start_date_col != "start_date" else df
        start_date_col = "start_date"
        end_date_col = "end_date"
    else:
        df.filter(pl.col(end_date_col)>pl.col(start_date_col))
    if group_by:
        return (df
                 .sort([group_by, start_date_col])
                 .group_by(group_by)
                 .agg(pl.struct(pl.col(end_date_col).shift(1).alias("next_date"), pl.exclude(group_by))).explode("next_date").unnest("next_date")
                 .with_columns(pl.when((pl.col(start_date_col) < pl.col("next_date")) & pl.col("next_date").is_not_null()).then(pl.col("next_date")).otherwise(pl.col(start_date_col)).alias(start_date_col)) 
                 .with_columns(roll_diff = (pl.col(start_date_col)-pl.col("next_date")).dt.total_days())
                 .with_columns(diff_flag = (pl.col("roll_diff")>threshold) | pl.col("roll_diff").is_null())
                 .with_columns(pl.col("diff_flag").cum_sum().alias("seq_num"))
                 .group_by([group_by,"seq_num"]).agg(pl.col(start_date_col).min().alias(start_date_col), pl.col(end_date_col).max().alias(end_date_col))
                 .drop("seq_num")
                 .with_columns((pl.col(end_date_col)-pl.col(start_date_col)).dt.total_days().alias("duration")))
    else:
        return (df
                 .sort([start_date_col])
                 .with_columns(pl.col(end_date_col).shift(1).alias("next_date"))
                 .with_columns(pl.when((pl.col(start_date_col) < pl.col("next_date")) & pl.col("next_date").is_not_null()).then(pl.col("next_date")).otherwise(pl.col(start_date_col)).alias(start_date_col)) 
                 .with_columns(roll_diff = (pl.col(start_date_col)-pl.col("next_date")).dt.total_days())
                 .with_columns(diff_flag = (pl.col("roll_diff")>threshold) | pl.col("roll_diff").is_null())
                 .with_columns(pl.col("diff_flag").cum_sum().alias("seq_num"))
                 .group_by(["seq_num"]).agg(pl.col(start_date_col).min().alias(start_date_col), pl.col(end_date_col).max().alias(end_date_col))
                 .drop("seq_num")
                 .with_columns((pl.col(end_date_col)-pl.col(start_date_col)).dt.total_days().alias("duration")))