import polars as pl
from polars import DataFrame
from typing import Tuple

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
