#
#FUNCTION
#
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.dates as mdates

#-------------------------------
'''
Filters a dataset with datetime in the index by months, so can extract based on any month's data or multiple months
'''
#-------------------------------



def filter_dataframe_by_months(dataframe, months):
    """
    Filters rows from a DataFrame based on the month of its DateTime index.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to filter. Must have a DateTime index.
    - months (list of int): List of months to filter for (e.g., [1, 2, 3] for Jan, Feb, Mar).

    Returns:
    - pd.DataFrame: A DataFrame containing only the rows for the specified months.
    """
    # Validate input
    if not isinstance(months, list) or not all(isinstance(month, int) for month in months):
        raise ValueError("Months must be provided as a list of integers.")
    if not hasattr(dataframe.index, "month"):
        raise ValueError("The DataFrame must have a DateTime index.")

    # Perform filtering
    return dataframe[dataframe.index.month.isin(months)]


#-------------------------------
'''
Filter Into WMO guideline periods, starts with sub-daily or daily data and fixes data into Monthly and Yearly based on 
missing data
'''
#-------------------------------



def WMO_Guidelines_Monthly_Yearly(Data,Date_Col,Type_Col,Rain):
    if(Rain==True):
        #Raindays/Rainfall means no sub-dailies
        Data_D = Data.dropna() #Exclude all NaN data
        #Monthly data needs to be subjected to WMO guidelines
        Data_M = WMO_Guide_Months(Data_D,Date_Col,Type_Col,True)
        #Now apply similar guideline ratios onto the Yearly
        Data_Y = WMO_Guide_Yearly(Data_M.dropna(),Date_Col,Type_Col,True)
        return(Data_D,Data_M,Data_Y)
    else:
        #Pressure cna be averaged
        Data_SD = Data.dropna() #Exclude all NaN data
        Data_D = Data_SD.resample('D').mean()
        #Monthly data needs to be subjected to WMO guidelines
        Data_M = WMO_Guide_Months(Data_D,Date_Col,Type_Col,False)
        #Now apply similar guideline ratios onto the Yearly
        Data_Y = WMO_Guide_Yearly(Data_M.dropna(),Date_Col,Type_Col,False)
        return(Data_SD,Data_D,Data_M,Data_Y)

#-------------------------------
'''
Filter Into WMO guidelines - monthly corrected
'''
#-------------------------------
# WMO Guidelined Months
def WMO_Guide_Months(D,Date_Column,Type_Column,Raindays):    

    #The appendenation of the daily data into monthlies
    Usuable = [] #WMO Standard 
    #Since we have dropped all periods with missing days lets work out days avalaible
    count_for_month = D.resample('MS').count().reset_index()

    #Get it into monthlies
    if (Raindays == True):
        M = D.resample('MS').sum()
    else:
        M = D.resample('MS').mean()

    #Fill in all days where no data is avalaible with nans
    nan_dates = pd.date_range(start=D.index.min(), end=D.index.max(), freq='D') 
    # Create an empty DataFrame with these monthly dates as index and label it MSLP
    Data_Missing = pd.DataFrame(index=nan_dates, columns=[Type_Column])
    Data_Missing.index.name = Date_Column  # Set the index name to 'Date'
    Data_Missing[:] = np.nan
    D = pd.concat([D, Data_Missing[~Data_Missing.index.isin(D.index)]])
    D = D.sort_index()

    #Reset Index - easier to apply on the for loop
    M = M.reset_index()

    #Initial check is to remove monthlies where 11 days or more are missing
    for i in range(0, len(count_for_month)):
        #Get the year and month 
        Year = count_for_month[Date_Column].loc[i].year
        Month = count_for_month[Date_Column].loc[i].month

        #Check for leap year and total days
        Total_Monthly_Count = [31,29 if Year % 4 == 0 else 28,31,30,31,30,31,31,30,31,30,31]

        #Chekc the number of days avalaible for that month
        Count_Month_Total = Total_Monthly_Count[Month-1]
        
        #Check the count
        Count = count_for_month.loc[i].loc[Type_Column] #Type column is Rainfall/Raindays/MLSP
    
        #If the count is less then 11, then we keep it
        if Count < Count_Month_Total - 11: #11 randomly missing days
            x = 0
        else:
            #Now we check whether it has 5 or more consecutive days missing
            #Checks current count
            Consec_NaNs = 0
            #Checks max count of the month
            Consec_Max = 0
                
            Months_Data = D.loc["{}-{}".format(Year, Month)].reset_index() #Extract the months daily data
            
            for q in range(0, len(Months_Data)):
                #If daily data is a nan, it will add to Consec NaNs
                if np.isnan(Months_Data[Type_Column].loc[q]) == True:
                    Consec_NaNs = Consec_NaNs + 1
                else:
                    #if not a nan, it will save the nan check count
                    Consec_Check = Consec_NaNs
                    #The save check count is then checked to see if its > then the max of that month
                    if Consec_Check >= Consec_Max:
                        #If so then the max check count is then updated
                        Consec_Max = Consec_Check
                    Consec_NaNs = 0

            #Once the month has been checked - if the max check count is 5 or more, then its a nan
            if Consec_Max >= 5: # this chekcs to on more then 5 consecutive months
               x = 0
            else: 
                Usuable.append(M.loc[i])
    
    #####################
    # Convert list of dictionaries to DataFrame
    Usuable = pd.DataFrame(Usuable)
    
    # Convert 'Date' column to datetime
    Usuable[Date_Column] = pd.to_datetime(Usuable[Date_Column])
    
    # Set 'Date' column as index
    Usuable.set_index(Date_Column, inplace=True)

    #Ensure all monthly dates are filled with NaNs
    start_date = Usuable.index.min()
    end_date = Usuable.index.max()
    # Create a monthly date range
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # 'MS' = Month Start
    # Create an empty DataFrame with these monthly dates as index and label it MSLP
    MSLP_Missing = pd.DataFrame(index=monthly_dates, columns=[Type_Column])
    MSLP_Missing.index.name = Date_Column  # Set the index name to 'Date'
    MSLP_Missing[:] = np.nan
    
    Usuable_infil = pd.concat([Usuable, MSLP_Missing[~MSLP_Missing.index.isin(Usuable.index)]])
    Usuable_infil = Usuable_infil.sort_index()
    
    return(Usuable_infil.round(1))


#-------------------------------
'''
Filter Into WMO guidelines - yearly corrected
'''
#-------------------------------
def WMO_Guide_Yearly(M,Date_Column,Type_Column,Raindays):
    Usuable = [] #WMO Standard 
    #The appendenation of the daily data into monthlies
    Usuable = [] #WMO Standard 
    #Since we have dropped all periods with missing days lets work out days avalaible
    count_for_year = M.resample('YS').count().reset_index()

    #Get it into monthlies
    if (Raindays == True):
        Y = M.resample('YS').sum()
    else:
        Y = M.resample('YS').mean()

    #Fill in all days where no data is avalaible with nans
    nan_dates = pd.date_range(start=M.index.min(), end=M.index.max(), freq='MS') 
    # Create an empty DataFrame with these monthly dates as index and label it MSLP
    Data_Missing = pd.DataFrame(index=nan_dates, columns=[Type_Column])
    Data_Missing.index.name = Date_Column  # Set the index name to 'Date'
    Data_Missing[:] = np.nan
    M = pd.concat([M, Data_Missing[~Data_Missing.index.isin(M.index)]])
    M = M.sort_index()

    #Reset Index - easier to apply on the for loop
    Y = Y.reset_index()

    #Initial check is to remove monthlies where 11 days or more are missing
    for i in range(0, len(count_for_year)):
        #Get the year and month 
        Year = count_for_year[Date_Column].loc[i].year

        #Check for leap year and total days
        Total_Yearly_Count = 12
        
        #Chekc the number of days avalaible for that month
        Count_Yearly_Total = Total_Yearly_Count
        
        #Check the count
        Count = count_for_year.loc[i].loc[Type_Column] #Type column is Rainfall/Raindays/MLSP
    
        #If the count is less then 11, then we keep it
        if Count < Count_Yearly_Total - 5: #5 randomly missing months
            x = 0
        else:
            #Now we check whether it has 5 or more consecutive days missing
            #Checks current count
            Consec_NaNs = 0
            #Checks max count of the month
            Consec_Max = 0
                
            Yearly_Data = M.loc["{}".format(Year)].reset_index() #Extract the months daily data
            
            for q in range(0, len(Yearly_Data)):
                #If daily data is a nan, it will add to Consec NaNs
                if np.isnan(Yearly_Data[Type_Column].loc[q]) == True:
                    Consec_NaNs = Consec_NaNs + 1
                else:
                    #if not a nan, it will save the nan check count
                    Consec_Check = Consec_NaNs
                    #The save check count is then checked to see if its > then the max of that month
                    if Consec_Check >= Consec_Max:
                        #If so then the max check count is then updated
                        Consec_Max = Consec_Check
                    Consec_NaNs = 0

            #Once the month has been checked - if the max check count is 5 or more, then its a nan
            if Consec_Max >= 3: # this chekcs to on more then 5 consecutive months
               x = 0
            else: 
                Usuable.append(Y.loc[i])
    
    #####################
    # Convert list of dictionaries to DataFrame
    Usuable = pd.DataFrame(Usuable)
    
    # Convert 'Date' column to datetime
    Usuable[Date_Column] = pd.to_datetime(Usuable[Date_Column])
    
    # Set 'Date' column as index
    Usuable.set_index(Date_Column, inplace=True)

    #Ensure all monthly dates are filled with NaNs
    start_date = Usuable.index.min()
    end_date = Usuable.index.max()
    # Create a monthly date range
    yearly_dates = pd.date_range(start=start_date, end=end_date, freq='YS')  # 'MS' = Month Start
    # Create an empty DataFrame with these monthly dates as index and label it MSLP
    MSLP_Missing = pd.DataFrame(index=yearly_dates, columns=[Type_Column])
    MSLP_Missing.index.name = Date_Column  # Set the index name to 'Date'
    MSLP_Missing[:] = np.nan
    
    Usuable_infil = pd.concat([Usuable, MSLP_Missing[~MSLP_Missing.index.isin(Usuable.index)]])
    Usuable_infil = Usuable_infil.sort_index()
    
    return(Usuable_infil.round(1))





#-------------------------------
'''
Extended summer period mean extraction Nov-1 to Mar+0
'''
#-------------------------------



def ext_sum_value(Data,summing = False,missing= 90):
    '''
    Extended summer period mean extraction Nov-1 to Mar+0
    '''

    
    # Extract year and month
    Data['year'] = Data.index.year
    Data['month'] = Data.index.month
    
    #----------------
    #3. Assign the summer to the summer in the next year aka 1831 - Nov-1830 to Mar-1831
    #----------------
    Data['season_year'] = Data['year']
    #If In 11 or 12, add to the next year
    Data.loc[Data['month'].isin([11, 12]), 'season_year'] = Data['year'] + 1  # Nov-Dec belong to next year's season
    
    # Filter to only include Nov–Mar which the adjustments above should make it the same year
    Data = Data[Data['month'].isin([11, 12, 1, 2, 3])]
    
    #Set Season adjusted date as month and season_year
    Data['Seasonal Date'] = pd.to_datetime({'year': Data['season_year'],'month': Data['month'],'day': Data.index.day})
    del Data['season_year']
    del Data['year']
    del Data['month']
    
    Data.set_index('Seasonal Date',inplace =  True)
    count = Data.dropna().resample('AS').count()
    
    if summing == True:
        Data = Data.resample('AS').sum().round(1)
        Data[count < missing] = np.nan
    
    else:
        Data = Data.resample('AS').mean().round(1)
        Data[count < missing] = np.nan

    return(Data)


#-------------------
#Running correlations over a yearly timescale
#---------------------

def running_correlation_with_p(data, col1, col2, window, min_valid_ratio=0.7):
    """
    Calculate running centered correlation and significance between two columns.
    NaN if any value missing in the window or if too much data is missing.

    Args:
        df: DataFrame with DateTime index
        col1: First column name
        col2: Second column name
        window: Window size (must be odd)
        min_valid_ratio: Minimum proportion of valid data required to calculate correlation
    
    Returns:
        pd.DataFrame with 'correlation' and 'significant' columns
    """
    if window % 2 == 0:
            raise ValueError("Window size must be odd for centered correlation.")
        
    half_window = window // 2 # gives the window to the lowest value - half a window is 1+5x2
    corr_values = []
    sig_values = []
    
    for i in range(len(data)):
        if i - half_window < 0 or i + half_window >= len(data):
            corr_values.append(np.nan)
            sig_values.append(np.nan)
            continue
        sub = data.iloc[i - half_window : i + half_window + 1][[col1, col2]]
        valid = sub.dropna()
        if len(valid) / window < min_valid_ratio:  #This checks to see if avalaible data that can be used, removes data with less then 70%
            # Too much missing data
            corr_values.append(np.nan)
            sig_values.append(np.nan)
            continue
        corr, p_value = pearsonr(valid[col1], valid[col2])
        corr_values.append(corr)
        significance = (
            '95%' if p_value < 0.05 else
            '90%' if p_value < 0.1 else
            '')
        sig_values.append(significance)
    result = pd.DataFrame({'correlation': corr_values,'significant': sig_values}, index=data.index)
    return result



#With the RHTested QQM get the data
def Quantile_Quantile_Matching(Q_step, Historical, Present, Hist_Dates, Pres_Date, show_counts=True,bounds = [0.05,0.95]):
    '''
    Q_step: float
    Step size used to build the quantiles.
    Must be greater than 0 and less than or equal to 0.1

    Historical: DataFrame
    Must have the date as the index
    Must contain columns:
        - tmax
        - tmin

    Present: DataFrame
    Must have the date as the index
    Must contain columns:
        - tmax
        - tmin

    Hist_Dates / Pres_Date: list or vector
    Start and end dates in a format that works with .loc slicing
    Example:
        Hist_Dates = ['1961-01-01', '1990-12-31']
        Pres_Date  = ['1991-01-01', '2020-12-31']

    show_counts: bool
    If True, prints a summary of how many values were:
        - below lower percentile bound
        - within percentile range
        - above upper percentile bound
        - missing
    '''

    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------
    if not (0 < Q_step <= 0.1):
        raise ValueError('Q_step must be greater than 0 and less than or equal to 0.1')

    required_columns = ['tmax', 'tmin']

    for df_name, df in [('Historical', Historical), ('Present', Present)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if len(missing_columns) > 0:
            raise ValueError(f"{df_name} is missing required columns: {missing_columns}")

    # ------------------------------------------------------------------
    # Set quantile steps
    # ------------------------------------------------------------------
    number = Q_step
    quantiles = np.round(np.arange(bounds[0], bounds[1] + number, number), 10)
    quantiles = quantiles[quantiles <= bounds[1] + 1e-12]

    lower_q = quantiles.min()
    upper_q = quantiles.max()

    # ------------------------------------------------------------------
    # Extract reference periods
    # ------------------------------------------------------------------
    Historical_30 = Historical.loc[Hist_Dates[0]:Hist_Dates[1], ['tmax', 'tmin']].copy()
    Present_30 = Present.loc[Pres_Date[0]:Pres_Date[1], ['tmax', 'tmin']].copy()

    # ------------------------------------------------------------------
    # Calculate quantiles
    # ------------------------------------------------------------------
    QHIS = Historical_30.quantile(quantiles).round(4)
    QPRE = Present_30.quantile(quantiles).round(4)

    # ------------------------------------------------------------------
    # Full historical record to adjust
    # ------------------------------------------------------------------
    Hist_All = Historical.reset_index().copy()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    Hist_Updated_Max = []
    Hist_Updated_Min = []
    Hist_Updated_Date = []

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    counts = {
        'tmax': {'below': 0, 'within': 0, 'above': 0, 'missing': 0},
        'tmin': {'below': 0, 'within': 0, 'above': 0, 'missing': 0}
    }

    # ------------------------------------------------------------------
    # Loop through full historical dataset
    # ------------------------------------------------------------------
    for i in range(0, len(Hist_All)):

        Hist_Updated_Date.append(Hist_All.iloc[i, 0])

        # ==============================================================
        # TMAX
        # ==============================================================
        if pd.isna(Hist_All['tmax'].loc[i]):
            Hist_Updated_Max.append(np.nan)
            counts['tmax']['missing'] += 1

        else:
            Temp_Old = Hist_All['tmax'].loc[i]

            Hist_Q = QHIS['tmax']
            Pres_Q = QPRE['tmax']

            Hist_QLow = Hist_Q.loc[lower_q]
            Hist_QHigh = Hist_Q.loc[upper_q]

            Pres_QLow = Pres_Q.loc[lower_q]
            Pres_QHigh = Pres_Q.loc[upper_q]
            #Between QQM adjusted, above added to upper tail, below subtracted to lower quatal
            if Hist_QLow <= Temp_Old <= Hist_QHigh:
                closest_quantile = (Hist_Q - Temp_Old).abs().idxmin()
                New_Temp = Pres_Q.loc[closest_quantile]
                Hist_Updated_Max.append(New_Temp)
                counts['tmax']['within'] += 1

            elif Temp_Old < Hist_QLow:
                Difference = Temp_Old - Hist_QLow
                New_Temp = Pres_QLow + Difference
                Hist_Updated_Max.append(New_Temp)
                counts['tmax']['below'] += 1

            elif Temp_Old > Hist_QHigh:
                Difference = Temp_Old - Hist_QHigh
                New_Temp = Pres_QHigh + Difference
                Hist_Updated_Max.append(New_Temp)
                counts['tmax']['above'] += 1

        # ==============================================================
        # TMIN
        # ==============================================================
        if pd.isna(Hist_All['tmin'].loc[i]):
            Hist_Updated_Min.append(np.nan)
            counts['tmin']['missing'] += 1

        else:
            Temp_Old = Hist_All['tmin'].loc[i]

            Hist_Q = QHIS['tmin']
            Pres_Q = QPRE['tmin']

            Hist_QLow = Hist_Q.loc[lower_q]
            Hist_QHigh = Hist_Q.loc[upper_q]

            Pres_QLow = Pres_Q.loc[lower_q]
            Pres_QHigh = Pres_Q.loc[upper_q]

            if Hist_QLow <= Temp_Old <= Hist_QHigh:
                closest_quantile = (Hist_Q - Temp_Old).abs().idxmin()
                New_Temp = Pres_Q.loc[closest_quantile]
                Hist_Updated_Min.append(New_Temp)
                counts['tmin']['within'] += 1

            elif Temp_Old < Hist_QLow:
                Difference = Temp_Old - Hist_QLow
                New_Temp = Pres_QLow + Difference
                Hist_Updated_Min.append(New_Temp)
                counts['tmin']['below'] += 1

            elif Temp_Old > Hist_QHigh:
                Difference = Temp_Old - Hist_QHigh
                New_Temp = Pres_QHigh + Difference
                Hist_Updated_Min.append(New_Temp)
                counts['tmin']['above'] += 1

    # ------------------------------------------------------------------
    # Build final DataFrame
    # ------------------------------------------------------------------
    Hist_Updated_Date = pd.DataFrame(Hist_Updated_Date, columns=['date'])
    Hist_Updated_Max = pd.DataFrame(Hist_Updated_Max, columns=['tmax'])
    Hist_Updated_Min = pd.DataFrame(Hist_Updated_Min, columns=['tmin'])

    Hist_Updated = pd.concat([Hist_Updated_Date, Hist_Updated_Max, Hist_Updated_Min], axis=1)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    Counts_Summary = pd.DataFrame(counts).T
    Counts_Summary['lower_quantile'] = lower_q
    Counts_Summary['upper_quantile'] = upper_q

    if show_counts:
        print('\nCount Summary:')
        print(Counts_Summary)

    return Hist_Updated, Counts_Summary

#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime


##HEATWAVE FUNCTION BASED ON HEATWAVE103 in my Heatwave Manuscript
def Heatwave_EHF_tmax_tmin_individual(Dataset, Dates_DataFrame, CDP_Matrix,
                         Percentile=85, window=7,
                         CDP_start_end_years=[1961, 1990],
                         Miss_Ratio=0.66):
    '''
    Parameters
    ----------
    Dataset : DataFrame
        A Tmax and Tmin Dataset that has index as numbers and not datetime.
        It should be in column form date_name, Tmax, Tmin
        datetime should be in format Year-Month-Day Already

    Dates_DataFrame : DataFrame
        This is just a DataFrame that has the dates of 366 days ready to be used where needed.

    CDP_Matrix : Array
        If set to [] then the functions and arguements relating to the CDP are irrelevant to the function by inputs should be
        for the function to work properly.

    Heatwave_Detail : True or False
        If True is selected the heatwaves will be expanded into more detail.

    Percentile : Integer/Decimal
        A number that is used for the CDP, it calculates the value where the temperature must exceed to be in
        that x percentile

    window : Integer
        Number of days either side of the day in focus that is used to calculate the percentile value in the CDP

    CDP_start_end_years : array of 2
        The years when the CDP should be calculated. Forms the basis of how many heatwaves we get

    Miss_Ratio : Value
        Determines the number of days that can be avaliable without causing a NaN in the Heat Stress Component

    RETURNS
    -----------------

    heatwaves : DataFrame
        The heatwave with all the relevant information.

    CDP : DataFrame
        Calendar Day Percentile so this can be inputted in the function again and save time.
    '''
    Dataset = Dataset.copy()
    Dates_DataFrame = Dates_DataFrame.copy()

    # Standardise first column name to 'date'
    if Dataset.columns[0] != 'date':
        Dataset = Dataset.rename(columns={Dataset.columns[0]: 'date'})
    if Dates_DataFrame.columns[0] != 'date':
        Dates_DataFrame = Dates_DataFrame.rename(columns={Dates_DataFrame.columns[0]: 'date'})

    # Force datetime
    Dataset['date'] = pd.to_datetime(Dataset['date'], errors='coerce')
    Dates_DataFrame['date'] = pd.to_datetime(Dates_DataFrame['date'], errors='coerce', dayfirst=True)

    if Dataset['date'].isna().any():
        raise ValueError("Dataset date column contains invalid dates.")

    Column_Dataset = Dataset.columns

    # Expanded version with year/month/day
    Dataset_Exp = Date_Splitter(Dataset)

    # CDP
    if len(CDP_Matrix) == 0:
        CDP_Max = Calendar_Day_Percentile(
            Dataset_Exp, Percentile, Column_Dataset[1],
            CDP_start_end_years[0], CDP_start_end_years[1],
            window, Dates_DataFrame
        )

        CDP_Min = Calendar_Day_Percentile(
            Dataset_Exp, Percentile, Column_Dataset[2],
            CDP_start_end_years[0], CDP_start_end_years[1],
            window, Dates_DataFrame
        )

        CDP = pd.concat(
            [CDP_Max[['date']], CDP_Max[[Column_Dataset[1]]], CDP_Min[[Column_Dataset[2]]]],
            axis=1
        )

    else:
        CDP = CDP_Matrix.copy()
        if CDP.columns[0] != 'date':
            CDP = CDP.rename(columns={CDP.columns[0]: 'date'})
        CDP['date'] = pd.to_datetime(CDP['date'], errors='coerce')

    # EHF
    EHF_Max, EHF_Min = EXCESS_HEAT_FACTOR(
        Dataset[['date', Column_Dataset[1], Column_Dataset[2]]].copy(),
        CDP,
        Miss_Ratio
    )

    # Use expanded dataset here, not raw Dataset
    Dataset_Date = Dataset_Exp.set_index('date').copy()

    # Remove helper columns
    Dataset_Date = Dataset_Date.drop(columns=['year', 'month', 'day'])

    # Rename EHF columns
    EHF_Max_Min_Col = EHF_Max.columns

    EHF_Max = EHF_Max.rename(columns={
        EHF_Max_Min_Col[1]: EHF_Max_Min_Col[1] + 'Max',
        EHF_Max_Min_Col[2]: EHF_Max_Min_Col[2] + 'Max',
        EHF_Max_Min_Col[3]: EHF_Max_Min_Col[3] + 'Max'
    })
    EHF_Max_Date = EHF_Max.set_index(EHF_Max_Min_Col[0])

    EHF_Min = EHF_Min.rename(columns={
        EHF_Max_Min_Col[1]: EHF_Max_Min_Col[1] + 'Min',
        EHF_Max_Min_Col[2]: EHF_Max_Min_Col[2] + 'Min',
        EHF_Max_Min_Col[3]: EHF_Max_Min_Col[3] + 'Min'
    })
    EHF_Min_Date = EHF_Min.set_index(EHF_Max_Min_Col[0])

    Full_Information_Vector = Dataset_Date.merge(EHF_Max_Date, left_index=True, right_index=True)
    Full_Information_Vector = Full_Information_Vector.merge(EHF_Min_Date, left_index=True, right_index=True)
    Full_Information_Vector = Full_Information_Vector.reset_index()

    # Binary flags
    Full_Information_Vector['EHFMx_Bin'] = (Full_Information_Vector['Excess Heat FactorMax'] > 0).astype(int)
    Full_Information_Vector['EHFMn_Bin'] = (Full_Information_Vector['Excess Heat FactorMin'] > 0).astype(int)
    Full_Information_Vector['EHMx_Bin'] = (Full_Information_Vector['Excess HeatMax'] > 0).astype(int)
    Full_Information_Vector['EHMn_Bin'] = (Full_Information_Vector['Excess HeatMin'] > 0).astype(int)

    Full_Information_Vector['EH_Bin'] = Full_Information_Vector['EHMx_Bin'] + Full_Information_Vector['EHMn_Bin']
    Full_Information_Vector['EHF_Bin'] = Full_Information_Vector['EHFMx_Bin'] + Full_Information_Vector['EHFMn_Bin']

    Heatwave_Finder = Full_Information_Vector[['date', 'EHFMx_Bin', 'EHF_Bin', 'EH_Bin']].copy()
    Heatwave_Finder['EHFMx_3Sum'] = Heatwave_Finder['EHFMx_Bin'].rolling(window=3, min_periods=1).sum()
    Heatwave_Finder['EHF_3Sum'] = Heatwave_Finder['EHF_Bin'].rolling(window=3, min_periods=1).sum()

    Warm_Spells_Matrix = Warmwaves(Full_Information_Vector, Heatwave_Finder)
    heatwaves = Heatwave_Extraction(Warm_Spells_Matrix)
    heatwaves_Detailed = Heatwave_Table_Generator(heatwaves)

    return heatwaves_Detailed, CDP, EHF_Max, EHF_Min


def Date_Splitter(Dataset):
    '''
    Parameters
    ----------
    Data : Dataframe
        CSV dataframe where the data is from.

    date_title : String
        Datetime Column Name for the extraction

    Returns
    -------
    Dataset : DataFrame
        DataFrame that has 3 new columns for Year Month and Day
    '''
    Dataset = Dataset.copy()

    # Only reset index if the index is not a simple default RangeIndex
    if not isinstance(Dataset.index, pd.RangeIndex):
        Dataset = Dataset.reset_index()

    # First column should be the date column
    date_col = Dataset.columns[0]

    # Force datetime
    Dataset[date_col] = pd.to_datetime(Dataset[date_col], errors='coerce')

    if Dataset[date_col].isna().any():
        raise ValueError(
            f"Date_Splitter: column '{date_col}' could not be fully converted to datetime."
        )

    Dataset['year'] = Dataset[date_col].dt.year
    Dataset['month'] = Dataset[date_col].dt.month
    Dataset['day'] = Dataset[date_col].dt.day

    return Dataset


def Calendar_Day_Percentile(Data, Percentile, Column_Name, start_year, end_year, window, Dates_DataFrame):
    '''
    Parameters
    ----------
    Data : Dataframe
        The DataFrame in the expanded date form with year, month and day done already.

    Percentile : Integer/Decimal
        A number that is used for the CDP, it calculates the value where the temperature must exceed to be in
        that x percentile

    Column_Name : String
        Determines if we are working out max or min temperatures

    start_year : Integer
        Year you want to start the CDP from

    end_year : Integer
        Year you want to end the CDP from

    Dates_DataFrame : DataFrame
        These are the 366 total days that the CDP function will append to so we can extract a day and month in the future
        when caculating the Excess Heat Factor

    Returns
    -------
    CDP : DataFrame
        Calendar Day Percentile of the entire year from the baseline and window chsoen in DataFrame format
    '''
    Data = Data.copy()
    Dates_DataFrame = Dates_DataFrame.copy()

    # Standardise dates frame column name if needed
    if Dates_DataFrame.columns[0] != 'date':
        Dates_DataFrame = Dates_DataFrame.rename(columns={Dates_DataFrame.columns[0]: 'date'})

    Dates_DataFrame['date'] = pd.to_datetime(Dates_DataFrame['date'], errors='coerce', dayfirst=True)

    # Only split if year/month/day are not already present
    if not {'year', 'month', 'day'}.issubset(Data.columns):
        Data = Date_Splitter(Data)
    else:
        date_col = Data.columns[0]
        Data[date_col] = pd.to_datetime(Data[date_col], errors='coerce')

    date_col = Data.columns[0]

    # Set date index
    Data_Extracted = Data.set_index(date_col).sort_index()

    # Baseline period
    Data_Extracted = Data_Extracted.loc[
        f'{start_year-1}-12-01':f'{end_year}-11-30'
    ]

    # Group by month/day
    group_days = Data_Extracted.groupby(['month', 'day'])

    # Build all 366 bins in the exact order of Dates_DataFrame
    Daily_Data = []
    for d in Dates_DataFrame['date']:
        key = (d.month, d.day)
        if key in group_days.groups:
            vals = group_days.get_group(key)[Column_Name].reset_index(drop=True)
        else:
            vals = pd.Series(dtype=float)
        Daily_Data.append(vals)

    CalendarDay = TnX_Rolling(window, Daily_Data, Percentile)

    CDP = pd.DataFrame({
        'date': Dates_DataFrame['date'].values,
        Column_Name: CalendarDay
    })

    return CDP


def TnX_Rolling(Window, Dataset, Percentile):
    '''
    Parameters
    ----------
    Window : Integer
        How many days before AND after that the CDP will use up

    Dataset : DataFrame
        It is the Daily_Data dataset that will be used from 3.

    Percentile : Integer/Decimal
        It is the percentile the temperature must reaach to be accepted

    Returns
    -------
        TnX : Series
        Array of length 366 of the CDP values.
    '''
    percent_to_quant = Percentile / 100
    TnX = []

    warnings.filterwarnings('ignore')

    for central_day in range(366):
        Temp_Storage = np.array([], dtype=float)

        for around_days in range(0, Window + 1):
            if around_days == 0:
                Temp_Storage = Dataset[central_day].to_numpy(dtype=float)
            else:
                if (central_day + around_days) > 365:
                    Window_Early_Year = central_day + around_days - 366
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[Window_Early_Year].to_numpy(dtype=float)))
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[central_day - around_days].to_numpy(dtype=float)))

                elif (central_day - around_days) < 0:
                    Window_Late_Year = central_day - around_days + 366
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[Window_Late_Year].to_numpy(dtype=float)))
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[central_day + around_days].to_numpy(dtype=float)))

                else:
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[central_day + around_days].to_numpy(dtype=float)))
                    Temp_Storage = np.concatenate((Temp_Storage, Dataset[central_day - around_days].to_numpy(dtype=float)))

        valid = Temp_Storage[~np.isnan(Temp_Storage)]

        if len(valid) == 0:
            Tn = np.nan
        else:
            Tn = np.quantile(valid, percent_to_quant)

        TnX.append(Tn)

    return TnX


def EXCESS_HEAT_FACTOR(Data, CDP_Data, Miss_Ratio):
    '''
    Parameters
    ----------
    Dataset : DataFrame
        A Tmax and Tmin Dataset that has index as numbers and not datetime.
        It should be in column form date_name, Tmax, Tmin
        datetime should be in format Year-Month-Day Already

    CDP_Data : DataFrame
        The calendar day percentile based off a percetnile where the temperature needs to reach to be in that percentile.

    Miss_Ratio : Value
        Determines the number of days that can be avaliable without causing a NaN in the Heat Stress Component

    Returns
    ----------
    Excess_Heat_Stress_Factor_Matrix_Max : DataFrame
        A DataFrame that includes the Excess Heat, Heat Stress and Excess Heat Factor variables for the tmax

    Excess_Heat_Stress_Factor_Matrix_Min : DataFrame
        A DataFrame that includes the Excess Heat, Heat Stress and Excess Heat Factor variables for the tmax
    '''
    Data = Data.copy()
    CDP_Data = CDP_Data.copy()

    Data_col = Data.columns
    CDP_col = CDP_Data.columns

    Data_Date = Data_col[0]
    CDP_Date = CDP_col[0]

    # Force datetime
    Data[Data_Date] = pd.to_datetime(Data[Data_Date], errors='coerce')
    CDP_Data[CDP_Date] = pd.to_datetime(CDP_Data[CDP_Date], errors='coerce')

    # Correct index setting
    Data_Date_I = Data.set_index(Data_Date).sort_index()
    CDP_Date_I = CDP_Data.set_index(CDP_Date).sort_index()

    # Heat Stress
    EHIacc_Max = Heat_Stress(Data_Date_I, Data_col[1], Miss_Ratio)
    EHIacc_Min = Heat_Stress(Data_Date_I, Data_col[2], Miss_Ratio)

    # Excess Heat
    EHIsig_Max = Excess_Heat(CDP_Date_I, CDP_col[1], Data_Date_I, Data_col[1])
    EHIsig_Min = Excess_Heat(CDP_Date_I, CDP_col[2], Data_Date_I, Data_col[2])

    Excess_Heat_Stress_Matrix_Max = pd.merge(EHIacc_Max, EHIsig_Max, how='left', on=[Data_Date])
    Excess_Heat_Stress_Matrix_Min = pd.merge(EHIacc_Min, EHIsig_Min, how='left', on=[Data_Date])

    # EHF
    EHF_Max = Excess_Heat_Factor_Calculator(Excess_Heat_Stress_Matrix_Max)
    EHF_Min = Excess_Heat_Factor_Calculator(Excess_Heat_Stress_Matrix_Min)

    Excess_Heat_Stress_Factor_Matrix_Max = pd.merge(
        EHF_Max, Excess_Heat_Stress_Matrix_Max, how='left', on=[Data_Date]
    )
    Excess_Heat_Stress_Factor_Matrix_Min = pd.merge(
        EHF_Min, Excess_Heat_Stress_Matrix_Min, how='left', on=[Data_Date]
    )

    return Excess_Heat_Stress_Factor_Matrix_Max, Excess_Heat_Stress_Factor_Matrix_Min


def Heat_Stress(Data, Max_Min_Ave_Col, Miss_Ratio):
    '''
    Parameters
    ----------
    Data : DataFrame
        This has the datetime as the index

    Max_Min_Col : Array
        The choose of choosing the max or min or average column to use from the dataset

    Miss_Ratio : Value
        Choose the amount of data missing before it can be excluded from the EHF calculation

    Returns
    ----------
    EHIacc_vector :  DataFrame
        The Heat Stress DataFrame
    '''
    # Extract the column
    Extracted_Data = Data[Max_Min_Ave_Col]

    # Reset the index to calculate the averages
    Extracted_Data = Extracted_Data.reset_index()
    Extracted_Data_col = Extracted_Data.columns

    # Necessary Columns to append
    date_Values = []
    EHIacc = []

    # Do the for loop
    for dt in np.arange(Extracted_Data.index[0] + 33, len(Data)):
        Date = Extracted_Data[Extracted_Data_col[0]].loc[dt]

        # 3-day mean where the day in focus is i
        length_3day = len(Extracted_Data[Max_Min_Ave_Col].loc[dt-2:dt].dropna())
        if length_3day < 3:
            mean_3_day = np.nan
        else:
            mean_3_day = Extracted_Data[Max_Min_Ave_Col].loc[dt-2:dt].mean()

        # 3 to 32 day mean
        length_30day = len(Extracted_Data[Max_Min_Ave_Col].loc[dt-32:dt-3].dropna())

        if length_30day < 30 * Miss_Ratio:
            mean_30_day = np.nan
        else:
            mean_30_day = Extracted_Data[Max_Min_Ave_Col].loc[dt-32:dt-3].dropna().mean()

        Heat_Stress_Value = mean_3_day - mean_30_day

        date_Values.append(Date)
        EHIacc.append(Heat_Stress_Value)

    EHIacc = pd.DataFrame(EHIacc, columns=['Heat Stress'])
    date_Values = pd.DataFrame(date_Values, columns=[Extracted_Data_col[0]])

    EHIacc_vector = pd.concat([date_Values, EHIacc], axis=1)

    return EHIacc_vector


def Excess_Heat(CDP, CDP_max_min_ave, Data, Max_Min_Ave_Col):
    '''
    Parameters
    ----------
    CDP : DataFrame
        The calendar day percentile based off a percetnile where the temperature needs to reach to be in that percentile.

    CDP_max_min_ave : string
        The choose of choosing max or min or average column to use from the CDP dataset

    Data : DataFrame
        This has the datetime as the index

    Max_Min_Col : string
        The choose of choosing the max or min or average column to use from the Data dataset

    Return
    ---------
    EHIsig_vector : DataFrame
        The Excess Heat DataFrame
    '''
    Extracted_Data = Data.reset_index().copy()
    date_col = Extracted_Data.columns[0]
    Extracted_Data[date_col] = pd.to_datetime(Extracted_Data[date_col], errors='coerce')

    if not isinstance(CDP.index, pd.DatetimeIndex):
        CDP = CDP.copy()
        CDP.index = pd.to_datetime(CDP.index, errors='coerce')

    date_Values = []
    EHIsig = []

    for dt in range(33, len(Extracted_Data)):
        Date = Extracted_Data.at[dt, date_col]

        cdp_key = pd.Timestamp(year=2020, month=Date.month, day=Date.day)

        if cdp_key in CDP.index:
            CDP_day = CDP.at[cdp_key, CDP_max_min_ave]
        else:
            CDP_day = np.nan

        Excess_Heat_Value = Extracted_Data.at[dt, Max_Min_Ave_Col] - CDP_day

        date_Values.append(Date)
        EHIsig.append(Excess_Heat_Value)

    EHIsig = pd.DataFrame(EHIsig, columns=['Excess Heat'])
    date_Values = pd.DataFrame(date_Values, columns=[date_col])

    EHIsig_vector = pd.concat([date_Values, EHIsig], axis=1)

    return EHIsig_vector


def Excess_Heat_Factor_Calculator(Excess_Heat_Stress_Matrix):
    '''
    Parameters
    ----------
    Excess_Heat_Stress_Matrix : DataFrame
        This is a DataFrame that combines the Excess Heat, Heat Stress together in one DataFrame

    Returns
    ----------
    EHF_vector : DataFrame
        This is the combination of the Excess Heat and Heat Stress as a value for each day.
    '''
    EH_col = Excess_Heat_Stress_Matrix.columns

    date_Values = []
    EHF = []

    # Make sure when there are 2 positive it remains positive, if there are two negatives it remains negative
    # and if one pos and one neg it remains negative
    for dt in np.arange(Excess_Heat_Stress_Matrix.index[0], len(Excess_Heat_Stress_Matrix)):

        Date = Excess_Heat_Stress_Matrix[EH_col[0]].loc[dt]
        HS = Excess_Heat_Stress_Matrix[EH_col[1]].loc[dt]
        EH = Excess_Heat_Stress_Matrix[EH_col[2]].loc[dt]

        if ((HS < 0) and (EH < 0)):
            EHF_single = -1 * EH * HS
        else:
            EHF_single = EH * HS

        date_Values.append(Date)
        EHF.append(EHF_single)

    EHF = pd.DataFrame(EHF, columns=['Excess Heat Factor'])
    date_Values = pd.DataFrame(date_Values, columns=[EH_col[0]])

    EHF_vector = pd.concat([date_Values, EHF], axis=1)

    return EHF_vector


def Warmwaves(Full_Dataset, binary_data):
    """
    Original initiation rule:
        - 3 days Tmax EHF > 0
        - at least 2 nights Tmin EHF > 0
      implemented as:
        EHFMx_3Sum == 3 and EHF_3Sum >= 5

    Continuation rule:
        - continue when EHIsig > 0

    Break rule:
        - break point when EHIsig <= 0
        - 1 or 2 consecutive break points are allowed
        - when the 3rd consecutive break point appears,
          the event is cut back to the LAST POSITIVE point

    Within-day point order:
        Tmin first, Tmax second
    """

    Data = Full_Dataset.copy().reset_index(drop=True)
    Bin = binary_data.copy().reset_index(drop=True)

    # Required columns
    req_data = ['date', 'Excess HeatMax', 'Excess HeatMin']
    req_bin = ['date', 'EHFMx_3Sum', 'EHF_3Sum']

    missing_data = [c for c in req_data if c not in Data.columns]
    missing_bin = [c for c in req_bin if c not in Bin.columns]

    if missing_data:
        raise KeyError(f"Warmwaves: missing required columns in Full_Dataset: {missing_data}")
    if missing_bin:
        raise KeyError(f"Warmwaves: missing required columns in binary_data: {missing_bin}")

    # Datetime + sort
    Data['date'] = pd.to_datetime(Data['date'], errors='coerce')
    Bin['date'] = pd.to_datetime(Bin['date'], errors='coerce')

    if Data['date'].isna().any():
        raise ValueError("Warmwaves: invalid dates in Full_Dataset['date']")
    if Bin['date'].isna().any():
        raise ValueError("Warmwaves: invalid dates in binary_data['date']")

    Data = Data.sort_values('date').reset_index(drop=True)
    Bin = Bin.sort_values('date').reset_index(drop=True)

    if len(Data) != len(Bin):
        raise ValueError("Warmwaves: Full_Dataset and binary_data are not the same length.")
    if not Data['date'].equals(Bin['date']):
        raise ValueError("Warmwaves: Full_Dataset and binary_data dates do not align.")

    # Build ordered point sequence: Tmin first, then Tmax
    points = []
    point_idx = 0

    for day_idx, row in Data.iterrows():
        points.append({
            'point_idx': point_idx,
            'day_idx': day_idx,
            'date': row['date'],
            'point_type': 'tmin',
            'ehisig': row['Excess HeatMin']
        })
        point_idx += 1

        points.append({
            'point_idx': point_idx,
            'day_idx': day_idx,
            'date': row['date'],
            'point_type': 'tmax',
            'ehisig': row['Excess HeatMax']
        })
        point_idx += 1

    points = pd.DataFrame(points)

    def positive_ehisig(x):
        return pd.notna(x) and (x > 0)

    def break_point(x):
        return pd.isna(x) or (x <= 0)

    def trailing_break_count(series):
        run = 0
        for val in reversed(list(series)):
            if break_point(val):
                run += 1
            else:
                break
        return run

    def last_positive_idx(series, index_values):
        pos = [idx for idx, val in zip(index_values, series) if positive_ehisig(val)]
        return pos[-1] if len(pos) > 0 else None

    active = False
    event_start_point = None
    last_included_point = None
    last_positive_point = None
    consecutive_breaks = 0
    events = []

    p = 0
    while p < len(points):
        row = points.loc[p]
        day_idx = row['day_idx']

        # ----------------------------------------------------------
        # Initiation: only check on Tmax point of the day
        # ----------------------------------------------------------
        if not active:
            if row['point_type'] == 'tmax':
                if (Bin.loc[day_idx, 'EHFMx_3Sum'] == 3) and (Bin.loc[day_idx, 'EHF_3Sum'] >= 5):
                    start_day_idx = day_idx - 2
                    if start_day_idx < 0:
                        start_day_idx = 0

                    event_start_point = 2 * start_day_idx
                    last_included_point = p

                    init_window = points.loc[event_start_point:p, 'ehisig']
                    init_index = points.loc[event_start_point:p].index

                    consecutive_breaks = trailing_break_count(init_window)
                    last_positive_point = last_positive_idx(init_window, init_index)

                    active = True

                    # If the initiation window already ends with 3+ break points,
                    # immediately trim back to the last positive point.
                    if consecutive_breaks > 2:
                        if last_positive_point is not None and last_positive_point >= event_start_point:
                            events.append((event_start_point, last_positive_point))
                        active = False
                        event_start_point = None
                        last_included_point = None
                        last_positive_point = None
                        consecutive_breaks = 0

            p += 1
            continue

        # ----------------------------------------------------------
        # Continuation / break logic using EHIsig
        # ----------------------------------------------------------
        if positive_ehisig(row['ehisig']):
            consecutive_breaks = 0
            last_included_point = p
            last_positive_point = p

        else:
            consecutive_breaks += 1

            # 1st and 2nd consecutive break points are still tentatively included
            if consecutive_breaks <= 2:
                last_included_point = p

            # 3rd consecutive break point:
            # END THE EVENT BACK AT THE LAST POSITIVE POINT
            else:
                if last_positive_point is not None and last_positive_point >= event_start_point:
                    events.append((event_start_point, last_positive_point))
                active = False
                event_start_point = None
                last_included_point = None
                last_positive_point = None
                consecutive_breaks = 0

        p += 1

    # If still active at the end of the record, keep the event through the
    # last included point, because a 3rd break never occurred.
    if active and (event_start_point is not None) and (last_included_point is not None):
        events.append((event_start_point, last_included_point))

    # No events found
    if len(events) == 0:
        Empty = Data.iloc[0:0].copy()
        if 'id' not in Empty.columns:
            Empty['id'] = pd.Series(dtype='int64')

        Empty = Empty.drop(
            columns=[
                c for c in [
                    'EHFMx_Bin', 'EHFMn_Bin',
                    'EHMx_Bin', 'EHMn_Bin',
                    'EH_Bin', 'EHF_Bin'
                ] if c in Empty.columns
            ],
            errors='ignore'
        )
        return Empty

    # Convert point events back to daily rows
    Warm_Spell_List = []
    for event_id, (start_p, end_p) in enumerate(events, start=1):
        event_dates = points.loc[start_p:end_p, ['date']].drop_duplicates().copy()
        event_dates['id'] = event_id
        Warm_Spell_List.append(event_dates)

    Warm_Spells = pd.concat(Warm_Spell_List, axis=0, ignore_index=True)

    Warm_Waves = Data.merge(Warm_Spells, on='date', how='inner')

    Warm_Waves = Warm_Waves.drop(
        columns=[
            c for c in [
                'EHFMx_Bin', 'EHFMn_Bin',
                'EHMx_Bin', 'EHMn_Bin',
                'EH_Bin', 'EHF_Bin'
            ] if c in Warm_Waves.columns
        ],
        errors='ignore'
    )

    return Warm_Waves.reset_index(drop=True)


def Heatwave_Extraction(Data):
    '''
    Parameters
    ----------
    Data : DataFrame
        The warm and heatwaves DataFrame
        date / Max / Min / Excess Heat FactorMax/Heat StressMax/Excess HeatMax/Excess Heat FactorMin/Heat StressMin/Excess HeatMin/id

    Returns
    ----------
    Heatwaves : DataFrames
        The warm and heatwaves DataFrame is then reduced to Nov to Mar aka the Extended Summer Season for heatwave research.
    '''
    if Data.empty:
        return Data.copy()

    Data_Col = Data.columns

    Hot_Per = Date_Splitter(Data)

    ext_sum_heatwave = Hot_Per[Hot_Per['month'] >= 11]
    ext_sum_heatwave2 = Hot_Per[Hot_Per['month'] <= 3]

    Extended_Summer_Season = pd.concat([ext_sum_heatwave, ext_sum_heatwave2]).sort_values(
        by=[Data_Col[0]], ascending=True
    )

    if Extended_Summer_Season.empty:
        return Extended_Summer_Season.drop(['day', 'month', 'year'], axis=1, errors='ignore')

    id_Max = Extended_Summer_Season['id']
    ids = id_Max.drop_duplicates(keep='first', inplace=False)

    for i in ids:
        CheckL = Extended_Summer_Season[Extended_Summer_Season['id'] == i]
        LeftCheck = CheckL[CheckL['day'] == 1]
        LeftCheck = LeftCheck[LeftCheck['month'] == 11]

        CheckR = Extended_Summer_Season[Extended_Summer_Season['id'] == i]
        RightCheck = CheckR[CheckR['day'] == 31]
        RightCheck = RightCheck[RightCheck['month'] == 3]

        if len(LeftCheck) == 1:
            Extended_Summer_Season = pd.concat(
                [Extended_Summer_Season, Hot_Per[Hot_Per['id'] == i]]
            ).sort_values(by=[Data_Col[0]], ascending=True)

        elif len(RightCheck) == 1:
            Extended_Summer_Season = pd.concat(
                [Extended_Summer_Season, Hot_Per[Hot_Per['id'] == i]]
            ).sort_values(by=[Data_Col[0]], ascending=True)

    Extended_Summer_Season = Extended_Summer_Season.drop_duplicates(subset=[Data_Col[0]], keep='first')
    Extended_Summer_Season = Extended_Summer_Season.drop(['day', 'month', 'year'], axis=1)

    Heatwaves = []
    id_n = 0
    for i in ids:
        Event = Extended_Summer_Season[Extended_Summer_Season['id'] == i]
        if len(Event) > 0:
            id_n = id_n + 1
            Event = Event.copy()
            Event['id'] = [id_n] * len(Event)
            Heatwaves.append(Event)

    if len(Heatwaves) == 0:
        return Extended_Summer_Season.iloc[0:0].copy()

    Heatwaves = pd.concat(Heatwaves, axis=0)

    return Heatwaves


def Heatwave_Table_Generator(Data):
    '''
    Parameters
    ----------
    Data : DataFrame
        The Heatwave dataframe

    Returns
    ----------
    Heatwaves : DataFrames
        An extension and clean up of the Heatwaves dataframe that provides more insight to the heatwaves.
    '''
    Cols = Data.columns
    Data = Data.rename(columns={Cols[1]: 'Tmax'})
    Data = Data.rename(columns={Cols[2]: 'Tmin'})
    Data = Data.rename(columns={Cols[3]: 'EHF_Mx'})
    Data = Data.rename(columns={Cols[4]: 'HS_Mx'})
    Data = Data.rename(columns={Cols[5]: 'EH_Mx'})
    Data = Data.rename(columns={Cols[6]: 'EHF_Mn'})
    Data = Data.rename(columns={Cols[7]: 'HS_Mn'})
    Data = Data.rename(columns={Cols[8]: 'EH_Mn'})

    Data = Data.round(1)
    Cols_2 = Data.columns

    Data['T_avg'] = (Data[Cols_2[1]] + Data[Cols_2[2]]) / 2
    Data['EHF_avg'] = (Data[Cols_2[3]] + Data[Cols_2[6]]) / 2
    Data['HS_avg'] = (Data[Cols_2[4]] + Data[Cols_2[7]]) / 2
    Data['EH_avg'] = (Data[Cols_2[5]] + Data[Cols_2[8]]) / 2

    duration = Data.groupby('id')['date'].agg([min, max]).reset_index()
    duration['Duration'] = (pd.to_datetime(duration['max']) - pd.to_datetime(duration['min'])).dt.days + 1

    Data = pd.merge(Data, duration[['id', 'Duration']], on='id')

    mean_values = Data.groupby('id')[[Cols_2[1], Cols_2[2], 'T_avg']].mean().reset_index()
    mean_values = mean_values.rename(columns={
        Cols_2[1]: 'Tmax HW Mean',
        Cols_2[2]: 'Tmin HW Mean',
        'T_avg': 'T_avg HW Mean'
    })
    Data = pd.merge(Data, mean_values, on='id')
    Data = Data.round(1)

    Cols_3 = Data.columns
    EHF_Means = Data.groupby('id')[
        [Cols_3[3], Cols_3[4], Cols_3[5], Cols_3[6], Cols_3[7], Cols_3[8], Cols_3[11], Cols_3[12], Cols_3[13]]
    ].mean().reset_index()

    EHF_Means = EHF_Means.rename(columns={
        Cols_3[3]: 'EHF_Mx HW Mean',
        Cols_3[4]: 'HS_Mx HW Mean',
        Cols_3[5]: 'EH_Mx HW Mean',
        Cols_3[6]: 'EHF_Mn HW Mean',
        Cols_3[7]: 'HS_Mn HW Mean',
        Cols_3[8]: 'EH_Mn HW Mean',
        Cols_3[11]: 'EHF_avg HW Mean',
        Cols_3[12]: 'HS_avg HW Mean',
        Cols_3[13]: 'EH_avg HW Mean'
    })

    Data = pd.merge(Data, EHF_Means, on='id')
    Data = Data.round(1)

    Cols_4 = Data.columns

    Data['Total Excess Heat Factor'] = Data[Cols_4[3]] + Data[Cols_4[6]]

    Total_Intenisty = Data.groupby('id')['Total Excess Heat Factor'].mean().reset_index()
    Total_Intenisty = Total_Intenisty.rename(columns={'Total Excess Heat Factor': 'Total Intenisty'})
    Data = pd.merge(Data, Total_Intenisty, on='id')
    Data = Data.round(1)

    def calculate_peak_intensity(event_id):
        event_data = Data[Data['id'] == event_id]
        top_3_factors = event_data['Total Excess Heat Factor'].nlargest(3)
        intensity = top_3_factors.mean()
        return intensity

    Data['Peak Intensity'] = Data['id'].apply(calculate_peak_intensity)
    Data = Data.round(1)

    def assign_rhc_category(intensity, duration):
        if intensity < 15 and duration <= 4:
            return 'RHC Cat 1'
        elif intensity < 15 and duration > 4:
            return 'RHC Cat 2'
        elif intensity >= 15 and intensity < 30 and duration <= 4:
            return 'RHC Cat 2'
        elif intensity >= 15 and intensity < 30 and duration > 4:
            return 'RHC Cat 3'
        elif intensity >= 30 and intensity < 45 and duration <= 4:
            return 'RHC Cat 3'
        elif intensity >= 30 and intensity < 45 and duration > 4:
            return 'RHC Cat 4'
        elif intensity >= 45 and intensity < 60 and duration <= 4:
            return 'RHC Cat 4'
        elif intensity >= 45 and intensity < 60 and duration > 4:
            return 'RHC Cat 5'
        elif intensity >= 60 and duration >= 3:
            return 'RHC Cat 5'

    Data['Rowes Heatwave Categorisation'] = Data.apply(
        lambda x: assign_rhc_category(x['Peak Intensity'], x['Duration']), axis=1
    )

    C5 = Data.columns

    Data = Data.reindex(columns=[
        C5[0],
        C5[9],
        C5[30],
        C5[1],
        C5[2],
        C5[14],
        C5[29],
        C5[3],
        C5[4],
        C5[5],
        C5[6],
        C5[7],
        C5[8],
        C5[10],
        C5[11],
        C5[12],
        C5[13],
        C5[15],
        C5[16],
        C5[17],
        C5[18],
        C5[19],
        C5[20],
        C5[21],
        C5[22],
        C5[23],
        C5[24],
        C5[25],
        C5[26],
        C5[27],
        C5[28]
    ])

    return Data
