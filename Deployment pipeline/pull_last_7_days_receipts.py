def read_last_7_days():
    import pandas as pd
    import numpy as np

    import sqlalchemy

    from datetime import datetime




    import os

    # Loading environment variables
    from dotenv import load_dotenv
    load_dotenv()


    # connecting to database using URL connection string
    sqlUrl = sqlalchemy.engine.URL.create(
        drivername = os.getenv('drivername'),
        username = os.getenv('username'),
        password = os.getenv('password'),
        host = os.getenv('host'),
        port = os.getenv('port'),
        database = os.getenv('database')
    )

    engine = sqlalchemy.create_engine(sqlUrl)


    sql_str = """select a.Receipt_id,
                    a.Venue_id,
                    a.Total_Nbr_of_Items,
                    a.Total_Price,
                    a.Receipt_Date,
                    b.Venue,
                    c.Item_name,
                    c.Item_Price

                    from hs.receipt as a left join
                    hs.venue_details as b
                    on a.Venue_id = b.Venue_id

                    left join hs.item as c
                    on a.Receipt_id = c.Receipt_id

                    where a.Receipt_date > curdate() -7;"""

    last_7_days = pd.read_sql_query(sql_str,engine)

    # drop Venue_id
    last_7_days.drop(['Venue_id'], axis=1, inplace=True)

    # convert Receipt_Date to datetime
    last_7_days['Receipt_Date'] =  pd.to_datetime(last_7_days.Receipt_Date, format="%Y-%m-%d")

    return last_7_days

def attach_new_receipt_data(input_data, venue_g, Total_Nbr_of_Items_g, Total_Price_g, Receipt_Date_g, Item_name_g, Item_Price_g):
    import pandas as pd
    import numpy as np
    last_7_days = input_data

    # create the new receipt_id
    receipt_id_g = last_7_days.Receipt_id.max() + 1

    # loop through the guest receipt and concatenate to last_7_days
    for i in range(len(Item_name_g)):
        new_row_i = {
            'Receipt_id': receipt_id_g,
            'Total_Nbr_of_Items': Total_Nbr_of_Items_g,
            'Total_Price': Total_Price_g,
            'Receipt_Date': Receipt_Date_g,
            'Venue': venue_g,
            'Item_name': Item_name_g[i],
            'Item_Price': Item_Price_g[i]
            
        }
        new_row_i = pd.DataFrame(new_row_i, columns=last_7_days.columns, index=[0])
        last_7_days = pd.concat([last_7_days, new_row_i], ignore_index=True)
    
    # continue on with deriving features

    # Working with receipt data only for deriving receipt features
    last_7_days_date_diff =\
    last_7_days[['Receipt_id','Receipt_Date',
                'Total_Price','Total_Nbr_of_Items']].drop_duplicates(subset=['Receipt_id'])
    
    # sort the data by Receipt_Date
    last_7_days_date_diff.sort_values('Receipt_Date',ascending=True, inplace=True)

    # derive Date_diff
    last_7_days_date_diff['Date_diff'] =\
    (last_7_days_date_diff.Receipt_Date - last_7_days_date_diff.Receipt_Date.shift()).dt.days.fillna(0)

    # converting Receipt_Date to datetime to avoid error
    last_7_days_date_diff['Receipt_Date'] = pd.to_datetime(last_7_days_date_diff.Receipt_Date, format='%Y-%m-%d')

    # Deriving week_of_year
    last_7_days_date_diff['week_of_year'] = last_7_days_date_diff.Receipt_Date.dt.isocalendar().year.map(str)+ "_" +  \
    last_7_days_date_diff.Receipt_Date.dt.isocalendar().week.map(str)


    # Calculate number of items per week
    last_7_days_date_diff['Nbr_items_per_wk'] =\
    last_7_days_date_diff.groupby(['week_of_year'])['Total_Nbr_of_Items'].transform('sum')

    # Calculate expenditure per week
    last_7_days_date_diff['Expenditure_per_wk'] = \
    last_7_days_date_diff.groupby(['week_of_year'])['Total_Price'].transform('sum')

    # Calculating receipt Total_Price as a percentage of the weeks expenditure
    last_7_days_date_diff['Total_Exp_wk_perc'] = \
    last_7_days_date_diff.Total_Price / last_7_days_date_diff.Expenditure_per_wk

    # merge back
    last_7_days = pd.merge(
        last_7_days,
        last_7_days_date_diff[['Receipt_id','Date_diff','week_of_year',
                            'Expenditure_per_wk','Total_Exp_wk_perc',
                            'Nbr_items_per_wk']],
        on='Receipt_id',
        how='left'
    )

    return last_7_days