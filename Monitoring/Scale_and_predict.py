#*********************** Import Models ***********************
import os
import pandas as pd
import numpy as np

import sqlalchemy

from datetime import datetime

from sklearn.preprocessing import OneHotEncoder

import joblib

#*************************************************************
# Loading environment variables
from dotenv import load_dotenv
load_dotenv()

# connecting to MySQL database using URL connection string
sqlUrl = sqlalchemy.engine.URL.create(
    drivername = os.getenv('drivername'),
    username = os.getenv('username'),
    password = os.getenv('password'),
    host = os.getenv('host'),
    port = os.getenv('port'),
    database = os.getenv('database')
)

engine = sqlalchemy.create_engine(sqlUrl)

#*************************************************************

def transform_data(Venue_id_g,Total_Nbr_of_Items_g,Total_Price_g,
            Receipt_Date_g,Receipt_Time_g,Payment_Type_g,
            Card_Source_g,Item_name_g,Item_Price_g
           ):
    sql_str = """select *,yearweek(Receipt_date) as week_of_year
                from hs.receipt
                where yearweek(Receipt_date) = yearweek(curdate())
                order by Receipt_id desc;"""
    
    Receipt = pd.read_sql_query(sql_str,engine)
    Receipt.drop(['Receipt_Nbr','Trans_number','Barcode','Date_Added'], axis=1, inplace=True)
    
    Receipt['Receipt_Time'] = Receipt['Receipt_Time'].apply(lambda x: 
                                              datetime.strptime(str(x).split(' ')[2],
                                                                '%H:%M:%S'
                                                               ).time())
    
    receipt_ids = tuple(Receipt.Receipt_id.values.tolist())    
    new_receipt_id = max(receipt_ids) + 1    
    Receipt_id_g = new_receipt_id
    
    week_of_year_g = max(tuple(Receipt.week_of_year.values.tolist()))
    
    # add new receipt
    new_row_r = {'Receipt_id':Receipt_id_g,
           'Venue_id': Venue_id_g,
           'Total_Nbr_of_Items': Total_Nbr_of_Items_g,
           'Total_Price': Total_Price_g,
           'Receipt_Date': Receipt_Date_g.date(),
           'Receipt_Time': Receipt_Time_g,
           'week_of_year': week_of_year_g
          }
    
    new_row_r = pd.DataFrame(new_row_r, columns=Receipt.columns, index=[0])
    Receipt = pd.concat([Receipt,new_row_r], ignore_index=True)
    
    sql_str0 = """select * from hs.payment
                where Receipt_id in {}
                order by Payment_id desc;""".format(receipt_ids)
    
    Payment = pd.read_sql_query(sql_str0,engine)
    Payment.drop(['Card_Nbr','Aid','Auth_Code','Date_Added'],axis=1, inplace=True)
    
    Payment_id_g = max(tuple(Payment.Payment_id.values.tolist())) + 1
    
    new_row_p = {'Payment_id': Payment_id_g, 
           'Receipt_id':Receipt_id_g,
           'Payment_Type': Payment_Type_g,
           'Card_Source': Card_Source_g
          }
    
    new_row_p = pd.DataFrame(new_row_p, columns=Payment.columns, index=[0])
    
    Payment = pd.concat([Payment,new_row_p], ignore_index=True)
    
    sql_str1 = """select * from hs.item
                where Receipt_id in {}
                order by Item_id desc;""".format(receipt_ids)
    
    Item = pd.read_sql_query(sql_str1,engine)
    
    Item.drop(['Venue_Item_code','Venue_id','Date_Added'], axis=1, inplace=True)
    
    Item_id_g = max(tuple(Item.Item_id.values.tolist())) + 1
    
    for i in range(len(Item_name_g)):
        new_row_i = {'Item_id': Item_id_g,
                     'Receipt_id': Receipt_id_g,
                     'Item_name':Item_name_g[i],
                     'Item_Price':Item_Price_g[i]
                    }
        new_row_i = pd.DataFrame(new_row_i, columns=Item.columns, index=[0])
        Item = pd.concat([Item,new_row_i], ignore_index=True)
        
    # Excluding refunds i.e. negative Total_Price
    Receipt = Receipt[Receipt.Total_Price > 0]
    
    # Deriving the difference in days between shopping trips

    # sorting the data in ascending date order
    Receipt.sort_values('Receipt_Date',ascending=True, inplace=True)

    # calculating the date difference using the shift() method to get the lag -1 value
    # and retrieving the numeric part of date difference
    Receipt['Date_diff'] = (Receipt.Receipt_Date - Receipt.Receipt_Date.shift()).dt.days

    # Deriving the weekdate
    week = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # [week[x.weekday()] for x in Receipt.Receipt_Date]
    Receipt['Week_day'] = Receipt.Receipt_Date.apply(lambda x: week[x.weekday()])

    # Retrieving only the numeric version of weekday()
    # adding 1 to remove the 0 for the first day
    Receipt['Week_day_numeric'] = Receipt.Receipt_Date.apply(lambda x: x.weekday()) + 1
    
    # Calculate the number of trips per week
    Receipt['Nbr_trips_per_wk'] = Receipt.groupby(['week_of_year'])['Receipt_id'].transform('count')

    # Calculate number of items bought per week
    Receipt['Nbr_items_per_wk'] = Receipt.groupby(['week_of_year'])['Total_Nbr_of_Items'].transform('sum')
    # Calculating receipt Total_Nbr_of_Items as a percentage of the weeks Total_Nbr_of_Items 
    Receipt['Nbr_items_wk_perc'] = Receipt.Total_Nbr_of_Items / Receipt.Nbr_items_per_wk

    # Calculate expenditure per week
    Receipt['Expenditure_per_wk'] = Receipt.groupby(['week_of_year'])['Total_Price'].transform('sum')
    # Calculating receipt Total_Price as a percentage of the weeks expenditure
    Receipt['Total_Exp_wk_perc'] = Receipt.Total_Price / Receipt.Expenditure_per_wk
    
    # extract the time of day as morning, afternoon, evening etc
    Receipt['hour'] = Receipt['Receipt_Time'].apply(lambda x: x.hour)

    bins_= [0,7,11,17,20,23]
    lbl = ['Early','Morning','Afternoon','Evening','Late_night']

    Receipt['Part_of_day'] = pd.cut(Receipt['hour'],bins=bins_, labels=lbl, include_lowest=True)
    
    # merging Payment to Receipt for the analysis
    Receipt_Payment = pd.merge(Receipt, 
                               Payment[['Receipt_id','Payment_Type','Card_Source']], 
                               on='Receipt_id', how='left')
    
    raw0 = pd.merge(Receipt_Payment,
                     Item[['Receipt_id','Item_id','Item_name','Item_Price']], 
                     on='Receipt_id', 
                     how='left' )
    
    # Grouping items and creating new features
    
    # breads
    nots = ['garlic','Ham','Garlic','ham']
    ins = ['Bread','bloomer','bread','Bloomer']

    raw0['Bread'] = raw0.Item_name.apply(lambda sentence: 1 if any(word in sentence for word in ins) 
                                and not any(word in sentence for word in nots) else 0)
    
    # Cooked meats indicator
    not_cook = ['glass','shampoo','water','conditioner','champagne']
    cooked_meats = ['chicken pasty Slices twin pack','steak and kidney pasty',
                    'chicken cooked','cooked chicken','roast chicken thighs',
                    'mackerel','Salmon','pork pies classic','ham',
                    'sardines','sausages cocktail','spicy chorizo sausages',
                    'sausages rolls','sausage rolls','salami','meatballs']

    raw0['Cooked_meats'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                any(word.lower() in sentence.lower() 
                                                    for word in cooked_meats) 
                                                and not any(word.lower() in sentence.lower() 
                                                            for word in not_cook) 
                                                else 0)
    
    # Raw meats indicator
    not_raw = ['pasty','cooked','roast','seasoning',
               'southern','fried','meal','piece',
               'box','bake','szechuan','pies',
               'mushroom','pie','salami','rolls','cocktails',
               'chips','chorizo']
    raw_meats = ['bacon','chicken','lamb','gammon','sausages',
                 'sausage','pork','fish','beef','eggs']

    raw0['Raw_meats'] = raw0.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower() for word in raw_meats) 
                                and not any(word.lower() in sentence.lower() for word in not_raw) else 0)
    
    # Creating eating out indicator using the restaurants and fastfoods Venue id
    eating_out = [11,20,25,31,34,35,40,41,42,48]
    raw0['Eating_out'] = raw0.apply(lambda x: 1 if x['Venue_id'] in eating_out or 
                                             x['Item_name'] in ['Food @ space centre',
                                                             'Drinks @ space centre'] 
                                             else 0,axis=1)

    # Creating snack indicator
    not_snack = ['diesel','james']
    snacks = ['snickers','digestive','digestives',
              'chocolate','yogurt','cake','cakes',
              'snack','nuts','donuts','doughnut',
              'mikati','fudge','maltesers','twix','marmalade',
              'jam','custard']

    raw0['Snacks'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                          any(word.lower() in sentence.lower() for word in snacks)
                                          and not 
                                          any(word.lower() in sentence.lower() for word in not_snack)
                                         else 0)
    
    # Creating drinks indicator variable
    # Note: this includes alcoholic and non alcoholic drinks

    not_drink = ['diesel','glass','socks','fan','heater',
                 'beef','source','ironing','plaster','ham','lockets']

    drinks = ['juice','vimto','ribena','squash','tropical','liquer',
              'dr pepper','coke','alcohol','beer','rubicon','courvoisier'
              'wine','irish','port','rum','original','smoothies','water',
              'honey','cordial','whiskey']

    raw0['Drinks'] = raw0.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower()
                                                                    for word in drinks) 
                                          and not any(word.lower() in sentence.lower() 
                                                      for word in not_drink)
                                         else 0)
    
    # Creating a vegetables indicator
    not_veg = ['seed','bread','fried','black','dr','lisbon']
    vegetables = ['cabbage','carrots','parsnip','greens','garlic','ginger',
                  'tomatoes','onions','chillies','ngai ngai','leaf',
                  'leaves','mushrooms','spinach','coriander','parsley',
                  'broccoli','pumpkin','peas','peppers','cucumber','leeks',
                 'brussel sprouts','mint','asparagus','beans']

    raw0['Vegetables'] = raw0.Item_name.apply(lambda sentence: 
                                              1 if any(word.lower() in sentence.lower() 
                                                       for word in vegetables)
                                             and not any(word.lower() in sentence.lower() 
                                                         for word in not_veg)
                                             else 0)
    
    # Creating a fruits indicator
    not_fruit = ['juice','rubicon','original','smoothies','yogurt','cordial',
                 'ribena','squash','volvic','water','lockets','bucket']
    fruit = ['olives','apples','mango','grape','grapes','bananas',
              'lime','lemon','strawberries','oranges']

    raw0['Fruit'] = raw0.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower() 
                                                                   for word in fruit) 
                                         and not any(word.lower() in sentence.lower() 
                                                     for word in not_fruit)
                                        else 0)
    
    # Creating an indicator for cooking base
    not_base = ['fried']
    cooking_base = ['pasta','spaghetti','rice','flour','potatoe','potatoes','potato']

    raw0['Cooking_base'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                any(word.lower() in sentence.lower() 
                                                    for word in cooking_base) 
                                                and not any(word.lower() in sentence.lower() 
                                                            for word in not_base) 
                                                else 0)
    
    # Creating an indicator for Dairy produce
    dairy_produce = ['cheese','brilliantly','butter','butterlicious','spread','margarine']
    raw0['Dairy_produce'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                 any(word.lower() in sentence.lower() 
                                                     for word in dairy_produce) 
                                                 else 0)
    
    # Creating an indicator for seasoning
    seasoning = ['black pepper','salt','seasoning','spice','cinnamon','paprika']

    raw0['Seasoning'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                             any(word.lower() in sentence.lower() 
                                                 for word in seasoning) 
                                             else 0 )
    
    # creating an indicator for breakfast food
    breakfast = ['granola','muesli','sultanas']

    raw0['Breakfast'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                             any(word.lower() in sentence.lower() 
                                                 for word in breakfast) 
                                             else 0 )
    
    # Creating an indicator for transport
    transport = ['unleaded','diesel','return ticket']
    raw0['Transport'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                             any(word.lower() in sentence.lower() 
                                                 for word in transport)
                                             else 0)
    
    # Creating an indicator for diy
    not_diy = ['sony']
    diy = ['fifty box 44l black','soil scoop','garden glove','compost','carrot amsterdam',
           'cabbage copenhagen','parsnip gladiator','Spring onion white lisbon seed','bucket',
           'onion white ailsa craig seed','dahlia assorted bright seed','gorilla',
           'kaze box','galvanised garden wheelbarrow','magnusson 500mm steel ruler',
           'tape measure','timber','bosch 34 piece drill accessory','wood screw steel',
           'bolted screws set','decking srew csk pz pk500','heavy duty rubble sacks 50l',
           'magnusson screw driver slot 100 x','wiha slotted screw driver 150 x',
           'general purpose plier set 3pc','mag ratchet precision Screwdriver',
           'diall l75 decking screws 250pck','chrome plated barrel latch','wire',
           'satin nickel barrel latch','chrome plated barrel latch',
           'ronseal varnish outdoor clear gloss','zipper metal silver teeth',
           'neodymium magnets']

    raw0['DIY'] = raw0.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower() 
                                                                 for word in diy) 
                                       and not any(word.lower() in sentence.lower() 
                                                   for word in not_diy) 
                                       else 0)
    
    # Creating an indicator for electronics
    electronics = ['macallister combi drill','macallister multipendulum jigsaw 600w',
                   'bench table saw','fan heater','voltmeter','hair clippers wahl',
                   'silk steamer','vacuum cleaner','table saw',
                   'rotary tool kit','reciprocating saws','air fryer oven']

    raw0['Electronics'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                               any(word.lower() in sentence.lower() 
                                                   for word in electronics) 
                                               else 0)
    
    # creating an indicator for education
    not_edu = ['clevo']
    education = ['King Richard Williams','linkedin','mysql','financial','python',
                 'bootcamp','web server','linux','apache','sas','pencils',
                 'eraser','WHS 15cm ruler','bic pen','a4','binders']

    raw0['Education'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                             any(word.lower() in sentence.lower() 
                                                 for word in education) 
                                             and not 
                                             any(word.lower() in sentence.lower() 
                                                 for word in not_edu) 
                                             else 0)

    # Creating an indicator for tech and services
    tech_and_services = ['macbook pro','flash drive','lonovo','sony','clevo','tesco','domain registration',
                         'membership payment']

    raw0['Tech_and_services'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                     any(word.lower() in sentence.lower() 
                                                         for word in tech_and_services) 
                                                     else 0)

    # Creating an indicator for cosmetics and self care

    not_cosmetic = ['sony']
    cosmetics_and_selfcare = ['shampoo','shower','tooth','colgate','wisdom','nivea',
                              'razor','body','blades','aqueous','shave','african',
                              'perfume','brut','roll on','Roll-on','bettina bath']

    raw0['Cosmetics_and_selfcare'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                          any(word.lower() in sentence.lower() 
                                                              for word in cosmetics_and_selfcare) 
                                                          and not 
                                                          any(word.lower() in sentence.lower() 
                                                              for word in not_cosmetic) 
                                                          else 0)
    
    # Creating an indicator for clothes and shoes
    clothes_and_shoes = ['lonsdale','slaz','trainers','addidas','puma','sondico','nike',
                         'umbrella','trousers','socks','shirt','boxers','gloves','boots',
                         'insoles']
    raw0['Clothes_and_shoes'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                     any(word.lower() in sentence.lower() 
                                                         for word in clothes_and_shoes) 
                                                     else 0)

    # creating an indicator for house and kitchen 
    not_house = ['cake']
    house_and_kitchen = ['fairy liquid','measure jug','fitted bed sheet','glass',
                         'turner (spatula)','rolling pin','fairy liquid','orange citrus',
                         'turkey baster','dish drainer','extension lead','grater','power spray',
                         'liquid','roaster and rack','kitchen roller','salad tongs',
                         'strainer 12cm','arial pods','metal scourer','bathmat','curtain hooks',
                         'ant killer spray','scouring pads','sponge','surf','foil','plaster',
                         'knife sharpener','electric hand mixer','athena cotton wool','mop',
                         'ofargo']

    raw0['House_and_kitchen'] = raw0.Item_name.apply(lambda sentence: 1 if 
                                                     any(word.lower() in sentence.lower() 
                                                         for word in house_and_kitchen) 
                                                     and not 
                                                     any(word.lower() in sentence.lower() 
                                                         for word in not_house) 
                                                     else 0)
    
    indicator_list = ['Bread','Cooked_meats','Raw_meats','Eating_out','Snacks','Drinks',
                  'Vegetables','Fruit','Cooking_base','Dairy_produce','Seasoning',
                  'Breakfast','Transport','DIY','Electronics','Education',
                  'Tech_and_services','Cosmetics_and_selfcare','Clothes_and_shoes',
                  'House_and_kitchen']

    # Looping through the indicator list to derive new features
    for x in indicator_list:
        # Calculate item x count by shopping trip(Receipt) and by week
        raw0["{}_receipt".format(x)] = raw0.groupby(['Receipt_id'])[x].transform('sum')
        raw0["{}_wk".format(x)] = raw0.groupby(['week_of_year'])[x].transform('sum')

        # Receipt item x as a proportion of week's item x
        raw0["{}_wk_perc".format(x)] = raw0["{}_receipt".format(x)] / raw0["{}_wk".format(x)]

        # Calculating item x expenditure by shopping trip(Receipt) and by week
        raw0["{}_exp_receipt".format(x)] = \
        raw0.query("{}==1".format(x)).groupby(['Receipt_id',x])['Item_Price'].transform('sum')

        raw0["{}_exp_wk".format(x)] = \
        raw0.query("{}==1".format(x)).groupby(['week_of_year',x])['Item_Price'].transform('sum')

        # Receipt item x expenditure as a proportion of week's item x expenditure
        raw0["{}_wk_exp_perc".format(x)] = \
        raw0["{}_exp_receipt".format(x)] / raw0["{}_exp_wk".format(x)]
        
        # Filling in the single missing value for Date_diff
        raw0['Date_diff'].fillna(0,inplace=True)
        
    # Filling in missing values from the check above
    fill_na_list = ['Bread_wk_perc','Bread_exp_receipt','Bread_exp_wk',
                 'Bread_wk_exp_perc','Cooked_meats_receipt','Cooked_meats_wk','Cooked_meats_wk_perc','Cooked_meats_exp_receipt',
                 'Cooked_meats_exp_wk','Raw_meats_receipt','Raw_meats_wk','Raw_meats_wk_perc','Raw_meats_exp_receipt',
                 'Raw_meats_exp_wk','Raw_meats_wk_exp_perc','Eating_out_receipt','Eating_out_wk','Eating_out_wk_perc',
                 'Eating_out_exp_receipt','Eating_out_exp_wk','Eating_out_wk_exp_perc','Snacks_receipt','Snacks_wk',
                 'Snacks_wk_perc','Snacks_exp_receipt','Snacks_exp_wk','Snacks_wk_exp_perc','Drinks_receipt','Drinks_wk',
                 'Drinks_wk_perc','Drinks_exp_receipt','Drinks_exp_wk','Drinks_wk_exp_perc','Vegetables_receipt',
                 'Vegetables_wk','Vegetables_wk_perc','Vegetables_exp_receipt','Vegetables_exp_wk','Vegetables_wk_exp_perc',
                 'Fruit_receipt','Fruit_wk','Fruit_wk_perc','Fruit_exp_receipt','Fruit_exp_wk','Fruit_wk_exp_perc',
                 'Cooking_base_receipt','Cooking_base_wk','Cooking_base_wk_perc','Cooking_base_exp_receipt','Cooking_base_exp_wk',
                 'Cooking_base_wk_exp_perc','Dairy_produce_receipt','Dairy_produce_wk','Dairy_produce_wk_perc',
                 'Dairy_produce_exp_receipt','Dairy_produce_exp_wk','Dairy_produce_wk_exp_perc','Seasoning_receipt',
                 'Seasoning_wk','Seasoning_wk_perc','Seasoning_exp_receipt','Seasoning_exp_wk','Seasoning_wk_exp_perc',
                 'Breakfast_receipt','Breakfast_wk','Breakfast_wk_perc','Breakfast_exp_receipt','Breakfast_exp_wk',
                 'Breakfast_wk_exp_perc','Transport_wk','Transport_wk_perc','Transport_exp_receipt','Transport_exp_wk',
                 'Transport_wk_exp_perc','DIY_receipt','DIY_wk','DIY_wk_perc','DIY_exp_receipt','DIY_exp_wk','DIY_wk_exp_perc',
                 'Electronics_receipt','Electronics_wk','Electronics_wk_perc','Electronics_exp_receipt','Electronics_exp_wk',
                 'Electronics_wk_exp_perc','Education_receipt','Education_wk','Education_wk_perc','Education_exp_receipt',
                 'Education_exp_wk','Education_wk_exp_perc','Tech_and_services_receipt','Tech_and_services_wk',
                 'Tech_and_services_wk_perc','Tech_and_services_exp_receipt','Tech_and_services_exp_wk',
                 'Tech_and_services_wk_exp_perc','Cosmetics_and_selfcare_receipt','Cosmetics_and_selfcare_wk',
                 'Cosmetics_and_selfcare_wk_perc','Cosmetics_and_selfcare_exp_receipt','Cosmetics_and_selfcare_exp_wk',
                 'Cosmetics_and_selfcare_wk_exp_perc','Clothes_and_shoes_receipt','Clothes_and_shoes_wk',
                 'Clothes_and_shoes_wk_perc','Clothes_and_shoes_exp_receipt','Clothes_and_shoes_exp_wk',
                 'Clothes_and_shoes_wk_exp_perc','House_and_kitchen_receipt','House_and_kitchen_wk',
                 'House_and_kitchen_wk_perc','House_and_kitchen_exp_receipt','House_and_kitchen_exp_wk',
                 'House_and_kitchen_wk_exp_perc']

    for x in fill_na_list:
        y = raw0.groupby(['Receipt_id'])[x].transform('sum')
        raw0[x]= raw0.groupby(['Receipt_id'])[x].apply(lambda x: x.fillna(y))

    # Encoding Part_of_day
    raw0['Part_of_day_num'] = raw0['Part_of_day']


    encode_Part_of_day = {'Part_of_day_num':{'Early':0,
                                             'Morning':1,
                                             'Afternoon':2,
                                             'Evening':3,
                                             'Late_night':4}}

    # Applying the encoder
    raw0.replace(encode_Part_of_day, inplace=True)

    # Encode Payment_Type
    raw0['Payment_Type_num'] = raw0['Payment_Type']

    encode_Payment_Type = {'Payment_Type_num': {'Card':0,'Cash':1,'Plan':2}}

    # applying the encoder
    raw0.replace(encode_Payment_Type, inplace=True)

    # Encoding Card_Source
    raw0['Card_Source_num'] = raw0['Card_Source']

    encode_Card_Source = {'Card_Source_num': {'Contactless':0,
                                              'Pin':1,
                                              '0':2,
                                              'DD':3,
                                              'DB':4,
                                              'Plan':5,'Transfer':6}}

    # Applying the encoder
    raw0.replace(encode_Card_Source, inplace=True)
    
    raw0=raw0.query("Receipt_id == {}".format(Receipt_id_g))

    raw1 = raw0[['Receipt_id','Venue_id','Total_Nbr_of_Items','Total_Price','Date_diff','Week_day_numeric','Nbr_trips_per_wk','Nbr_items_per_wk',
         'Nbr_items_wk_perc','Expenditure_per_wk','Total_Exp_wk_perc','hour','Part_of_day_num',
         'Payment_Type_num','Card_Source_num','Bread_receipt','Bread_wk','Bread_wk_perc',
         'Bread_exp_receipt','Bread_exp_wk','Bread_wk_exp_perc','Cooked_meats_receipt','Cooked_meats_wk','Cooked_meats_wk_perc',
         'Cooked_meats_exp_receipt','Cooked_meats_exp_wk','Raw_meats_receipt','Raw_meats_wk','Raw_meats_wk_perc','Raw_meats_exp_receipt',
         'Raw_meats_exp_wk','Raw_meats_wk_exp_perc','Eating_out_receipt','Eating_out_wk','Eating_out_wk_perc','Eating_out_exp_receipt',
         'Eating_out_exp_wk','Eating_out_wk_exp_perc','Snacks_receipt','Snacks_wk','Snacks_wk_perc','Snacks_exp_receipt','Snacks_exp_wk',
         'Snacks_wk_exp_perc','Drinks_receipt','Drinks_wk','Drinks_wk_perc','Drinks_exp_receipt','Drinks_exp_wk','Drinks_wk_exp_perc',
         'Vegetables_receipt','Vegetables_wk','Vegetables_wk_perc','Vegetables_exp_receipt','Vegetables_exp_wk',
         'Vegetables_wk_exp_perc','Fruit_receipt','Fruit_wk','Fruit_wk_perc','Fruit_exp_receipt','Fruit_exp_wk','Fruit_wk_exp_perc',
         'Cooking_base_receipt','Cooking_base_wk','Cooking_base_wk_perc','Cooking_base_exp_receipt','Cooking_base_exp_wk',
         'Cooking_base_wk_exp_perc','Dairy_produce_receipt','Dairy_produce_wk','Dairy_produce_wk_perc',
         'Dairy_produce_exp_receipt','Dairy_produce_exp_wk','Dairy_produce_wk_exp_perc','Seasoning_receipt',
         'Seasoning_wk','Seasoning_wk_perc','Seasoning_exp_receipt','Seasoning_exp_wk','Seasoning_wk_exp_perc',
         'Breakfast_receipt','Breakfast_wk','Breakfast_wk_perc','Breakfast_exp_receipt','Breakfast_exp_wk',
         'Breakfast_wk_exp_perc','Transport_wk','Transport_wk_perc','Transport_exp_receipt','Transport_exp_wk',
         'Transport_wk_exp_perc','DIY_receipt','DIY_wk','DIY_wk_perc','DIY_exp_receipt','DIY_exp_wk','DIY_wk_exp_perc',
         'Electronics_receipt','Electronics_wk','Electronics_wk_perc','Electronics_exp_receipt','Electronics_exp_wk',
         'Electronics_wk_exp_perc','Education_receipt','Education_wk','Education_wk_perc','Education_exp_receipt',
         'Education_exp_wk','Education_wk_exp_perc','Tech_and_services_receipt','Tech_and_services_wk',
         'Tech_and_services_wk_perc','Tech_and_services_exp_receipt','Tech_and_services_exp_wk',
         'Tech_and_services_wk_exp_perc','Cosmetics_and_selfcare_receipt','Cosmetics_and_selfcare_wk',
         'Cosmetics_and_selfcare_wk_perc','Cosmetics_and_selfcare_exp_receipt','Cosmetics_and_selfcare_exp_wk',
         'Cosmetics_and_selfcare_wk_exp_perc','Clothes_and_shoes_receipt','Clothes_and_shoes_wk',
         'Clothes_and_shoes_wk_perc','Clothes_and_shoes_exp_receipt','Clothes_and_shoes_exp_wk',
         'Clothes_and_shoes_wk_exp_perc','House_and_kitchen_receipt','House_and_kitchen_wk',
         'House_and_kitchen_wk_perc','House_and_kitchen_exp_receipt','House_and_kitchen_exp_wk',
         'House_and_kitchen_wk_exp_perc']].drop_duplicates(subset=['Receipt_id'])

    # Droping Receipt_id
    raw1.drop(['Receipt_id'], axis=1, inplace=True)

    # capping outliers for Total_Nbr_of_Items at the 99th quantile
    #raw1['Total_Nbr_of_Items'].clip(upper=raw1['Total_Nbr_of_Items'].quantile(.99), inplace=True)

    # capping outliers for Total_Price at the 95th quantile (a value of 35.2)
    # this is more reasonable than the 99th quantile
    raw1['Total_Price'].clip(upper=raw1['Total_Price'].quantile(.95), inplace=True)

    # capping outliers for Nbr_trips_per_wk at the 95th quantile 
    raw1['Nbr_trips_per_wk'].clip(upper=raw1['Nbr_trips_per_wk'].quantile(.95), inplace=True)

    # capping outliers for Nbr_items_per_wk at the 95th quantile 
    raw1['Nbr_items_per_wk'].clip(upper=raw1['Nbr_items_per_wk'].quantile(.95), inplace=True)

    # capping outliers for Expenditure_per_wk at the 95th quantile 
    raw1['Expenditure_per_wk'].clip(upper=raw1['Expenditure_per_wk'].quantile(.95), inplace=True)

    # capping outliers for Electronics_exp_receipt at the 99th quantile 
    raw1['Electronics_exp_receipt'].clip(upper=raw1['Electronics_exp_receipt'].quantile(.99), inplace=True)

    # capping outliers for Electronics_exp_wk at the 99th quantile 
    raw1['Electronics_exp_wk'].clip(upper=raw1['Electronics_exp_wk'].quantile(.99), inplace=True)

    # capping outliers for Education_exp_receipt at the 99th quantile 
    raw1['Education_exp_receipt'].clip(upper=raw1['Education_exp_receipt'].quantile(.99), inplace=True)

    # capping outliers for Education_exp_wk at the 99th quantile 
    raw1['Education_exp_wk'].clip(upper=raw1['Education_exp_wk'].quantile(.99), inplace=True)

    # capping outliers for DIY_exp_receipt at the 99th quantile 
    raw1['DIY_exp_receipt'].clip(upper=raw1['DIY_exp_receipt'].quantile(.99), inplace=True)

    # capping outliers for DIY_exp_wk at the 99th quantile 
    raw1['DIY_exp_wk'].clip(upper=raw1['DIY_exp_wk'].quantile(.99), inplace=True)

    # capping outliers for Cosmetics_and_selfcare_exp_receipt at the 99th quantile 
    raw1['Cosmetics_and_selfcare_exp_receipt'].clip(upper=raw1['Cosmetics_and_selfcare_exp_receipt'].quantile(.99), inplace=True)

    # capping outliers for Cosmetics_and_selfcare_exp_wk at the 99th quantile 
    raw1['Cosmetics_and_selfcare_exp_wk'].clip(upper=raw1['Cosmetics_and_selfcare_exp_wk'].quantile(.99), inplace=True)
    
    return raw1
    

#*************************************************************

# For standard scaler fit transform 
# use the saved version of standard scaler
# load it with joblib
def predict(raw1):
    import joblib
    import pandas as pd
    raw1.drop(['Trips_response_lt_5'], axis=1,inplace=True)
    scaler = joblib.load('../Models/StandardScaler_models/StandardScaler_06102022')
    X_transform = scaler.transform(raw1)
    X_transform = pd.DataFrame(X_transform, columns=raw1.columns)

        # predict using the classifier  model
    rfc_loaded = joblib.load('../Models/Classifier_models/RandomForestClassifier_Model_27092022')
    Rfc_Guest_pred = rfc_loaded.predict(X_transform)

    # dropping Total_Price as it is the target feature for the regressor model
    X_transform.drop(['Total_Price'],axis=1, inplace=True)

    # predict using the regressor model
    RFR_loaded = joblib.load('../Models/Regressor_models/RandomForestRegressor_Model_06102022')
    RFR_Guest_exp_pred = RFR_loaded.predict(X_transform)

    # inverse transforming Total_Price estimate
    # the manual inverse_transform is done because
    # StandardScaler was outputing errors that have yet to be resolved
    # in the interest of time, I opted for this:
    mean = scaler.mean_[2]
    std = np.sqrt(scaler.var_[2])
    estimate_total_price = (RFR_Guest_exp_pred * std) + mean
    
    output_df = pd.DataFrame({'Classifier':Rfc_Guest_pred,
                          'Regressor':RFR_Guest_exp_pred,
                          'Regressor_transform':estimate_total_price})
    return output_df
