def item_name_features(input_data):
    import pandas as pd
    import numpy as np

    last_7_days = input_data

    # bread indicator
    nots = ['garlic','ham','garlic','ham']
    ins = ['bloomer','bread']

    last_7_days['Bread'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                any(word.lower() in sentence.lower() 
                                    for word in ins) 
                                and not any(word.lower() in sentence.lower() 
                                            for word in nots) 
                                else 0)
    

    # Cooked meats indicator
    not_cook = ['glass','shampoo','water','conditioner','champagne']
    cooked_meats = ['chicken pasty slices twin pack','steak and kidney pasty',
                    'chicken cooked','cooked chicken','roast chicken thighs',
                    'mackerel','salmon','pork pies classic','ham',
                    'sardines','sausages cocktail','spicy chorizo sausages',
                    'sausages rolls','sausage rolls','salami','meatballs']

    last_7_days['Cooked_meats'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
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

    last_7_days['Raw_meats'] = last_7_days.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower() 
                                                                    for word in raw_meats) 
                                and not any(word.lower() in sentence.lower() for word in not_raw) else 0)

    # Creating snack indicator
    not_snack = ['diesel','james']
    snacks = ['snickers','digestive','digestives',
            'chocolate','yogurt','cake','cakes',
            'snack','nuts','donuts','doughnut',
            'mikati','fudge','maltesers','twix','marmalade',
            'jam','custard']

    last_7_days['Snacks'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
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
            'honey','cordial','whiskey','whisky']

    last_7_days['Drinks'] = last_7_days.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower()
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
                'brussel sprouts','mint','asparagus','beans','Soup']

    last_7_days['Vegetables'] = last_7_days.Item_name.apply(lambda sentence: 
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

    last_7_days['Fruit'] = last_7_days.Item_name.apply(lambda sentence: 1 if any(word.lower() in sentence.lower() 
                                                                for word in fruit) 
                                        and not any(word.lower() in sentence.lower() 
                                                    for word in not_fruit)
                                        else 0)

    # Creating an indicator for cooking base
    not_base = ['fried']
    cooking_base = ['pasta','spaghetti','rice','flour','potatoe','potatoes','potato']

    last_7_days['Cooking_base'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                                any(word.lower() in sentence.lower() 
                                                    for word in cooking_base) 
                                                and not any(word.lower() in sentence.lower() 
                                                            for word in not_base) 
                                                else 0)

    # Creating an indicator for Dairy produce
    dairy_produce = ['cheese','brilliantly','butter','butterlicious','spread','margarine']
    last_7_days['Dairy_produce'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                                any(word.lower() in sentence.lower() 
                                                    for word in dairy_produce) 
                                                else 0)

    # Creating an indicator for seasoning
    seasoning = ['black pepper','salt','seasoning','spice','cinnamon','paprika']

    last_7_days['Seasoning'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                            any(word.lower() in sentence.lower() 
                                                for word in seasoning) 
                                            else 0 )

    # creating an indicator for breakfast food
    breakfast = ['granola','muesli','sultanas','porridge']

    last_7_days['Breakfast'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                            any(word.lower() in sentence.lower() 
                                                for word in breakfast) 
                                            else 0 )

    # creating an indicator for education
    not_edu = ['clevo']
    education = ['king richard williams','linkedin','mysql','financial','python',
                'bootcamp','web server','linux','apache','sas','pencils',
                'eraser','whs 15cm ruler','bic pen','a4','binders','feature', 
                'engineering','full stack', 'optimization','machine learning',
                'tensor flow','pytorch','statistics','hadoop','sas', 'regression',
                'bootcamp','javascript','research methodology','quantitative']

    last_7_days['Education'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                            any(word.lower() in sentence.lower() 
                                                for word in education) 
                                            and not 
                                            any(word.lower() in sentence.lower() 
                                                for word in not_edu) 
                                            else 0)

    # Creating an indicator for cosmetics and self care
    not_cosmetic = ['sony']
    cosmetics_and_selfcare = ['shampoo','shower','tooth','colgate','wisdom','nivea',
                            'razor','body','blades','aqueous','shave','african',
                            'perfume','brut','roll on','Roll-on','bettina bath']

    last_7_days['Cosmetics_and_selfcare'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                                        any(word.lower() in sentence.lower() 
                                                            for word in cosmetics_and_selfcare) 
                                                        and not 
                                                        any(word.lower() in sentence.lower() 
                                                            for word in not_cosmetic) 
                                                        else 0)

    # creating an indicator for house and kitchen 
    not_house = ['cake','bonnlo']
    house_and_kitchen = ['fairy liquid','measure jug','fitted bed sheet','glass',
                        'turner (spatula)','rolling pin','fairy liquid','orange citrus',
                        'turkey baster','dish drainer','extension lead','grater','power spray',
                        'liquid','roaster and rack','kitchen roller','salad tongs',
                        'strainer 12cm','arial pods','metal scourer','bathmat','curtain hooks',
                        'ant killer spray','scouring pads','sponge','surf','foil','plaster',
                        'knife sharpener','electric hand mixer','athena cotton wool','mop',
                        'ofargo']

    last_7_days['House_and_kitchen'] = last_7_days.Item_name.apply(lambda sentence: 1 if 
                                                    any(word.lower() in sentence.lower() 
                                                        for word in house_and_kitchen) 
                                                    and not 
                                                    any(word.lower() in sentence.lower() 
                                                        for word in not_house) 
                                                    else 0)


    #*********************************************************************************
    # ********************** Derive the remaining week features **********************
    #*********************************************************************************

    # feature list
    features = ['Total_Nbr_of_Items', 'Venue', 'Date_diff', 'Nbr_items_per_wk',
               'Expenditure_per_wk', 'Total_Exp_wk_perc', 'Drinks', 'Vegetables',
               'Cosmetics_and_selfcare', 'House_and_kitchen', 'Bread_wk',
               'Cooked_meats_wk', 'Raw_meats_wk', 'Snacks_wk', 'Snacks_exp_receipt',
               'Snacks_exp_wk', 'Drinks_wk', 'Drinks_exp_wk', 'Vegetables_exp_wk',
               'Fruit_wk', 'Cooking_base_wk', 'Dairy_produce_wk', 'Seasoning_wk',
               'Breakfast_wk', 'Education_wk', 'Cosmetics_and_selfcare_wk',
               'Cosmetics_and_selfcare_wk_exp_perc', 'House_and_kitchen_wk']
    
    indicator_list = ['Bread', 'Cooked_meats', 'Raw_meats', 'Snacks', 'Drinks', 'Vegetables','Fruit',
                    'Cooking_base', 'Dairy_produce', 'Seasoning', 'Breakfast', 'Education', 
                    'Cosmetics_and_selfcare', 'House_and_kitchen'
                    ]

    # Looping through the indicator list to derive new features
    for x in indicator_list:
        # Calculate item x count by shopping trip(Receipt) and by week
        last_7_days["{}_receipt".format(x)] = last_7_days.groupby(['Receipt_id'])[x].transform('sum')
        last_7_days["{}_wk".format(x)] = last_7_days.groupby(['week_of_year'])[x].transform('sum')

        # Receipt item x as a proportion of week's item x
        last_7_days["{}_wk_perc".format(x)] = last_7_days["{}_receipt".format(x)] / last_7_days["{}_wk".format(x)]

        # Calculating item x expenditure by shopping trip(Receipt) and by week
        last_7_days["{}_exp_receipt".format(x)] = \
        last_7_days.query("{}==1".format(x)).groupby(['Receipt_id',x])['Item_Price'].transform('sum')
        
        last_7_days["{}_exp_wk".format(x)] = \
        last_7_days.query("{}==1".format(x)).groupby(['week_of_year',x])['Item_Price'].transform('sum')
        
        # Receipt item x expenditure as a proportion of week's item x expenditure
        last_7_days["{}_wk_exp_perc".format(x)] = \
        last_7_days["{}_exp_receipt".format(x)] / last_7_days["{}_exp_wk".format(x)]

    # list of features derived from item_name
    list3 = ['Bread', 'Cooked_meats', 'Raw_meats', 'Snacks', 'Drinks', 'Vegetables','Fruit',
            'Cooking_base', 'Dairy_produce', 'Seasoning', 'Breakfast', 'Education', 
            'Cosmetics_and_selfcare', 'House_and_kitchen'
            ]

    # summing the indicator by receipt to get the total count for the receipt
    for x in list3:
        z = last_7_days.groupby(['Receipt_id'])[x].transform('sum')
        last_7_days[x] = z

    fill_na_list = ['Bread_wk','Cooked_meats_wk','Raw_meats_wk','Snacks_wk', 'Snacks_exp_receipt', 'Snacks_exp_wk',
            'Drinks_wk', 'Drinks_exp_wk', 'Vegetables_exp_wk', 'Fruit_wk', 'Cooking_base_wk', 'Dairy_produce_wk',
            'Seasoning_wk','Breakfast_wk', 'Education_wk', 'Cosmetics_and_selfcare_wk', 
                'Cosmetics_and_selfcare_wk_exp_perc', 'House_and_kitchen_wk'
            ]

    # filling in the missing values by receipt with the max value

    for x in fill_na_list:
        y = last_7_days.groupby(['Receipt_id'])[x].transform('max')
        last_7_days[x]= last_7_days.groupby(['Receipt_id'])[x].apply(lambda x: x.fillna(y))

    # filling the remaining missing values
    last_7_days.fillna(0, inplace=True)

    # de-duplicate the data using Receipt_id
    last_7_days = last_7_days.drop_duplicates(subset=['Receipt_id'])

    # return only the max receipt_id data
    last_7_days = last_7_days.query("Receipt_id == Receipt_id.max()")
    last_7_days

    # Restricting the model features 
    last_7_days = last_7_days[features]
    last_7_days

    return last_7_days