def transform_and_predict(input_data):
    import pandas as pd
    import numpy as np
    
    import joblib

    # for feature engineering
    from feature_engine.discretisation import EqualWidthDiscretiser
    from feature_engine.discretisation import ArbitraryDiscretiser
    from feature_engine.encoding import RareLabelEncoder
    from feature_engine.outliers import ArbitraryOutlierCapper

    # for Weight of evidence
    from category_encoders.woe import WOEEncoder

    # Scaling the values
    from sklearn.preprocessing import binarize

    # renaming input data
    last_7_days = input_data

    # ***************************************************************************************************************
    # cap expenditure
    capper = joblib.load("../Data transformers/capper_expenditure12062023")
    last_7_days = capper.transform(last_7_days)

    # ***************************************************************************************************************
    # bin expenditure
    Expenditure_bin = joblib.load("../Data transformers/Equal_width_bin_expenditure12062023")
    last_7_days = Expenditure_bin.transform(last_7_days)

    # ***************************************************************************************************************
    # high cardinality features list for conversion to object
    high_card_num = ['Total_Nbr_of_Items', 'Nbr_items_per_wk', 'Expenditure_per_wk', 'Total_Exp_wk_perc', 'Drinks',
    'Vegetables', 'Cosmetics_and_selfcare', 'Bread_wk', 'Cooked_meats_wk', 'Raw_meats_wk',
    'Snacks_wk', 'Snacks_exp_receipt', 'Snacks_exp_wk', 'Drinks_wk', 'Drinks_exp_wk', 'Vegetables_exp_wk',
    'Fruit_wk', 'Cooking_base_wk', 'Dairy_produce_wk', 'Breakfast_wk', 'Education_wk', 'Cosmetics_and_selfcare_wk',
    'Cosmetics_and_selfcare_wk_exp_perc', 'House_and_kitchen_wk'
    ]

    # ***************************************************************************************************************
    # Bin/transform high cardinality
    high_card_bin = joblib.load("../Data transformers/Hig_cardinality_12062023")
    # last_7_days_high_card = high_card_bin.transform(last_7_days[high_card_num])

    last_7_days = high_card_bin.transform(last_7_days)

    # ***************************************************************************************************************
    # change variable type to object in preparation for rare value fit and transform
    for var in high_card_num:
        #last_7_days_high_card[var] = pd.Series(last_7_days_high_card[var], dtype=object)
        last_7_days[var] = pd.Series(last_7_days[var], dtype=object)

    rare_encoder_high_card = joblib.load("../Data transformers/Rare_enc_High_cardinality_12062023")

    # last_7_days_high_card = rare_encoder_high_card.transform(last_7_days_high_card)
    last_7_days = rare_encoder_high_card.transform(last_7_days)

    # ***************************************************************************************************************
    # Date_diff transforming
    Date_diff_trans = joblib.load("../Data transformers/Date_diff_transformer_12062023")

    last_7_days = Date_diff_trans.transform(last_7_days)

    # ***************************************************************************************************************
    # list of low cardinality features for  conversion to object
    low_card_num = ['House_and_kitchen', 'Seasoning_wk']

    # change variable type to object in preparation for rare label encoding
    for var in low_card_num:
        last_7_days[var] = pd.Series(last_7_days[var], dtype=object)


    low_card_enc =  joblib.load("../Data transformers/low_cardinality_rare_transformer_12062023")

    last_7_days = low_card_enc.transform(last_7_days)

    # ***************************************************************************************************************
    # Transforming categorical features
    Categorical_enc = joblib.load("../Data transformers/Categorical_rare_transformer_12062023")

    last_7_days = Categorical_enc.transform(last_7_days)

    # ***************************************************************************************************************
    # woe transformation
    woe_enc = joblib.load("../Data transformers/WOE_transformer_12062023")
    last_7_days = woe_enc.transform(last_7_days)

    # ***************************************************************************************************************
    # load the classifier model
    xgb_loaded = joblib.load('../Models/xgboost_classifier10062023')

    # Making predictions with stored model
    loaded_prob = xgb_loaded.predict_proba(last_7_days)[:,1]

    # binarize with the threshold
    loaded_pred_class_binarize = binarize([loaded_prob],threshold=0.14)[0]
    loaded_pred_class_binarize

    return loaded_pred_class_binarize