<h1 style="color: green;">Home Shopping project</h1>

</p>
I have tendency of collecting my shopping receipts. In May 2020, I decided to build a database of
these shopping receipts. This database is called Home Shopping. It has provided me with a useful way
not only of keeping an eye on my expenditure but also gaining insight into my consumption habbits:
<ul>
<li>where do I spend most of my money</li>
<li>How much have I spent in each venue overall</li>
<li>when do I spend most of my money</li>
<li>what do I spend most of my money on</li>
<li>what products do I buy most</li>
<li>How much do I spend per week, per month, per year</li>
<li>How many items do I buy per week, per month, per year</li>
<li>etc ...</li>
</ul>
</p>

<p>
Having gathered several years worth of data, I wanted to apply machine learning to this data.
The project involves several tables from the database
<ul>
<li>Receipt table - This contains summary receipt data, total price, total number of items receipt date, receipt time and shopping venue</li>
<li>Receipt table - This contains summary receipt data, total price, total number of items receipt date, receipt time and shopping venue</li>
<li>Payment table - This contains payment information e.g. payment type; cash, card, plan Card_Source; Contactless, Pin, 0,DB. DD, Transfer</li>
<li>Item table - This contains items/product information e.g. item name and item price</li>
</ul>
</p>

<h3 style="color: green;">Problem statement</h3>
<p>
I need a set of tools that guide my expenditure such that I feel more in control of
my expenditure while saving time spent shopping.

I have a small fridge so I don't tend to bulk buy and consume over a longer period.
I buy small quantities, as a result I do many shopping trips in a week, which consumes
a very important resource, time.

In additions, I go through phases where I buy lots of things in a short period of time
whether online or in store. This reflects negatively on my budget.

With respect to the actually expenditure, I don't have a consistent expenditure pattern
i.e. there are significant variations/variance between expenditure on similar shopping trips

I want to smooth the shopping experience; I want to reduce the time spent shopping, the number of
trips I do per week; I want to reduce the variability in expenditure using a planning tool that
provides a good estimate of expenditure given shopping list.

These tools will be used in combination to optimise the shopping experience: reduce time spent shopping
and stabilize expenditure.

</p>
<h3 style="color: green;">What's inside</h3>
<h5 style="color: green;">Data</h5>
<p>
Inside the data folder you will find all of the datasets used in the project.<br>
Individual notebooks will read in or create these datasets
</p>

<h5 style="color: green;">Feature engineering</h5>
<p>
Inside Feature engineering folder is the notebook where all the feature engineering are performed.<br>
The output of this is model_data.csv
</p>

<h5 style="color: green;">Modelling</h5>
<p>
In here is where all the modelling notebooks are stored.<br>
Note, the "Classifier models Home Shopping.ipynb" is the same as "Classifier models Home Shopping feature selection.ipynb"<br>
the difference being the absence of feature selection on the former.
</p>

<h5 style="color: green;">Models</h5>
<p>
In here all the regressor and classifier models are stored also all the StandardScaler X and y scalers are stored.
</p>

<h5 style="color: green;">Implementation</h5>
<p>
In here is the test example of the application of the models<br>
The long term aim is to demonstrate deployment in AWS. But for now, this is how it's implemented
</p>

<h5 style="color: green;">Monitoring</h5>
<p>
This is where mock monitoring scenarios are developed to mimick real situations<br>
where after deployment various monitoring concerns are considered.<br>
Note, the majority of the monitoring reports are for a previous run of the project without XGBoost algorithm.<br>
Instead the best models were random forest classifier and regressor.
</p>

<h5 style="color: green;">Reports</h5>
<p>
In hear, the pandas reports, exploratory data analysis is done on the model_data.csv and Evidently AI monitoring reports are stored.<br>
Note, for a comprehensive monitoring reports, look inside the Random_forest_reports folder.<br><br>
Looking forward, the objective is to resolve the issues with feature selection and a full deployment with AWS
</p>
