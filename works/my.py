import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# df =  pd.read_csv("datas/dm_office_sales.csv")
# sns.scatterplot(x='salary' , y='sales' , data=df , hue='division') 
# plt.show()


dx = pd.read_csv("datas/all_sites_scores.csv")

# my_columns = ['RottenTomatoes','Metacritic','IMDB']

# describe_stats = dx[my_columns].describe()
avengers = dx[dx['FILM']  == "Avengers: Age of Ultron (2015)"] 
print(avengers)


avengers_rows = dx[dx['FILM'].apply(lambda x: x== "Avengers: Age of Ultron (2015)")]
print(avengers_rows.columns)
# print(describe_stats)

# print("- - -")
# print("---- - -")
# print(dx.head())
# print(dx.info())
# print(dx.describe())

# sns.scatterplot(x='RottenTomatoes' ,y='RottenTomatoes_User' , data=dx)
 
# plt.show()