import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 


data = pd.read_csv('datas/all_sites_scores.csv')
fandango = pd.read_csv('datas/fandango_scrape.csv')

#show 
corr_fandango = fandango[["RATING" , "VOTES" , "STARS"]].corr()
print(corr_fandango)


# show in scatterPlot
plt.figure(figsize=(10,5) , dpi=150)
sns.scatterplot(data=fandango , x='RATING' , y='VOTES'  )
plt.show()



# task one seperate year
fandango["YEAR"] = fandango['FILM'].apply(lambda title: title.split("(")[-1].replace(")" ,""))
print(fandango.head())

print(fandango["YEAR"].value_counts())





