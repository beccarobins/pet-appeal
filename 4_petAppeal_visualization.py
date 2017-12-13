import pandas as pd
import petAppeal
import re
import string
import seaborn as sns
import numpy as np

local_file_path = ''
petfinder_file = local_file_path + 'petfinder_data_clean'

shelter_animals = pd.read_csv(petfinder_file)

drop_cols = ['Unnamed: 0', 'address1', 'address2', 'city', 'description', 'email', 'lastUpdate',
             'name', 'pet_id', 'phone', 'photos', 'shelter_id', 'state', 'zip']

shelter_animals = shelter_animals.drop(labels=drop_cols, axis=1).reset_index(drop=True)

##Set a logical (i.e., not alphabetical) order to view variables on figures
status = ['Available', 'Adopted']
age = ['Baby', 'Young', 'Adult', 'Senior']
sex = ['M', 'F', 'U']
size = ['S', 'M', 'L', 'XL']
animal = ['Cat', 'Dog', 'Rabbit', 'Bird', 'Scales, Fins & Other',
          'Small & Furry', 'Horse', 'Barnyard']
bi_var = ['yes', 'no']

#Determine unqiue breeds by removing coat color
breed = [re.sub("\((.*?)\)",'', i) for i in shelter_animals['breed']]
breed = sorted(set(filter(None, breed)))

print 'There are ', len(breed), 'unique breeds (cats & dogs combined)'
shelter_animals = shelter_animals.drop(labels=['breed'], axis=1).reset_index(drop=True)

##View categorical variables for the overall dataset
reorder_dict = {'status': status, 'age': age, 'sex': sex, 'size': size,
                'animal': animal, 'multi_adoption': bi_var, 'altered': bi_var, 
                'hasShots': bi_var, 'housetrained': bi_var,
                'noCats': bi_var, 'noClaws': bi_var, 'noDogs': bi_var, 
                'noKids': bi_var, 'specialNeeds': bi_var, 'mix': bi_var}

for key, value in reorder_dict.iteritems():
    data = shelter_animals.groupby(key).size()
    data = data.reindex(index= value)
    title = key.translate(None, string.punctuation).upper() + ' - ALL ANIMALS' 
    petAppeal.piePlot(data, data.index.values, title)
    
##View categorical variables by status label
adopted = shelter_animals[(shelter_animals.status == 'Adopted')]
available = shelter_animals[(shelter_animals.status == 'Available')]

reorder_dict.pop('status', None)
    
for key, value in reorder_dict.iteritems():
    data1 = adopted.groupby(key).size()
    data2 = available.groupby(key).size()
    df = pd.DataFrame({'Adopted': data1, 'Available': data2})
    df.columns = status
    df = df.reset_index()
    petAppeal.group_bar_graph(df, ['Adopted', 'Available'], key)

##View numerical variables by status label
reorder_dict = {'description_length': shelter_animals['description_length'],
                'description_polarity': shelter_animals['description_polarity'], 
                'description_subjectivity': shelter_animals['description_subjectivity']}

adopted = adopted.select_dtypes(include=['int', 'float']).copy()
available = available.select_dtypes(include=['int', 'float']).copy()

for key, value in reorder_dict.iteritems():
    data1 = adopted[key]
    data2 = available[key]
    petAppeal.plot_hist(data1, data2, key)

##View correlation matrix for numerical variables
corr = shelter_animals.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)