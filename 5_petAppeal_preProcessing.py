#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd

file = '/home/becca/Insight Project/data files/cat_dogs.csv'
cats_dogs = pd.read_csv(file)
cats_dogs = cats_dogs.drop(labels=['Unnamed: 0','address1', 'address2', 'email', 'pet_id', 'phone'], axis=1)

##DROP FOR NOW
photo_list = cats_dogs['photos']
cats_dogs = cats_dogs.drop(labels=['breed','lastUpdate', 'name', 'photos','description'], axis=1)
cats_dogs = cats_dogs.drop(labels=['zip', 'city', 'state', 'shelter_id'], axis=1)
###BINARY VARIABLES

cats_dogs.mix = cats_dogs['mix'].astype("category").cat.codes
cats_dogs.sex = cats_dogs['sex'].astype("category").cat.codes
cats_dogs.animal = cats_dogs['animal'].astype("category").cat.codes
#cats_dogs.city = cats_dogs['city'].astype("category").cat.codes
#cats_dogs.shelter_id = cats_dogs['shelter_id'].astype("category").cat.codes
#cats_dogs.state = cats_dogs['state'].astype("category").cat.codes

##MULTI CLASS VARIABLES
age_ordered = ['Baby', 'Young', 'Adult', 'Senior']
cats_dogs.age = cats_dogs['age'].astype("category", ordered=True, categories=age_ordered).cat.codes
#
size_ordered = ['S', 'M', 'L', 'XL']
cats_dogs.size = cats_dogs['size'].astype("category", ordered=True, categories=size_ordered).cat.codes

cats_dogs.altered = cats_dogs['altered'].astype("category").cat.codes
cats_dogs.hasShots = cats_dogs['hasShots'].astype("category").cat.codes
cats_dogs.housetrained = cats_dogs['housetrained'].astype("category").cat.codes
cats_dogs.noCats= cats_dogs['noCats'].astype("category").cat.codes
cats_dogs.noClaws = cats_dogs['noClaws'].astype("category").cat.codes
cats_dogs.noDogs = cats_dogs['noDogs'].astype("category").cat.codes
cats_dogs.noKids = cats_dogs['noKids'].astype("category").cat.codes
cats_dogs.specialNeeds = cats_dogs['specialNeeds'].astype("category").cat.codes


cats_dogs.to_csv('/home/becca/Insight Project/data files/cats_dogs_processed.csv',
                 sep=',', na_rep='', float_format=None, columns=None, header=True,
                 index=True, index_label=None, mode='w', encoding=None,
                 compression=None, quoting=None, quotechar='"', line_terminator='\n',
                 chunksize=None, tupleize_cols=False, date_format=None,
                 doublequote=True, escapechar=None, decimal='.')