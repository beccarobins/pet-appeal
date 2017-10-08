#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

file = '/home/becca/Insight Project/data files/shelter_animals.csv'

shelter_animals = pd.read_csv(file)
shelter_animals = shelter_animals.drop_duplicates(subset=['pet_id'])
shelter_animals = shelter_animals.drop(labels='Unnamed: 0', axis=1)
shelter_animals = shelter_animals.reset_index(drop=True)

print 'Analyzing animals from', len(shelter_animals.shelter_id.unique()), 'animal shelters in', len(shelter_animals.state.unique()), 'states'

##WHAT ARE THE DATA TYPES WE'RE WORKING WITH?
#print shelter_animals.dtypes
#print shelter_animals.head()
##Convert lastUpdate to datetime
shelter_animals.lastUpdate = pd.to_datetime(shelter_animals['lastUpdate'])

period_start = min(shelter_animals.lastUpdate)
period_end = max(shelter_animals.lastUpdate)

print 'This dataset begins on', period_start, 'and ends on', period_end


name_original = shelter_animals['name']
##DATA IS CATEGORICAL
##OPTIONS NEEDS TO BE ENGINEERED
###BUT WHAT ARE ALL THE OPTIONS?
import re
options_list = []
list_len = []
for i in range(len(shelter_animals.options)):
    test_str = str(shelter_animals.options[i])
    test_str = re.findall(r'\w+', test_str)
    options_list.append(test_str)
    list_len.append(len(test_str))
    
all_options = []
for i in range(len(options_list)):
    options_list[i]
    if len(options_list[i]) == 1:
        all_options.append(str(options_list[i]))
    if len(options_list[i]) != 1:
        for j in range(len(options_list[i])):
            all_options.append(options_list[i][j])

options = list(set(all_options))
options_amend = []
import string

for i in range(len(options)):
    option = str(options[i])
    option = option.replace('nan','')
    options_amend.append(option.translate(None, string.punctuation))

options = list(set(options_amend))
options = list(filter(None, options))

altered = []
for i in range(len(shelter_animals)):
    test = "altered" in str(shelter_animals.options[i])
    if test==True:
        altered.append('yes')
    else:
        altered.append('no')
        
hasShots = []
for i in range(len(shelter_animals)):
    test = "hasShots" in str(shelter_animals.options[i])
    if test==True:
        hasShots.append('yes')
    else:
        hasShots.append('no')
        
housetrained = []
for i in range(len(shelter_animals)):
    test = "housetrained" in str(shelter_animals.options[i])
    if test==True:
        housetrained.append('yes')
    else:
        housetrained.append('no')
        
noKids = []
for i in range(len(shelter_animals)):
    test = "noKids" in str(shelter_animals.options[i])
    if test==True:
        noKids.append('yes')
    else:
        noKids.append('no')
        
noCats = []
for i in range(len(shelter_animals)):
    test = "noCats" in str(shelter_animals.options[i])
    if test==True:
        noCats.append('yes')
    else:
        noCats.append('no')
        
noDogs = []
for i in range(len(shelter_animals)):
    test = "noDogs" in str(shelter_animals.options[i])
    if test==True:
        noDogs.append('yes')
    else:
        noDogs.append('no')

noClaws = []
for i in range(len(shelter_animals)):
    test = "noClaws" in str(shelter_animals.options[i])
    if test==True:
        noClaws.append('yes')
    else:
        noClaws.append('no')
                
specialNeeds = []
for i in range(len(shelter_animals)):
    test = "specialNeeds" in str(shelter_animals.options[i])
    if test==True:
        specialNeeds.append('yes')
    else:
        specialNeeds.append('no')

    
options_df =  pd.DataFrame({'altered': altered, 'hasShots': hasShots, 'housetrained': housetrained, 'noCats': noCats, 'noClaws': noClaws, 'noDogs': noDogs, 'noKids': noKids, 'specialNeeds': specialNeeds })
shelter_animals = options_df.merge(shelter_animals, left_index=True, right_index=True)
shelter_animals = shelter_animals.drop(labels=['options'], axis=1)

import re
from textblob import TextBlob
num_words = []
description_polarity = []
description_subjectivity = []
description_class = []
description_pos = []
description_neg = []

for i in range(len(shelter_animals.description)):
    line = str(shelter_animals.description[i])
    num_words.append(len(re.findall(r'\w+', line)))
    try:
        opinion = TextBlob(line)
        polarity, subjectivity = opinion.sentiment
        description_polarity.append(polarity)
        description_subjectivity.append(subjectivity)
#        opinion = TextBlob(line, analyzer=NaiveBayesAnalyzer())
#        classification, pos, neg = opinion.sentiment
#        description_class.append(classification)
#        description_pos.append(pos)
#        description_neg.append(neg)
    except:
        description_polarity.append(0.0)
        description_subjectivity.append(0.5)
#        description_class.append('neu')
#        description_pos.append(0.5)
#        description_neg.append(0.5)
    
stopwords = ['adopts', 'neutered',"trn'd",'tnr', 'shots', 'spayed', 'petsmart', 'pend', 'pendg','pendin', 'pending', 'hold', 'shelter', 'foster', 'adoption', 'reduced', 'fee', 'adopted', 'care', 'in', 'kitten', 'cat', 'dog', 'puppy', 'pup', 'litter', '#']

name_num = []
name = []
import string

multi_animal_adoption = []
multi_adoption_parameters = ['and', 'with', 'two', 'three']

for i in range(len(name_original)):
    line = str(name_original[i])
    line = ''.join([j for j in line if not j.isdigit()])
    line = line.replace('&', 'and')
    line = line.translate(None, string.punctuation)
    line = ' '.join( [w for w in line.split() if len(w)>1] )
    line = line.lower()
    for k in range(len(stopwords)):
        line = line.replace(stopwords[k], '')
    name.append(line)
    name_num.append(len(re.findall(r'\w+', line)))
    multi_pet_potential = 0
    for j in range(len(multi_adoption_parameters)):
        test = multi_adoption_parameters[j] in line
        if test==True:
            multi = 1
            multi_pet_potential = np.vstack([multi_pet_potential, multi])
        else:
            multi = 0
            multi_pet_potential = np.vstack([multi_pet_potential, multi])
    multi_pet_potential = multi_pet_potential.sum()
    if multi_pet_potential>0:
        multi_animal_adoption.append('yes')
    else:
        multi_animal_adoption.append('no')

text_features = pd.DataFrame({'multi_adoption': multi_animal_adoption,'description_length': num_words, 'description_polarity': description_polarity, 
                              'description_subjectivity': description_subjectivity})

#text_features = pd.DataFrame({})
shelter_animals = text_features.merge(shelter_animals, left_index=True, right_index=True)

##Rename status labels to something understandable
status_list = []
for i in range(len(shelter_animals.status)):
    status = str(shelter_animals.status[i])
    if status == 'A':
        status_list.append('Available')
    else:
        status_list.append('Adopted')

status_df =  pd.DataFrame({'status': status_list})
shelter_animals = shelter_animals.drop(labels=['status'], axis=1)
shelter_animals = status_df.merge(shelter_animals, left_index=True, right_index=True)

shelter_animals.to_csv('/home/becca/Insight Project/data files/munged_shelter_animals.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')