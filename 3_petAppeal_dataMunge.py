import pandas as pd
import numpy as np
import petAppeal

local_file_path = ''
petfinder_file = local_file_path + 'petfinder_shelter_animals'

shelter_animals = pd.read_csv(petfinder_file)
shelter_animals = shelter_animals.drop(labels='Unnamed: 0', axis=1)

print 'Analyzing animals from', len(shelter_animals.shelter_id.unique()), 'animal shelters in', len(shelter_animals.state.unique()), 'states'

shelter_animals['breed'] = shelter_animals['breed'].replace('()',np.nan)
shelter_animals.lastUpdate = pd.to_datetime(shelter_animals['lastUpdate'])
shelter_animals['zip'] = shelter_animals['zip'].fillna(0).apply(np.int64)

print 'This dataset begins on', min(shelter_animals.lastUpdate), 'and ends on', max(shelter_animals.lastUpdate)

options_df = petAppeal.sort_options(shelter_animals['options'])
shelter_animals = options_df.merge(shelter_animals, left_index=True, right_index=True)
shelter_animals = shelter_animals.drop(labels=['options'], axis=1)

description_df = petAppeal.description_analysis(shelter_animals['description'])
shelter_animals = description_df.merge(shelter_animals, left_index=True, right_index=True)

multi_adoption_df = petAppeal.multi_adoption(shelter_animals['name'])
shelter_animals = multi_adoption_df.merge(shelter_animals, left_index=True, right_index=True)

image_df = petAppeal.image_analysis(shelter_animals['photos'])
shelter_animals = image_df.merge(shelter_animals, left_index=True, right_index=True)

local_file_path = ''
petfinder_file = local_file_path + 'petfinder_data_clean'

shelter_animals.to_csv(petfinder_file)