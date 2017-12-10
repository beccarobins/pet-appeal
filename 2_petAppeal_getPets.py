import pandas as pd
import petAppeal

##Request api key and api secret from https://www.petfinder.com/developers/api-docs
petFinder_api_key = ''

##This uses the animal shelters queried from the No Kill list to find animals
##Animal shelter ids are required to query for pets
url = 'https://raw.githubusercontent.com/beccarobins/PetAppeal/master/Petfinder%20No%20Kill%20Shelters'
shelters = pd.read_csv(url)
shelter_animals = pd.DataFrame()

for i in range(0, len(shelters)):
    shelter_id = shelters.shelter_id[i]
    shelter_animals = shelter_animals.append(petAppeal.getPets(shelter_id, petFinder_api_key))
        
shelter_animals = shelter_animals.drop_duplicates()
shelter_animals = shelter_animals.reset_index().drop(['index'], axis = 1)

##Add the file path where the shelter animal list should be saved
local_file_path = ''
petfinder_file = local_file_path + 'petfinder_shelter_animals'

shelter_animals.to_csv(petfinder_file)