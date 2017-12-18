import pandas as pd
import petAppeal

##Request api key and api secret from https://www.petfinder.com/developers/api-docs
petFinder_api_key = ''

##This uses the animal shelters queried from the No Kill list to find animals
##Animal shelter ids are required to query for pets
url = 'https://raw.githubusercontent.com/beccarobins/PetAppeal/master/Petfinder%20No%20Kill%20Shelters.csv'
shelters = pd.read_csv(url)
shelter_animals = pd.DataFrame()

status_ids = ['X', 'A', 'H', 'P']
##Runs through the list of shelter IDs and queries the Petfinder shelter.getPets method
##Returns a dataframe of all animals in the specified shelters with details
for i in range(0, len(shelters)):
    shelter_id = shelters.shelter_id[i]
    
    for status in status_ids:    
        shelter_animals = shelter_animals.append(petAppeal.getPets(shelter_id, petFinder_api_key, status))

local_file_path = ''
petfinder_file = local_file_path + 'petfinder_shelter_animals.csv'

shelter_animals.to_csv(petfinder_file)