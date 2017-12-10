import pandas as pd
import petAppeal

##Request api key and api secret from https://www.petfinder.com/developers/api-docs
petFinder_api_key = ''

##This uses the scraped No Kill Network to query for animal shelters
##Any list of (US) zip codes can be used for querying
url = 'https://raw.githubusercontent.com/beccarobins/PetAppeal/master/No%20Kill%20Network%20Animal%20Shelters.csv'
zip_code_list = pd.read_csv(url)
zip_code_list = zip_code_list.shelter_zip_code.dropna(axis=0).reset_index().drop(labels=['index'], axis=1)

shelters = pd.DataFrame()

for i in range(0, len(zip_code_list)):
    
    zipcode = str(int(zip_code_list.iloc[i]))
    
    while len(zipcode)<5:
        zipcode = '0'+zipcode
    
    shelters = shelters.append(petAppeal.shelterFinder(zipcode, petFinder_api_key))

shelters = shelters.drop_duplicates(subset=['shelter_id'])
shelters = shelters.reset_index().drop(['index'], axis = 1)

##Add the file path where the shelter list should be saved
local_file_path = ''
petfinder_file = local_file_path + 'petfinder_shelter_list'

shelters.to_csv(petfinder_file)