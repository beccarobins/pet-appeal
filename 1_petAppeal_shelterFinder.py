import pandas as pd
import petAppeal

##Request api key and api secret from https://www.petfinder.com/developers/api-docs
petFinder_api_key = ''

##This uses the scraped No Kill Network list to query for animal shelters
##Any list of (US) zip codes can be used for querying
url = 'https://raw.githubusercontent.com/beccarobins/PetAppeal/master/No%20Kill%20Network%20Animal%20Shelters.csv'
zip_code_list = pd.read_csv(url)
zip_code_list = zip_code_list.dropna(axis=0).reset_index().drop(labels=['Unnamed: 0', 'index'], axis=1)

shelters = pd.DataFrame()

##Runs through the list of zip codes and queries the Petfinder shelter.find method
##Returns a dataframe of all shelters with details in the surrounding areas
for i in range(0, len(zip_code_list)):
    
    zipcode = str(int(zip_code_list['shelter_zip_code'].iloc[i]))
    
    while len(zipcode)<5:
        zipcode = '0'+zipcode
    
    shelters = shelters.append(petAppeal.shelterFinder(zipcode, petFinder_api_key))

shelters = shelters.drop_duplicates(subset=['shelter_id'])
shelters = shelters.reset_index().drop(['index'], axis = 1)

##Gotchas
##the shelterFinder method will return all shelters in a specified zip code
##and sometimes those in the surrounding areas
##If seeking only No Kill shelters, the list will need to be filtered again
no_kill_shelters = pd.merge(left=zip_code_list, right=shelters, left_on=['shelter_name', 'state_abbr'], right_on=['name', 'state'])
no_kill_shelters = no_kill_shelters.drop(labels=['shelter_zip_code', 'shelter_city', 'state_abbr','name'], axis=1)

##Add the file path where the shelter list should be saved
local_file_path = ''
petfinder_file = local_file_path + 'Petfinder No Kill Shelters.csv'

no_kill_shelters.to_csv(petfinder_file)