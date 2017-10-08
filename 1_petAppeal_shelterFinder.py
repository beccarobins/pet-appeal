#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import json
import sys
import pandas as pd
import urllib
import numpy as np

#Pet Finder API key requested: 9.15.2017
petFinder_api_key = '1baf2f363bd60991ac552f9168cbaec0'
petFinder_api_secret = '7465719af394876d7a509000e69c673a'

##Pet Finder API key requested: 9.16.2017
petFinder_api_key = '49eacc3915af403849e24c8ec3e62b20'
petFinder_api_secret = '4eac7b0518bfd9552f79e85328035b09'

##see https://www.petfinder.com/developers/api-docs for shelter.find method arguments

def shelterFinder(zipcode):
    url = 'http://api.petfinder.com/shelter.find?key='+petFinder_api_key+'&location='+zipcode+'&format=json'
    ##sample url: url = 'https://www.googleapis.com/customsearch/v1?q=+cat+filetype:jpg+solid+brown&cx='+googleCustomSearch_CSE_ID+'&searchType=image&key='+googleCustomSearch_api_key+'&num='+num_images+'&fields=items%2Flink'
    print url

    try:
        json_obj = urllib.urlopen(url)
        data = json.load(json_obj)

        individual_shelters = data['petfinder']['shelters']['shelter']
        for i in range(len(individual_shelters)):
            shelter_info = individual_shelters[i]
            try:
                address1.append(shelter_info['address1']['$t'].encode("utf-8"))
            except:
                address1.append(np.nan)
                print 'Address1 blank'
            try:
                address2.append(shelter_info['address2']['$t'].encode("utf-8"))
            except:
                address2.append(np.nan)
                print 'Address2 blank'
            try:
                city.append(shelter_info['city']['$t'].encode("utf-8"))
            except:
                city.append(np.nan)
                print 'City not provided'
            try:
                country.append(shelter_info['country']['$t'].encode("utf-8"))
            except:
                country.append(np.nan)
                print 'City not provided' 
            try:
                email.append(shelter_info['email']['$t'].encode("utf-8"))
            except:
                email.append(np.nan)
                print 'E-mail not provided'
            try:
                fax.append(shelter_info['fax']['$t'].encode("utf-8"))
            except:
                fax.append(np.nan)
                print 'Who uses fax machines?'
            try:
                shelter_id.append(shelter_info['id']['$t'].encode("utf-8"))
            except:
                shelter_id.append(np.nan)
                print 'No shelter id'
            try:
                latitude.append(shelter_info['latitude']['$t'].encode("utf-8"))
            except:
                latitude.append(np.nan)
                print 'No shelter id'
            try:
                longitude.append(shelter_info['longitude']['$t'].encode("utf-8"))
            except:
                longitude.append(np.nan)
                print 'No shelter id'
            try:
                name.append(shelter_info['name']['$t'].encode("utf-8"))
            except:
                name.append(np.nan)
                print 'No name provided'     
            try:
                phone.append(shelter_info['phone']['$t'].encode("utf-8"))
            except:
                phone.append(np.nan)
                print 'Phone number not provided'
            try:
                state.append(shelter_info['state']['$t'].encode("utf-8"))
            except:
                state.append(np.nan)
                print 'State not listed'
            try:
                zip_code.append(shelter_info['zip']['$t'].encode("utf-8"))
            except:
                zip_code.append(np.nan)
                print 'Zip code not listed'
    except:
        print "Oops!",sys.exc_info(),"occured"
    
    return address1, address2, city, country,  email, fax, shelter_id, phone, latitude, longitude, name, phone, state, zip_code


file = '/home/becca/Insight Project/data files/no_kill_shelters.csv'
no_kill_shelters = pd.read_csv(file)
no_kill_shelters = no_kill_shelters.drop(labels=['Unnamed: 0'], axis=1)

zipcodes_to_check = no_kill_shelters.drop_duplicates(subset=['zip_code'])

address1 = []
address2 = []
city = []
country = []
email = []
fax = []
shelter_id = []
phone = []
latitude = []
longitude = []
name = []
phone = []
state = []
zip_code = []
   
for i in range(len(zipcodes_to_check)):
    print i, 'of', len(zipcodes_to_check)
    zipcode = str(zipcodes_to_check.zip_code[i])
    shelterFinder(zipcode)


found_shelters = pd.DataFrame({'address1': address1, 'address2': address2,
                               'city': city, 'country': country, 'email': email,
                               'fax': fax, 'shelter_id': shelter_id, 'phone': phone,
                               'latitude': latitude, 'longitude': longitude,
                               'name': name, 'phone': phone, 'state': state, 'zip_code': zipcode})
  
found_shelters = found_shelters.drop_duplicates(subset=['shelter_id'])
found_shelters = found_shelters.reset_index()
found_shelters = found_shelters.drop(['index'], axis = 1)
    
found_shelters.to_csv('/home/becca/Insight Project/data files/no_kill_shelters_detailed.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')