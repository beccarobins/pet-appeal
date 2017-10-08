#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 19:59:38 2017

@author: becca
"""

import json
import sys
import pandas as pd
import urllib
import numpy as np

##import csv file with shelter list
file = '/home/becca/Insight Project/data files/no_kill_shelters_detailed.csv'
shelters = pd.read_csv(file)

#Pet Finder API key requested: 9.15.2017
petFinder_api_key = '1baf2f363bd60991ac552f9168cbaec0'
petFinder_api_secret = '7465719af394876d7a509000e69c673a'

###Pet Finder API key requested: 9.16.2017
#petFinder_api_key = '49eacc3915af403849e24c8ec3e62b20'
#petFinder_api_secret = '4eac7b0518bfd9552f79e85328035b09'

##see https://www.petfinder.com/developers/api-docs for shelter.getPets method arguments

status_types = ['A','X']

def getPets(shelter_id, adoption_status):
    url = 'http://api.petfinder.com/shelter.getPets?key='+petFinder_api_key+'&id='+shelter_id+'&status='+adoption_status+'&format=json'
    print url

    try:
        json_obj = urllib.urlopen(url)
        data = json.load(json_obj)
        
        individual_pets = data['petfinder']['pets']['pet']
        for i in range(len(individual_pets)):
            pet_info = individual_pets[i]
            try:
                age.append(pet_info['age']['$t'].encode("utf-8"))
            except:
                age.append(np.nan)
                print 'No age provided'
            try:
                animal.append(pet_info['animal']['$t'].encode("utf-8"))
            except:
                animal.append(np.nan)
                print 'Animal type unknown'
            try:
                breed_list = []
                if len(pet_info['breeds']['breed']) == 1:
                    breed.append(pet_info['breeds']['breed']['$t'].encode("utf-8"))
                else:
                    for j in range(len(breed_list)):
                        breed_list.append(pet_info['breeds']['breed'][j]['$t'].encode("utf-8"))
                    breed.append(tuple(breed_list))
            except:
                breed.append(np.nan)
                print 'Breed not provided'
            try:
                description.append(pet_info['description']['$t'].encode("utf-8"))
            except:
                description.append(np.nan)
                print 'No description provided'
            try:
                address1.append(pet_info['contact']['address1']['$t'].encode("utf-8"))
            except:
                address1.append(np.nan)
                print 'Address1 blank'
            try:
                address2.append(pet_info['contact']['address2']['$t'].encode("utf-8"))
            except:
                address2.append(np.nan)
                print 'Address2 blank'
            try:
                city.append(pet_info['contact']['city']['$t'].encode("utf-8"))
            except:
                city.append(np.nan)
                print 'City not provided'
            try:
                email.append(pet_info['contact']['email']['$t'].encode("utf-8"))
            except:
                email.append(np.nan)
                print 'E-mail not provided'
            try:
                phone.append(pet_info['contact']['phone']['$t'].encode("utf-8"))
            except:
                phone.append(np.nan)
                print 'Phone number not provided'
            try:
                state.append(pet_info['contact']['state']['$t'].encode("utf-8"))
            except:
                state.append(np.nan)
                print 'State not listed'
            try:
                zip_code.append(pet_info['contact']['zip']['$t'].encode("utf-8"))
            except:
                zip_code.append(np.nan)
                print 'Zip code not listed'
            try:
                pet_id.append(pet_info['id']['$t'].encode("utf-8"))
            except:
                pet_id.append(np.nan)
                print 'No pet id'
            try:
                lastUpdate.append(pet_info['lastUpdate']['$t'])
            except:
                lastUpdate.append(np.nan)
                print 'Last update not available'
            try:
                photo_list = []
                for j in range(len(pet_info['media']['photos']['photo'])):
                    photo_list.append(pet_info['media']['photos']['photo'][j]['$t'].encode("utf-8"))
                photos.append(tuple(photo_list))
            except:
                photos.append(np.nan)
                print 'No photo list'
            try:
                mix.append(pet_info['mix']['$t'].encode("utf-8"))
            except:
                mix.append(np.nan)
                print 'Mix not provided'
            try:
                name.append(pet_info['name']['$t'].encode("utf-8"))
            except:
                name.append(np.nan)
                print 'No name provided'
            try:                  
                options_list = []
                if len(pet_info['options']['option']) == 1:
                    options_list.append(pet_info['options']['option']['$t'].encode("utf-8"))
                else:
                    for j in range(len(pet_info['options']['option'])):
                        options_list.append(pet_info['options']['option'][j]['$t'].encode("utf-8"))
                options.append(tuple(options_list))
            except:
                options.append(np.nan)
                print 'No options listed'
            try:
                sex.append(pet_info['sex']['$t'].encode("utf-8"))
            except:
                sex.append(np.nan)
                print "Animal's sex not listed"
            try:
                shelterId.append(pet_info['shelterId']['$t'].encode("utf-8"))
            except:
                shelterId.append(np.nan)
                print 'No shelter id'
            try:
                size.append(pet_info['size']['$t'].encode("utf-8"))
            except:
                size.append(np.nan)
                print 'Animal size not listed'
            try:
                status.append(pet_info['status']['$t'].encode("utf-8"))
            except:
                status.append(np.nan)
                print 'Animal status unclear'
    except:
        print "Oops!",sys.exc_info(),"occured"
        print 'Url did not work'
    return options, status, address1, address2, phone, state, email, city, zip_code, age, size, photos, pet_id, breed, name, sex, description, mix, shelter_id, lastUpdate, animal

options = []
status = []
address1 = []
address2 = []
phone = []
state = []
email = []
city = []
zip_code = []
age = []
size = []
photos = []
pet_id = []
breed = []
name = []
sex = []
description = []
mix = []
shelterId = []
lastUpdate = []
animal = []
encoded = []

for i in range(len(shelters)):
    print i, 'of', len(shelters.shelter_id)
    shelter_id = shelters.shelter_id[i]
    for j in range(len(status_types)):
        adoption_status = status_types[j]
        getPets(shelter_id, adoption_status)
        
shelter_animals = pd.DataFrame({'options': options, 'status': status, 'address1': address1, 'address2': address2, 'phone': phone, 'state': state, 'email': email, 'city': city,
                     'zip': zip_code, 'age': age, 'size': size, 'photos': photos, 'pet_id': pet_id, 'breed': breed, 'name': name, 'sex': sex,
                     'description': description, 'mix': mix, 'shelter_id': shelterId, 'lastUpdate': lastUpdate, 'animal': animal})
    
file_name = '/home/becca/Insight Project/data files/shelter_animals.csv'
shelter_animals.to_csv(file_name, sep=',', na_rep='', float_format=None,
                       columns=None, header=True, index=True, index_label=None, 
                       mode='w', encoding=None, compression=None, quoting=None,
                       quotechar='"', line_terminator='\n', chunksize=None, 
                       tupleize_cols=False, date_format=None, doublequote=True, 
                       escapechar=None, decimal='.')