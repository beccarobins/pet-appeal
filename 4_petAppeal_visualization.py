#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

path = '/home/becca/Insight Project/data files'
os.chdir(path)

file = '/home/becca/Insight Project/data files/munged_shelter_animals.csv'
shelter_animals = pd.read_csv(file)

shelter_animals = shelter_animals.drop_duplicates(subset=['pet_id'])
shelter_animals = shelter_animals.drop(labels='Unnamed: 0', axis=1)
shelter_animals = shelter_animals.reset_index(drop=True)

##DROP FOR NOW
print shelter_animals.columns
shelter_animals = shelter_animals.drop(labels=['address1', 'address2', 'breed', 'city', 'description', 'description_length','description_polarity','description_subjectivity','email', 'lastUpdate',
                                               'name', 'pet_id', 'phone', 'photos', 'shelter_id', 'state', 'zip'], axis=1)

def my_autopct(pct):
    return ('%.1f%%' % pct) if pct > 10 else ''
    
def piePlot(data, labels, title):
    colors = ['#820fdf', '#0bc7ff', '#f8685f', '#f1b82d', '#df0fd9', '#0fdf35', '#f17e24', '#244ff1']
    #font = {'family' : 'normal','weight' : 'bold','size'   : 22}
    # Plot
    fig = plt.figure()
    fig = plt.gcf() # get current figure
    fig.set_size_inches(8, 5)
    mpl.rcParams['font.size'] = 15.0
    plt.pie(data, colors=colors, autopct=my_autopct,startangle=140, pctdistance=0.5, labeldistance=1.0)
    #plt.legend(labels, loc="lower right")
    
    plt.legend(loc='upper left',labels=labels, frameon=False, bbox_to_anchor=(0.85,1.025))
    plt.axis('equal')
    plt.title(title.upper())
    plt.tight_layout()
    plt.show()
    fig.savefig(title+'.png', transparent=True, dpi=100)
#    mng = plt.get_current_fig_manager()
#    mng.frame.Maximize(True)
#    mng = plt.get_current_fig_manager()
#    mng.full_screen_toggle()

    
#    figManager = plt.get_current_fig_manager()
#    figManager.window.showMaximized()
#    mng = plt.get_current_fig_manager()
#    mng.resize(*mng.window.maxsize())
#    plt.savefig(title+'.png', dpi=None, facecolor='w', edgecolor='w',
#                orientation='portrait', papertype=None, format=None,
#                transparent=True, bbox_inches=None, pad_inches=0.1,
#                frameon=None)
    plt.close()

###################################################
##INITIAL VISUALIZATION
##create index lists for labels that need reordering
status = ['Available', 'Adopted']
age = ['Baby', 'Young', 'Adult', 'Senior']
sex = ['M', 'F', 'U']
size = ['S', 'M', 'L', 'XL']
animal = ['Cat', 'Dog', 'Rabbit', 'Bird', 'Scales, Fins & Other', 'Small & Furry', 'Horse', 'Barnyard']

reorder = ['status', 'age', 'sex', 'size', 'animal']
reorder_list = [status]+[age]+[sex]+[size]+[animal]

feature_list = list(shelter_animals.columns.values)
print feature_list

for i in range(len(feature_list)):
    feature = feature_list[i]
    data = shelter_animals.groupby(feature).count().iloc[:,0]
    title = feature.upper() + ' - ALL ANIMALS'
    for j in range(len(reorder)):
        if feature == reorder[j]:
            data = data.reindex(index = reorder_list[j])
    labels = data.index.values
    print 'Labels for', feature, ':', labels
    piePlot(data, labels, title)

#######################################
#INCLUDE ONLY DOGS AND CATS IN FINAL ANALYSIS
cats_dogs = shelter_animals[(shelter_animals.animal == 'Cat') | (shelter_animals.animal == 'Dog')]

####################################################
##CHECK OUT BALANCE OF CLASSES

imbalance_check = cats_dogs.groupby('status').count().iloc[:,0]
labels = imbalance_check.index.values
piePlot(imbalance_check, labels, 'Status - Imbalanced')

#################################
data = pd.DataFrame({'data': cats_dogs.groupby(['status']).count().iloc[:,0].reindex(index = ['Available', 'Adopted'])})

data = data.as_matrix()

np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots()
fig = plt.figure(1)
# We define a fake subplot that is in fact only the plot.  

# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
status = ('Available', 'Adopted')
y_pos = np.arange(len(status))
performance = 3 + 10 * np.random.rand(len(status))
colors = ['#f8685f', '#f1b82d']
ax.barh(y_pos, data, color=colors, ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(status, fontsize=(18))
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count', fontsize=(18))
ax.set_xlim([0, 20000])
#ax.set_title('Animals by Status', fontsize=(22), fontweight='bold')
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.tick_params(axis=u'both', which=u'both',length=0)
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected (use 'major' or 'minor' for one or the other)
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off')
plt.show()
fig.savefig('Animals by Status.png', transparent=True)
#plt.savefig('Animals by Status.png', dpi=None, facecolor='w', edgecolor='w',
#            orientation='portrait', papertype=None, format=None,
#            transparent=True, bbox_inches=None, pad_inches=0.5,
#            frameon=None)
plt.close()

############################################################
available_cats_dogs = cats_dogs[(cats_dogs.status == 'Available')]
adopted_cats_dogs = cats_dogs[(cats_dogs.status == 'Adopted')]

if len(available_cats_dogs)<len(adopted_cats_dogs):
    adopted_cats_dogs = adopted_cats_dogs.sample(len(available_cats_dogs))
else:
   available_cats_dogs = available_cats_dogs.sample(adopted_cats_dogs)

cats_dogs = pd.concat([available_cats_dogs, adopted_cats_dogs], axis=0)

balance_check = cats_dogs.groupby('status').count().iloc[:,0]
####################################################
###Cheack out features of cats and dogs only
animal = ['Cat','Dog']
reorder_list = [status]+[age]+[sex]+[size]+[animal]

for i in range(len(feature_list)):
    feature = feature_list[i]
    data = cats_dogs.groupby(feature).count().iloc[:,0]
    title = feature.upper() + ' - Cats & Dogs'
    for j in range(len(reorder)):
        if feature == reorder[j]:
            data = data.reindex(index = reorder_list[j])
    labels = data.index.values
    print 'Labels for', feature, ':', labels
    piePlot(data, labels, title) 


data = pd.DataFrame({'data': cats_dogs.groupby(['status']).count().iloc[:,0].reindex(index = ['Available', 'Adopted'])})

data = data.as_matrix()

np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots()
fig = plt.figure(1)
# We define a fake subplot that is in fact only the plot.  

# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
status = ('Available', 'Adopted')
y_pos = np.arange(len(status))
performance = 3 + 10 * np.random.rand(len(status))
colors = ['#f8685f', '#f1b82d']
ax.barh(y_pos, data, color=colors, ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(status, fontsize=(18))
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count', fontsize=(18))
ax.set_xlim([0, 20000])
#ax.set_title('Animals by Status', fontsize=(22), fontweight='bold')
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.tick_params(axis=u'both', which=u'both',length=0)
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected (use 'major' or 'minor' for one or the other)
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off')
plt.show()
fig.savefig('Animals by Status - Balanced.png', transparent=True)
#plt.savefig('Animals by Status.png', dpi=None, facecolor='w', edgecolor='w',
#            orientation='portrait', papertype=None, format=None,
#            transparent=True, bbox_inches=None, pad_inches=0.5,
#            frameon=None)
plt.close()
##############################
###############################################################################
###DRILL INTO THE DATA BY GROUPING

#exclude status from graphs as this should be included in all
#feature_list = feature_list[0:12]
#
#for i in range(len(feature_list)):
#    feature1 = feature_list[i]    
#    for j in range(len(feature_list)):
#     if i != j:
#        feature2 = feature_list[j]
#        print feature1, 'and', feature2
#        
#        data1 = available_cats_dogs.groupby([feature1, feature2]).count().iloc[:,0]
#        data2 = adopted_cats_dogs.groupby([feature1, feature2]).count().iloc[:,0]
#        
#
#available_grouped = available_cats_dogs.groupby(['size']).count().iloc[:,0].reindex(index = ['S', 'M', 'L', 'XL']).tolist()
#adopted_grouped = adopted_cats_dogs.groupby(['size']).count().iloc[:,0].reindex(index = ['S', 'M', 'L', 'XL']).tolist()
#
#raw_data = {'Size': size, 'Available': available_grouped, 'Adopted': adopted_grouped}
#df = pd.DataFrame(raw_data, columns = ['Size', 'Available', 'Adopted'])
#
#pos = list(range(3)) 
#width = 0.25
#pos = list(range(len(df['Available']))) 
#width = 0.25 
#fig, ax = plt.subplots(figsize=(10,5))
#plt.bar(pos, df['Available'], width, alpha=0.5, color='#EE3224', label=df['Size'][0]) 
#plt.bar([p + width for p in pos], df['Adopted'],width, alpha=0.5,  color='#F78F1E', label=df['Size'][1]) 
#ax.set_ylabel('Score')
#ax.set_title('Test Subject Scores')
#ax.set_xticks([p + 0.5 * width for p in pos])
#ax.set_xticklabels(df['Size'])
#plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, max(df['Available'] + df['Adopted'])])
#plt.legend(['Available', 'Adopted'], loc='upper left')
#plt.show()

#test2 = test.reindex(index = ['S', 'M', 'L', 'XL'])

##SOME FORM OF GRAPH THAT ACTUALLY MAKES SENSE
        
#        
#        
#        
#data = data1.unstack()
#data=data.sort_values('yes',ascending=False)
#fig, ax = plt.subplots()
#plt.title('Words')
#plt.ylabel('Count')
#labels = ['Male', 'Female', 'Unknown']
#x = np.arange(3)
#data1.plt.bar(x, data, color='#0bc7ff')
#plt.xticks(x, labels);
#plt.show()
#ind = np.arange(3) 
#plt.xticks(ind, labels, rotation='vertical')
#ax.set_xticklabels(labels)
#plt.set_xticklabels(['Male', 'Female', 'Unknown'])
#       
#
#
#
#
#
#import seaborn as sns
#
#
#cats_dogs.set_index('size').ix[order].groupby('group').plot(kind='bar')
#
#data = cats_dogs.groupby(['status',feature1]).count().iloc[:,0]
#data1 = available_animals.groupby('size').count().iloc[:,0]
#data1 = data1.reindex(index = ['S', 'M', 'L', 'XL'])
#data2 = homed_animals.groupby('size').count().iloc[:,0]
#data2 = data2.reindex(index = ['S', 'M', 'L', 'XL'])
#
#data_all = pd.DataFrame({'available': data1, 'adopted': data2})
#data_test = data_all.tranpose()
#
#
#data_test = data_test.dropna(axis=1)
#state = pd.Series(['Adopted', 'Available'])
#data_test['e'] = state
#
#test = data_test.append(state, ignore_index=False)
#sns.set(style="darkgrid")
#ax = sns.countplot(x='size', data=cats_dogs)    
#ax = sns.countplot(x="status", hue="size", data=cats_dogs)
#       
#       
########################################333 
#data = cats_dogs.groupby(['status',feature1]).count().iloc[:,0]
#data1 = available_animals.groupby('size').count().iloc[:,0]
#data2 = homed_animals.groupby('size').count().iloc[:,0]
#
#size = ['S', 'M', 'L', 'XL']
#data1 = data1.reindex(index = ['S', 'M', 'L', 'XL'])
#data2 = data2.reindex(index = ['S', 'M', 'L', 'XL'])
#data_test = data_all.transpose()

########################################################################
##NEED TO SAVE DATAFRAME WITH ALL FEATURES
##DO NOT REMOVE
file = '/home/becca/Insight Project/data files/munged_shelter_animals.csv'
shelter_animals = pd.read_csv(file)

shelter_animals = shelter_animals.drop_duplicates(subset=['pet_id'])
shelter_animals = shelter_animals.drop(labels='Unnamed: 0', axis=1)
shelter_animals = shelter_animals.reset_index(drop=True)

cats_dogs = shelter_animals[(shelter_animals.animal == 'Cat') | (shelter_animals.animal == 'Dog')]

available_cats_dogs = cats_dogs[(cats_dogs.status == 'Available')]
adopted_cats_dogs = cats_dogs[(cats_dogs.status == 'Adopted')]

if len(available_cats_dogs)<len(adopted_cats_dogs):
    adopted_cats_dogs = adopted_cats_dogs.sample(len(available_cats_dogs))
else:
   available_cats_dogs = available_cats_dogs.sample(adopted_cats_dogs)

cats_dogs = pd.concat([available_cats_dogs, adopted_cats_dogs], axis=0)
cats_dogs.to_csv('/home/becca/Insight Project/data files/cat_dogs.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')