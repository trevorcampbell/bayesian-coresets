import numpy as np
import sys
import pandas as pd




#load geocoding data and preprocess into sorted format

geodata_fields = ['country code', 'postal_code', 'place_name',
               'state_name', 'state_code', 'county_name', 'county_code',
               'community_name', 'community_code',
               'latitude', 'longitude', 'accuracy']

print('loading full GB postcodes database')
geodata = pd.read_csv('GB_FULL.txt', sep='\t', header=0, names=geodata_fields, dtype={'postal_code': str})

post_code_to_int = lambda x : int(''.join([str(ord(a.lower())-97) if ord(a.lower())-97 >= 0 else str(ord(a.lower())-48)  for a in x.replace(' ', '')]))

print('extracting postcode, lat, lon')
geodata = geodata[['postal_code', 'latitude', 'longitude']]
print('converting post codes to integers')
geodata['postal_code'] = geodata['postal_code'].apply(post_code_to_int)
print('sorting by integer tags')
geodata.sort_values(by='postal_code', inplace=True)
print('converting to np array')
geodata = np.array(geodata)

print('loading price paid data')
f = open('pp-2018.csv', 'r')
lines = f.readlines()
f.close()

#[integer post code tag, price]
data = np.zeros((len(lines), 2))
print('extracting post code and price')
for i in range(data.shape[0]):
  print('processing entry ' + str(i+1)+'/'+str(data.shape[0]))
  tokens = [s.strip(' "') for s in lines[i].split(',')]
  price = int(tokens[1])
  try:
    postcode = post_code_to_int(tokens[3])
  except:
    postcode = -1
  data[i, :] = np.array([postcode, price])

print('found ' + str(data.shape[0]) + ' entries')

print('removing bad entries')
data = data[data[:, 0] >= 0, :]

print(str(data.shape[0]) + ' entries remaining')

#sort by integer post tag
print('sorting by integer post tag')
data = data[data[:,0].argsort(), :]

#now iterate through geodata and data, incrementing post tags on each as needed
data_lat_lon = np.zeros((data.shape[0], 3))
geo_idx = 0
print('converting post code to lat lon')
for i in range(data.shape[0]):
  print('processing entry ' + str(i+1)+'/'+str(data.shape[0]))
  while(geodata[geo_idx,0] < data[i,0]):
    geo_idx += 1
  if geodata[geo_idx,0] != data[i, 0]:
    #geodata doesn't have this post code, give up
    data_lat_lon[i, 0] = np.nan
  else:    
    data_lat_lon[i, 0] = geodata[geo_idx, 1]
    data_lat_lon[i, 1] = geodata[geo_idx, 2]
    data_lat_lon[i, 2] = data[i, 1]

print('filtering bad entries')
data_lat_lon = data_lat_lon[np.logical_not(np.isnan(data_lat_lon[:, 0])), :]
print(str(data_lat_lon.shape[0]) + ' entries remaining')

np.save('prices2018.npy', data_lat_lon)

print('done')
