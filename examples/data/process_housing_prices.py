import numpy as np
import pgeocode



f = open('pp-2018.csv', 'r')
lines = f.readlines()
f.close()

#[latitude, longitude, price]
data = np.zeros((len(lines), 3))
nomi = pgeocode.Nominatim('gb')
for i in range(data.shape[0]):
  print('processing entry ' + str(i+1)+'/'+str(data.shape[0]))
  tokens = [s.strip(' "') for s in lines[i].split(',')]
  price = int(tokens[1])
  postcode = tokens[3]
  pgeo = nomi.query_postal_code(postcode)
  data[i, :] = np.array([pgeo.latitude, pgeo.longitude, price])

np.save('prices2018.npy', data)
