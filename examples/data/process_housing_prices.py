import numpy as np
import pgeocode
import sys


f = open('pp-2018.csv', 'r')
lines = f.readlines()
f.close()

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

lines = lines[start_idx:end_idx]

#[latitude, longitude, price]
data = np.zeros((len(lines), 3))
nomi = pgeocode.Nominatim('gb_full')
for i in range(data.shape[0]):
  print('processing entry ' + str(i+1)+'/'+str(data.shape[0]))
  sys.stderr.flush()
  sys.stdout.flush()
  tokens = [s.strip(' "') for s in lines[i].split(',')]
  price = int(tokens[1])
  postcode = tokens[3]
  pgeo = nomi.query_postal_code(postcode)
  data[i, :] = np.array([pgeo.latitude, pgeo.longitude, price])

np.save('prices2018-'+str(start_idx)+'-'+str(end_idx)+'.npy', data)

print('done')
