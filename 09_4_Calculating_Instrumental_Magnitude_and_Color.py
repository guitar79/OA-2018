import numpy as np
from astropy.io.ascii import read
from astropy import table
#%%
data = read('Standardization_data_0.dat')
print('type(data)\n', type(data))
print('data\n', data)
data.pprint

#%%
# b, v, c means instrumental B, V magnitudes and instrumental B-V color.
v  = table.Column(name='v' , data = -2.5 * np.log10(data['count_V']/data['T_V']))
b  = table.Column(name='b' , data = -2.5 * np.log10(data['count_B']/data['T_B']))
dv = table.Column(name='dv', data = 1.0857 / np.sqrt(data['count_V']))
db = table.Column(name='db', data = 1.0857 / np.sqrt(data['count_B']))

c  = table.Column(name='color' , data = b-v)
dc = table.Column(name='dcolor', data = np.sqrt(db**2 + dv**2) )
#print('v\n', v)
#print('b\n', b)
#print('c\n', c)
#%%
# Only save upto 3 or 5 decimal points
v.format  ='%6.3f'
b.format  ='%6.3f'
c.format  ='%6.3f'
dv.format ='%6.5f'
db.format ='%6.5f'
dc.format ='%6.5f'
data.add_columns([v, dv, b, db, c, dc])

#print('v\n', v)
#print('b\n', b)
#print('c\n', c)
#print('data\n', data)
#%%
# To be more visual, I will "sort" with respect to the column 'Target':
data = data.group_by('Target') 
#print('data\n', data)
# Then print:
data.pprint(max_width=250)  # max_width is used to print out all the values