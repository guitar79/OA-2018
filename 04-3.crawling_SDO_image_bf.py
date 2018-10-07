# Based on Python 3.5.2 | Anaconda 4.1.1
# Print software information
#print('Source : https://github.com/seungwonpark/SunSpotTracker')
# Based on python 2.7.12 by Seungwon Park
# change to python 3.6 by guitar79@gs.hs.kr
# get data from https://sdo.gsfc.nasa.gov/assets/img/browse/
# file name structure : 20170228_231038_1024_MHII.jpg

# conda install beautifulsoup

site = 'https://sdo.gsfc.nasa.gov/assets/img/browse/'
target = '1024_HMII.jpg'

from datetime import datetime, timedelta
import time
import os
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

startdate = '20180101' #start date
enddate = '20180105' #end date
req_hour = [0, 3, 6, 9, 12, 15, 18, 21]

drout = target[:-4] + '/'
if not os.path.exists(drout):
	os.makedirs(drout)

start_date = datetime.date(datetime.strptime(startdate, '%Y%m%d'))
end_date = datetime.date(datetime.strptime(enddate, '%Y%m%d'))
duration = (end_date - start_date).days

def filename_to_time(filename):
    fileinfo = filename.split('_')
    return time.strptime(fileinfo[0]+fileinfo[1], '%Y%m%d%H%M%S')
#print(filename_to_time('20170228_231038_1024_MHII.jpg'))

for i in range(duration):
    dn_date = start_date + timedelta(i)
    directory = dn_date.strftime('%Y') + '/' + dn_date.strftime('%m') + '/' + dn_date.strftime('%d') + '/'
    url = site + directory
    print ('*'*80)
    print ('trying %s ' % url)
    soup = BeautifulSoup(urlopen(url), "html.parser")
    pre_list = soup.find_all('pre')
    file_list = pre_list[0].find_all('a')
    for i in range(5, len(file_list)):
        filename = file_list[i].text
        #print ('debug', filename)
        if (filename[(-len(target)):] == target) :
            if filename_to_time(filename).tm_hour in(req_hour) : # hour gap between downloading images
                print ('Trying %s' % filename)
                if os.path.exists('%s/%s' % (drout, filename)):
                    print ('*'*40)
                    print (filename + 'is exist')
                else :
                    urllib.request.urlretrieve(url+filename, '%s/%s' % (drout, filename))
                    print ('*'*60)
                    print ('Downloading' + filename)
        else:
            #print ('Skipping ' + filename)
