# Print software information
# print('Source : https://github.com/seungwonpark/SunSpotTracker')
# Based on python 2.7.12 by Seungwon Park
# change to python 3.6 by guitar79@gs.hs.kr

# get data from https://sdo.gsfc.nasa.gov/assets/img/browse/
# file name structure : 20170228_231038_1024_MHII.jpg
# conda install beautifulsoup

site = 'https://sdo.gsfc.nasa.gov/assets/img/browse/'
target = '1024_HMII.jpg'

from datetime import datetime, timedelta
import os
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

# downloading perid and time gap
startdate = '20180101' #start date
enddate = '20180105' #end date
time_gap = 3 #time gap
request_hour = range(0,24,time_gap)
#request_hour = [0, 3, 6, 9, 12, 15, 18, 21] #request hour

#make directory for saving files
save_folder = startdate + '-' + enddate + '-' + target[:-4] + '/'
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

#for calculating date
start_date = datetime.date(datetime.strptime(startdate, '%Y%m%d'))
end_date = datetime.date(datetime.strptime(enddate, '%Y%m%d'))
duration = (end_date - start_date).days

def filename_to_hour(filename):
    fileinfo = filename.split('_')
    return datetime.strptime(fileinfo[0]+fileinfo[1], '%Y%m%d%H%M%S')

#for downloading 1 file per request_hour
download_file_time = datetime.today()

for i in range(duration):
    download_date = start_date + timedelta(i)
    directory = download_date.strftime('%Y') + '/' + download_date.strftime('%m') + '/' + download_date.strftime('%d') + '/'
    url = site + directory
    print ('*'*80)
    print ('trying %s ' % url)
    soup = BeautifulSoup(urlopen(url), "html.parser")
    pre_list = soup.find_all('pre')
    file_list = pre_list[0].find_all('a')
    for i in range(5, len(file_list)):
        filename = file_list[i].text
        #print ('debug', filename)
        if (filename[(-len(target)):] == target) \
            and int(filename_to_hour(filename).strftime('%H')) in(request_hour) \
            and download_file_time.strftime('%Y%m%d%H') != filename_to_hour(filename).strftime('%Y%m%d%H') : 
            print ('Trying %s' % filename)
            if os.path.exists('%s/%s' % (save_folder, filename)):
                print ('*'*40)
                print (filename + 'is exist')
            else :
                urllib.request.urlretrieve(url+filename, '%s/%s' % (save_folder, filename))
                print ('*'*60)
                print ('Downloading' + filename)
                download_file_time = filename_to_hour(filename)
        else:
            print ('Skipping ' + filename)
