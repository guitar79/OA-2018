# Print software information
# print('Source : https://github.com/seungwonpark/SunSpotTracker')
# Based on python 2.7.12 by Seungwon Park
# change to python 3.6 by guitar79@gs.hs.kr

# get data from https://sdo.gsfc.nasa.gov/assets/img/browse/
# file name structure : 20170228_231038_1024_MHII.jpg
# conda install beautifulsoup

from datetime import datetime, timedelta
import os
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
# some variables for downloading (site, file, perid and time gap, etc.)
site = 'https://sdo.gsfc.nasa.gov/assets/img/browse/'
target = '1024_HMII.jpg' #this tpye of image will be downloading

startdate = '20150101' #start date
enddate = '20181010' #end date
time_gap = 6 #time gap
request_hour = range(0,24,time_gap) #make list
#request_hour = [0, 3, 6, 9, 12, 15, 18, 21] #make list

#variable for calculating date
start_date = datetime.date(datetime.strptime(startdate, '%Y%m%d')) #convert startdate to date type
end_date = datetime.date(datetime.strptime(enddate, '%Y%m%d')) #convert enddate to date type
duration = (end_date - start_date).days #total days for downloading
print ('*'*80)
print ((duration+1), 'days', int((duration+1)*(24/time_gap)), 'files will be downloaded')

#make directory for saving files
save_folder = startdate + '-' + enddate + '-' + target[:-4] + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print ('*'*80)
    print (save_folder, 'is created')
else :
    print ('*'*80)
    print (save_folder, 'is exist')
    
def filename_to_hour(filename):
    fileinfo = filename.split('_')
    return datetime.strptime(fileinfo[0]+fileinfo[1], '%Y%m%d%H%M%S')

download_file_time = datetime.today() #variable for comparing with downloading filename
for i in range(duration):
    try : 
        download_date = start_date + timedelta(i)
        directory = download_date.strftime('%Y') + '/' + download_date.strftime('%m') + '/' + download_date.strftime('%d') + '/'
        url = site + directory
        print ('*'*80)
        print ('trying %s ' % url)
        # using BeutifulSoup for crowling
        soup = BeautifulSoup(urlopen(url), "html.parser")
        #print('soup : ', soup)
        pre_list = soup.find_all('pre')
        #print('pre_list', pre_list)
        file_list = pre_list[0].find_all('a')
        #print('file_list', file_list)
        # select file fot downloading
        for i in range(5, len(file_list)):
            filename = file_list[i].text
            if (filename[(-len(target)):] == target) \
                and int(filename_to_hour(filename).strftime('%H')) in(request_hour) \
                and download_file_time.strftime('%Y%m%d%H') != filename_to_hour(filename).strftime('%Y%m%d%H') : 
                try : 
                    print ('Trying %s' % filename)
                    if os.path.exists('%s/%s' % (save_folder, filename)):
                        print ('*'*40)
                        print (filename + 'is exist')
                        download_file_time = filename_to_hour(filename)
                    else :
                        urllib.request.urlretrieve(url+filename, '%s/%s' % (save_folder, filename))
                        print ('*'*60)
                        print ('Downloading' + filename)
                        download_file_time = filename_to_hour(filename)
                except : 
                    print ('*'*80)
                    print ('Download error %s ' % url)
            else:
                print ('Skipping ' + filename)
    except Exception as err : 
        print ('*'*80)
        print(err, url)
