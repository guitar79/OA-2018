# Based on Pyhon 3.6 by guitar79@gs.hs.kr

from datetime import datetime, timedelta
import os

# some variables for downloading (site, file, perid and time gap, etc.)
site = 'https://sdo.gsfc.nasa.gov/assets/img/browse/'
target = '2048_HMII.jpg' #this tpye of image will be downloading

startdate = '20120101' #start date
enddate = '20181010' #end date
time_gap = 6 #time gap
request_hour = range(0,24,time_gap) #make list

#variable for calculating date
start_date = datetime.date(datetime.strptime(startdate, '%Y%m%d')) #convert startdate to date type
end_date = datetime.date(datetime.strptime(enddate, '%Y%m%d')) #convert enddate to date type
duration = (end_date - start_date).days #total days for downloading
print ('*'*80)
print ((duration+1), 'days', int((duration+1)*(24/time_gap)), 'files will be downloaded')

save_folder = '20120101-20181010-1024_HMII/'

def filename_to_hour(filename):
    fileinfo = filename.split('_')
    return datetime.strptime(fileinfo[0]+fileinfo[1], '%Y%m%d%H%M%S')

file_list = sorted(os.listdir(save_folder))
for i in range(len(file_list)-1):
    try : 
        if int(filename_to_hour(file_list[i]).strftime('%H')) in(request_hour) :
            print (file_list[i] + 'is O.K.')
            if filename_to_hour(file_list[i]).strftime('%Y%m%d%H') == filename_to_hour(file_list[i+1]).strftime('%Y%m%d%H') : 
                os.remove("%s/%s" % (save_folder, file_list[i+1]))
                print(file_list[i+1], "is removed!")
        else :
            os.remove("%s/%s" % (save_folder, file_list[i]))
            print(file_list[i], "is removed!")
            
        print ('*'*40)
        print (file_list[i] + 'is O.K.')

    except Exception as err : 
        print ('*'*80)
        print(err, file_list[i])
