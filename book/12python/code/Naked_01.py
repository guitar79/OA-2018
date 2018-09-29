import numpy
import matplotlib.pyplot

#variables
TimeStep = 20.0         # years
waterDepth = 4000.0      # meters
L = 1350.0               # Watts/m^2
albedo = 0.3             # No dim
epsilon = 1.0            # No dim
sigma = 0.0000000567     # W/m^2 K^4
Time = 0.                 #year
Teaperature = 0.00        #K
Heatcontents = 0.00         #J/m^2
Heatcapacity = 4200000. * waterDepth     #J/K m^2

#arrays
Teaperature_list=[]
Time_list=[]
Heatcontents_list=[]
HeatIN_list=[]
HeatOUT_list=[]

#calculate
for i in range(0,100):
    Time_list.append(Time)
    Teaperature_list.append(Teaperature)
    Heatcontents_list.append(Heatcontents)
    HeatIN = L*(1-albedo)/4
    HeatOUT = epsilon * sigma * Teaperature**4
    HeatIN_list.append(HeatIN)
    HeatOUT_list.append(HeatOUT)
    Heatcontents = Heatcontents + ((HeatIN - HeatOUT) * TimeStep * 265.2425 * 24 * 3600)
    Teaperature = Heatcontents / Heatcapacity
    Time = Time + TimeStep

#draw chart
matplotlib.pyplot.plot(Time_list, Teaperature_list)
matplotlib.pyplot.show()