from django.shortcuts import render
import pandas as pd
from sklearn import preprocessing
import datetime
import time
import io
import numpy as np
import math
from django.http import HttpResponse
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def data():
	organs = ['РУВД Алатауского района', 'РУВД Алмалинского района', 'РОП Алмалинского района', 'РУВД Ауэзовского района', "РОП Ауэзовского района",
         "РОП Бостандыкского района", "РУВД Бостандыкского района", "РОП Медеуского района", "РУВД Медеуского района", "УВД Наурызбайского района ДВД города Алматы"
         ,"ДВД города Алматы", "РУВД Жетысуского района", "РУВД Турксибского района"]

	data = pd.read_csv('crimes.csv', header = None,skiprows=1)

	crimes = data[data[2].isin(organs)]
	le = preprocessing.LabelEncoder()
	crimes[2] = le.fit_transform(crimes[2])
	crimes[3] = le.fit_transform(crimes[3])
	crimes[4], crimes[5] = crimes[4].str.split(' ', 1).str
	crimes[5], crimes[6] = crimes[5].str.split(':', 1).str
	return crimes;

 #Create your views here.
def Save(request):
	crimes=data()

	test = crimes[crimes[4].isin(['12/11/18'])]
	train = pd.concat([crimes, test, test]).drop_duplicates(keep=False)
	test[4] = [time.mktime(datetime.datetime.strptime(i, "%m/%d/%y").timetuple()) for i in test[4]]
	train[4] = [time.mktime(datetime.datetime.strptime(i, "%m/%d/%y").timetuple()) for i in train[4]]
	train_X = train.loc[: ,1:]
	train_Y = train.loc[:, 0]
	test_X = test.loc[: ,1:]
	test_Y = test.loc[:, 0]

	tree = DecisionTreeClassifier(criterion = 'entropy')
	tree.fit(train_X, train_Y)
	  

	treePred = tree.predict(test_X)
	
	acc = f1_score(test_Y, treePred, average='micro') 
	print("Accuracy is: " + str(round(acc,4) * 100) + "%")


	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(train_X, train_Y)
	neighPred = neigh.predict(test_X)
	acc1 = f1_score(test_Y, neighPred, average='micro') 
	print("Accuracy is: " + str(round(acc1,4) * 100) + "%")

	gnb = GaussianNB()
	gnb.fit(train_X, train_Y)
	gnbPred = gnb.predict(test_X)
	acc2 = f1_score(test_Y, gnbPred, average='micro') 
	print("Accuracy is: " + str(round(acc2,4) * 100) + "%")


	lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
	lr.fit(train_X, train_Y)
	lrPred = lr.predict(test_X)
	acc3 = f1_score(test_Y, gnbPred, average='micro') 
	print("Accuracy is: " + str(round(acc3,4) * 100) + "%")
	stealing = crimes[crimes[0].isin(['1880','1990','1910','1870'])]
	stealing_locs = []
	for i, x in stealing[7].iteritems():
		locations = getLocations(stealing[7][i], stealing[8][i])
		stealing_locs.append(locations)

	crimes=pd.concat([crimes, stealing, stealing]).drop_duplicates(keep=False)
	vandalism = crimes[crimes[0].isin(['2930','3070','1920','2940','1490'])]
	vandalism_locs = []
	for i ,x in vandalism[7].iteritems():
		locations =  getLocations(vandalism[7][i], vandalism[8][i])
		vandalism_locs.append(locations)

	crimes=pd.concat([crimes, vandalism, vandalism]).drop_duplicates(keep=False)
	pain = crimes[crimes[0].isin(['2960','2970','3000','3020'])]
	pain_locs = []
	for i ,x in pain[7].iteritems():
		locations = getLocations(pain[7][i], pain[8][i])
		pain_locs.append(locations)
	crimes=pd.concat([crimes, pain, pain]).drop_duplicates(keep=False)

	drug = crimes[crimes[0].isin(['2920','3090' ,'2880' ,'2910' ,'3190'])]
	drug_locs = []
	for i ,x in drug[7].iteritems():
		locations = getLocations(drug[7][i], drug[8][i])
		drug_locs.append(locations)
	
	other=pd.concat([crimes, drug, drug]).drop_duplicates(keep=False)
	other_locs = []
	for i ,x in other[7].iteritems():
		locations = getLocations(other[7][i], other[8][i])
		other_locs.append(locations)

	

	return render(request,'index.html',{'crimes':stealing_locs,'vandalism':vandalism_locs,'pain':pain_locs,'drug':drug_locs,'other':other_locs,'acc':acc,'acc1':acc1,'acc2':acc2,'acc3':acc3})


def getLocations(longitude, latitude):
    k0 = 0.9996 #scale on central meridian
    a = 6378388.0
    f = 1/298.2572236
    drad = math.pi/180
    b = a*(1-f) #polar axis.
    e = math.sqrt(1 - (b/a)*(b/a)) #eccentricity
    _ = e/math.sqrt(1 - e*e) #Called e prime in reference
    esq = (1 - (b/a)*(b/a)) #e squared for use in expansions
    e0sq = e*e/(1-e*e) #e0 squared - always even powers
    x = longitude
    y =  latitude
    #alert(y)
    utmz = 42
    zcm = 3 + 6*(utmz-1) - 180 #Central meridian of zone
    e1 = (1 - math.sqrt(1 - e*e))/(1 + math.sqrt(1 - e*e)) #Called e1 in USGS PP 1395 also
    M0 = 0.0 #In case origin other than zero lat - not needed for standard UTM
    M = M0 + y/k0 #Arc length along standard meridian.
    mu = M/(a*(1 - esq*(1/4 + esq*(3/64 + 5*esq/256))))
    phi1 = mu + e1*(3/2 - 27*e1*e1/32)*math.sin(2*mu) + e1*e1*(21/16 - 55*e1*e1/32)*math.sin(4*mu) #Footprint Latitude
    phi1 = phi1 + e1*e1*e1*(math.sin(6*mu)*151/96 + e1*math.sin(8*mu)*1097/512)
    C1 = e0sq*pow(math.cos(phi1),2)
    T1 = pow(math.tan(phi1),2)
    N1 = a/math.sqrt(1-pow(e*math.sin(phi1),2))
    R1 = N1*(1-e*e)/(1-pow(e*math.sin(phi1),2))
    D = (x-500000)/(N1*k0)
    phi = (D*D)*(1/2 - D*D*(5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*e0sq)/24)
    phi = phi + pow(D,6)*(61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*e0sq - 3*C1*C1)/720
    phi = phi1 - (N1*math.tan(phi1)/R1)*phi
    #Output Latitude
    Lat = phi/drad
    #Longitude
    lng = D*(1 + D*D*((-1 - 2*T1 - C1)/6 + D*D*(5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*e0sq + 24*T1*T1)/120))/math.cos(phi1)
    lngd = float(zcm) + float(lng/drad)
    #Output Longitude
    Lon = lngd
    coords = []
    coords.append(Lat)
    coords.append(Lon)
    return coords

def get_image(request):
	crimes=data()
	types = []
	for i in crimes[0]:
	    if i == 1880 or i == 1990 or i == 1910 or i == 1870:
	        types.append('Кража/Мошенничество/Грабеж/Хищение')
	    elif i == 2930 or i == 3070 or i == 1920 or i == 2940 or i == 1490:
	        types.append("Хулиганство/Разбой/Вред чужому имуществу")
	    elif i == 1070 or i == 1050 or i == 1060 or i == 1200 or i == 1210 or i == 990 or i == 1140 or i == 1460 or i == 1190 or i == 1040 or i == 1080 or i == 1120:
	        types.append("Причинение вреда здоровью/Изнасилование/Убийство")
	    elif i == 2960 or i == 2970 or i == 3000 or i == 3020:
	        types.append("Наркоторговля/Контрабанда")
	    elif i == 2920 or i == 3090 or i == 2880 or i == 2910 or i == 3190:
	        types.append("Другое")
	    
	crimeType = pd.DataFrame(types)
	  #An "interface" to matplotlib.axes.Axes.hist() method
	pltt = crimeType[0].value_counts().plot.barh()
	f = io.BytesIO(pltt) 
	plt.savefig(f, format="png", facecolor=(0.95,0.95,0.95))
	return HttpResponse(f.getvalue(), content_type="1/png")
def get_second(request):
	crimes=data()
	organData = le.inverse_transform(crimes[2])
	od = pd.Series(organData)
	pltt=od.value_counts().plot.barh()
	t = io.BytesIO(pltt) 
	plt.savefig(t, format="png", facecolor=(0.95,0.95,0.95))
	return HttpResponse(f.getvalue(), content_type="2/png")

