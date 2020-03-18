from xlrd import open_workbook
wb = open_workbook('20200305_V4_UPDATED.xlsx')
sheet = wb.sheets()[0]

def getval(row,col):
    value = sheet.cell_value(row,col)
    return value

def getdata(rangestart,rangeend,datatype = False):
    datalist = []
    for i in range(rangestart,rangeend):
        isempty = getval(i,1)
        if isempty != "":        
            class data:
                measurement = getval(i,0)
                time = getval(i,1)
                height = getval(i,3)*convert.ft_to_m
                IAS = getval(i,4)*convert.kts_to_ms
                AoA = getval(i,5)
                if datatype:
                    de = getval(i,6)
                    detr = getval(i,7)
                    Fe = getval(i,8)
                    FFl = getval(i,9)*convert.lbs_to_kg
                    FFr = getval(i,10)*convert.lbs_to_kg
                    Fused = getval(i,11)*convert.lbs_to_kg
                    TAT = getval(i,12)
                else:    
                    FFl = getval(i,6)*convert.lbs_to_kg
                    FFr = getval(i,7)*convert.lbs_to_kg
                    Fused = getval(i,8)*convert.lbs_to_kg
                    TAT = getval(i,9)
#               ET  = float(time.split(":")[0])*60 + float(time.split(":")[1])
            datalist.append(data)
    return datalist

#Conversions: 
class convert:
    ft_to_m = 0.3048
    lbs_to_kg = 0.45359237
    kts_to_ms = 1852/3600
    
#Passsenger list:
passlist = []
for i in range(7,16):
    class passenger:
        location = getval(i,0)
        weight = getval(i,7)
    passlist.append(passenger)

fuelblock = getval(17,3)*convert.lbs_to_kg
CLCD1 = getdata(27,33)
#CLCD2 = getdata(43,50)
EleTrimCurve = getdata(58,64,True)
CGpos1 = getval(70,2)
CGpos2 = getval(70,7)
CGshift= getdata(74,76,True)
