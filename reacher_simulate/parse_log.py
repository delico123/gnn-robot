global DP

def spliter(data,NAME,_f):
    temp2 = data.split(NAME)
    if len(temp2)>1:
        temp2 = temp2[1].split('([')[1]
        temp2 = temp2.split('])')[0]
        temp2 = temp2.split(',')
        if len(temp2)==2:
            res = []
            temp2 = temp2[0]
            temp3 = temp2.split('[')
            if len(temp3)>1:
                temp2 = temp3[1]
            temp3 = temp2.split(']')
            if len(temp3)>1:
                temp2 = temp3[0]
                temp2= float(temp2)
            res.append(temp2)
            while True:
                data = _f.readline()[:-1]
                temp = data.split('])')
                if len(temp)>1:
                    if len(temp[-2].split('DP:tensor'))>1:
                        temp = temp[-2].split('DP:tensor')[1]
                        temp = temp.split('[')[1]
                        temp = temp.split(']')[0]
                        temp = temp.split(',')
                        temp = list(map(float,temp))
                        global DP
                        DP.append(temp)
                    return res
                else:
                    temp3 = temp[0].split('[')
                    if len(temp3)>1:
                        temp = temp3[1]
                    temp3 = temp.split(']')
                    if len(temp3)>1:
                        temp = temp3[0]
                    temp =  float(temp)
                    res.append(temp)
        elif temp2[-1]=='':
            while True:
                data = _f.readline()[:-1]
                temp = data.split('])')
                if len(temp)==1:
                    break
                else:
                    temp = temp[0].split(',')
                    temp = list(map(float,temp))
                    temp2 = temp2[:-1]+temp
                    return temp2
        else:
            temp2 = list(map(float,temp2))
            return temp2

def x_spliter(data,_f):
    ch = data.split('X: tensor')
    res=[]
    flag=0
    if len(ch)>1:
        temp = ch[1]
        temp = temp.split('([[')[1]
        temp = temp.split(']')[0]
        temp = temp.split(',')
        temp = list(map(float,temp))
        res.append(temp)
        while True:
            data = _f.readline()[:-1]
            if data == '':
                break
            temp = data.split('[')[1]
            temp = temp.split(']')
            if len(temp)>2:
                flag=1
            temp = temp[0]
            temp = temp.split(',')
            temp = list(map(float,temp))
            res.append(temp)
            if flag==1:
                return res

def paser(filename=None):
    if filename == None:
        file = open('../test.log', 'r')
    else:
        file= open(filename,'r')
    O = []
    S = []
    T = []
    X = []
    global DP
    DP = []
    flag=0
    while True:
        data = file.readline()[:-1]
        if data == '':
            break
        x = x_spliter(data,file)
        if x != None:
            #print(x)
            X.append(x)
        o = spliter(data,'O: tensor',file)
        if o != None:
            O.append(o)
        t = spliter(data,'T:tensor',file)
        if t != None:
            T.append(t)
        s = spliter(data,'S:tensor',file)
        if s != None:
            S.append(s)
    # print('X: ',X)
    # print('O: ',O)
    # print('T: ',T)
    # print('S: ',S)
    # print('DP: ',DP)
    return X,O, T,S,DP
if __name__ == '__main__':
    paser()