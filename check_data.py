train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()
counter=0

def get_key(s):
    pos=0
    for i in range(len(s)):
        if s[i]=="@":
            pos=i
    key=""
    for i in range(pos):
        key+=str(s[i])
    return key

def get_val(s):
    pos=0
    for i in range(len(s)):
        if s[i]=="@":
            pos=i
    val=""
    for i in range(pos+1,len(s)):
        val+=str(s[i])
    return val



with open('./train_num.txt') as f:#@が2個以上あると要チェック
    for line in f:
        line.rstrip('\n')
        z=0
        for i in range(len(line)):
            if str(line[i])=="@":
                z+=1
        if z >= 2:
            print(str(line))

try:
    with open('./datakun.txt') as f:
        a=0
        for line in f:
            line.rstrip('\n')
            key=get_key(str(line))
            val=get_val(str(line))
            datakun[str(key)]=int(val)
            a+=1
            if a < 10:
                print("datakun")
                print("key="+str(key)+",value="+str(val))
    with open('./train.txt') as f:
        a=0
        for line in f:
            line.rstrip('\n')
            key=get_key(str(line))
            val=get_val(str(line))
            train[str(key)]=int(val)
            a+=1
            if a < 10:
                print("train")
                print("key="+str(key)+",value="+str(val))

    with open('./train_num.txt') as f:
        a=0
        for line in f:
            line.rstrip('\n')
            key=get_key(str(line))
            val=get_val(str(line))
            train_num[str(key)]=str(val)
            a+=1
            if a < 10:
                print("train_num")
                print("key="+str(key)+",value="+str(val))

    with open('./NoAns.txt') as f:
        a=0
        for line in f:
            line.rstrip('\n')
            key=get_key(str(line))
            val=get_val(str(line))
            NoAns[str(key)]=int(val)
            a+=1
            if a < 10:
                print("NoAns")
                print("key="+str(key)+",value="+str(val))

    with open('./counter.txt') as f:
        for line in f:
            line.rstrip('\n')
            counter=int(line)
    
except Exception:
    pass


print(counter)
print(type(counter))
