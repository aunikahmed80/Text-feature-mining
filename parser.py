ProductCount={}
#KeyWordlist=['AT&T','Verizon','T-Mobile','Sprint','Nextel','Cingular','Virgin Mobile']
KeyWordlist=['Apple iPod nano 2 GB Black (1st Generation) OLD MODEL']


with open('Electronics.txt','r') as inputfile:
    for line in inputfile:
        if line.startswith('product/productId:'):
            productID=line.split(':')[1].strip()
            line=inputfile.readline()
            if all(x in line for x in KeyWordlist):
                productName=line.split(':')[1].strip()
                for i in range(0,8):
                    line=inputfile.readline()
                if productName in  ProductCount:
                    ProductCount[productName]+=len(line.split(':')[1].split('.'))
                else:
                    ProductCount[productName]=len(line.split(':')[1].split('.'))

SelectedProducts=[]
TotalLines=0
for product in sorted(ProductCount, key=ProductCount.get, reverse=True):
    if ProductCount[product]< 50:
        break
    SelectedProducts.append(product)
    TotalLines+=ProductCount[product]
    print(product,ProductCount[product])
print(TotalLines)



with open('iPod.Large','w') as outputfile:
    with open('Electronics.txt','r') as inputfile:
        for line in inputfile:
            if line.startswith('product/productId:'):
                productID=line.split(':')[1].strip()
                line=inputfile.readline()
                productName=line.split(':')[1].strip()
                if productName in SelectedProducts:
                    for i in range(0,7):
                        line=inputfile.readline()
                    outputfile.write('[t]'+line.split(':')[1])
                    line=inputfile.readline()
                    for sentence in line.split(':')[1].split('.'):
                        if len(sentence.strip())>0:
                            outputfile.write('##'+sentence.strip()+'\n')

