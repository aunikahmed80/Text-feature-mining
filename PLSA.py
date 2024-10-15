import numpy as np, numpy.random
import math

max_iter=1000
K=5
HP={}
HPB={}
WordcountPerLine={}
BackgroundProbability={}
WordcountPerDocument={}
TotalNumberOfWords=0
lineNum=0
lambdaB=0.5
dist_threshold=1e-6



punctuation = ".,:;!?'*/@%()[]{}#$&!0123456789"
def remove_stopwords(input_string,stopword):
     input_string=input_string.lower()
     output_string=''
     for item in input_string.split():
         if item not in stopword and len(item)>1:
             output_string= output_string+ item+' '
     for item in punctuation:
        output_string = output_string.replace(item, '')
     return output_string

stopword={}

with open('lemur-stopwords.txt','r') as myfile:
    for line in myfile:
        stopword[line.strip()]=1
    
with open('forumText.txt','r') as myfile:
    for line in myfile:
        if remove_stopwords(line,stopword).strip() !='':
            line=remove_stopwords(line,stopword)
            if lineNum%1000==0:
                print(lineNum)
            #if lineNum==1000:
                #break
            WordcountPerDocument[lineNum]={}
            WordcountPerLine[lineNum]=len(line.split())
            HP[lineNum]={}
            HPB[lineNum]={}
            for word in line.split():
                TotalNumberOfWords+=1
                if word in BackgroundProbability:
                    BackgroundProbability[word]=BackgroundProbability[word]+1
                else:
                    BackgroundProbability[word]=1
                if word in WordcountPerDocument[lineNum]:
                    WordcountPerDocument[lineNum][word]+=1
                else:
                    WordcountPerDocument[lineNum][word]=1
                if word not in HP[lineNum]:
                    HP[lineNum][word]=[0.0 for x in range(K)]
                    HPB[lineNum][word]=0.0
            lineNum=lineNum+1

for word in BackgroundProbability:
    BackgroundProbability[word]=BackgroundProbability[word]/TotalNumberOfWords


TopicModel=[{} for x in range(K)]
for topic in TopicModel:
    for word in BackgroundProbability:
        topic[word]=1/len(BackgroundProbability)

PI=[np.random.dirichlet(np.ones(K),size=1)[0] for x in range(lineNum)]

'''
print('PI[0]'+str(PI[0]))
print('PI[1]'+str(PI[1]))

print('TopicModel[0]'+str(TopicModel[0]))
print('TopicModel[1]'+str(TopicModel[1]))
'''
print("Starting EM------------------------")

for i in range(max_iter):
     print('iteration: ' +str(i))
     i+=1
     #E-step
     print('E-step')
     for d in range(lineNum):
        for word in HP[d]:
            mysum=0
            for j in range(K):
                mysum+=PI[d][j]*TopicModel[j][word]
            for j in range(K):
                HP[d][word][j]=PI[d][j]*TopicModel[j][word]/mysum

            HPB[d][word]=(lambdaB*BackgroundProbability[word])/(lambdaB*BackgroundProbability[word]+((1-lambdaB)*mysum))
     '''
     print('HP[0]'+str(HP[0]))
     print('HP[1]'+str(HP[1]))
     print('HPB[0]'+str(HPB[0]))
     print('HPB[1]'+str(HPB[1]))
'''
     print('M-step')

     #M-step
     #print('Computing PI')
     for d in range(lineNum):
        denom=0
        for jdash in range(K):
            for word in WordcountPerDocument[d]:
                denom+=WordcountPerDocument[d][word]*(1-HPB[d][word])*HP[d][word][jdash]
                
        for j in range(K):
            nom=0
            for word in WordcountPerDocument[d]:
                nom+=WordcountPerDocument[d][word]*(1-HPB[d][word])*HP[d][word][j]

            PI[d][j]=nom/denom

     #print('Computing Topic Model')
     previousTopicModel=[{} for x in range(K)]
     for j in range(K):
          for word in BackgroundProbability:
               previousTopicModel[j][word]=TopicModel[j][word]
     
     for j in range(K):
        denom=0
        for d in range(lineNum):
            for wordDash in WordcountPerDocument[d]:
                denom=denom+WordcountPerDocument[d][wordDash]*(1-HPB[d][wordDash])*HP[d][wordDash][j]

        for word in BackgroundProbability:
            nom=0
            for d in range(lineNum):
                if word in WordcountPerDocument[d]:
                    nom+=WordcountPerDocument[d][word]*(1-HPB[d][word])*HP[d][word][j]
                 
            TopicModel[j][word]=nom/denom
     dist=0.0
     for j in range(K):
          for word in BackgroundProbability:
               dist=dist+math.pow(TopicModel[j][word]-previousTopicModel[j][word],2)
     print('dist='+str(dist))
     if dist < dist_threshold:
          break

     
     '''
     print('PI[0]'+str(PI[0]))
     print('PI[1]'+str(PI[1]))

     print('TopicModel[0]'+str(TopicModel[0]))
     print('TopicModel[1]'+str(TopicModel[1]))
'''

with open('TopicResults.txt','w') as myfile:
     myfile.write('Background Topic:\n------------------------------\n')
     for word in sorted(BackgroundProbability, key=BackgroundProbability.get, reverse=True)[:20]:
          myfile.write(word+','+str(BackgroundProbability[word])+'\n')
     myfile.write('\n\n\n\n')    
     for j in range(K):
          myfile.write('Topic No: '+ str(j)+'\n------------------------------\n')
          for word in sorted(TopicModel[j], key=TopicModel[j].get, reverse=True)[:20]:
            myfile.write(word+','+str(TopicModel[j][word])+'\n')
          myfile.write('\n\n\n\n')




















    
    
