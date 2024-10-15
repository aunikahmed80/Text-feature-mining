import Stemming
import nltk
import math
import numpy as np, numpy.random
from nltk.corpus import wordnet as wn
import NDCG
import os
import errno
import copy

Data_Directory='./Annotated Data/iPod/'
DataFile='iPod.final'
Results_Directory='./Results/iPod/EM_Sentence/'
EquivalenceFile='Selected Features.txt'
#EquivalenceFile='All_ExplicitFeatures.txt'
dir = Results_Directory+'ExplicitFeatures.txt'
if not os.path.exists(os.path.dirname(dir)):
    try:
        os.makedirs(os.path.dirname(dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


punctuation = "'.,:;!?1234567890"
StopWords=[]

class aspectSentiment:
  def __init__(self, aspect, sentiment, implicit):
    self.aspect = aspect
    self.sentiment = sentiment
    self.implicit=implicit

def remove_punctuation(input_string):
  input_string1=''
  for word in input_string.split():
    if word not in StopWords:
      input_string1 = input_string1+' '+word
  for item in punctuation:
    input_string1 = input_string1.replace(item, ' ')
  return input_string1

Precision=[]
Recall=[]
F_Measure=[]
GlobalNDCG=[]




for GlobalIteration in range(0,1):
  AllData=[]
  TrainingData=[]
  TestData=[]
  Reviews=[]
  RawReviews=[]
  GroundTruth=[]
  Collection={}
  TopicModel={}
  ExplicitFeatures={}
  IDF={}
  WordcountPerLine={}
  BackgroundProbability={}
  EquivalentAspects={}




  TotalNoOfReviews=0
  TotalNoOfWords=0

  with open(Data_Directory+EquivalenceFile,'r') as inputfile:
    for line in inputfile:
      aspects=line.strip().split(',')
      for aspect in aspects:
        EquivalentAspects[aspect]=aspects[0]

  
  with open(Data_Directory+'lemur.stopwords','r') as inputfile:
    for line in inputfile:
      StopWords.append(line.strip())
  
  with open(Data_Directory+DataFile,'r') as inputfile:
    for line in inputfile:
      if (line.startswith('[t]') or line.startswith('*')):
        TotalNoOfReviews+=1
      AllData.append(line)


  for line in AllData:
    TrainingData.append(Stemming.perform_Stemming(line))


  NoOfReviews=-1
  i=0
  j=0
  for line in TrainingData:
    i+=1 # why consider tag sentence in corpus!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    TotalNoOfWords+=len(line.split())
    if (line.startswith('[t]') or line.startswith('*')):
      NoOfReviews+=1
      Reviews.append(list())
      RawReviews.append(list())
      GroundTruth.append(list())
      GroundTruth[NoOfReviews].append(list()) # No need of this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      j=-1
    else:      
      
      #GroundTruth[NoOfReviews].append(list()) #No need if loop continue !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      review_with_aspect=line.replace('(','').replace(')','').replace('-','').split('##')

      review_Sentence=remove_punctuation(review_with_aspect[1])
      if review_Sentence.strip()=='':
        continue
      GroundTruth[NoOfReviews].append(list())
      j+=1
      RawReviews[NoOfReviews].append(line)
      Reviews[NoOfReviews].append({})

      for word in review_Sentence.split():
        if word in Collection:
          Collection[word]+=1
        else:
          Collection[word]=1
        if word in Reviews[NoOfReviews][j]:
          Reviews[NoOfReviews][j][word]+=1
        else:
          Reviews[NoOfReviews][j][word]=1


      for word in set(review_Sentence.split()):
        if word in IDF:
          IDF[word]+=1
        else:
          IDF[word]=1


      if len(review_with_aspect[0])>0:
        explicitFeatures=review_with_aspect[0].split(',')
        for feature in explicitFeatures:
          feature_sentiment_context=feature.split('[')
          sentiment_context=feature_sentiment_context[1].split('@')
          if '[u]' not in feature:
            if EquivalentAspects[feature_sentiment_context[0].strip()] in ExplicitFeatures:
              ExplicitFeatures[EquivalentAspects[feature_sentiment_context[0].strip()]]+=1
            else:
              ExplicitFeatures[EquivalentAspects[feature_sentiment_context[0].strip()]]=1
            
            if EquivalentAspects[feature_sentiment_context[0].strip()] not in TopicModel:
              TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]]={}

            for word in review_Sentence.split():
              if word in TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]]:
                  TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]][word]+=1
              else:
                TopicModel[EquivalentAspects[feature_sentiment_context[0].strip()]][word]=1
            GroundTruth[NoOfReviews][j].append(aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()],sentiment_context[0],False))
          else:
            GroundTruth[NoOfReviews][j].append(aspectSentiment(EquivalentAspects[feature_sentiment_context[0].strip()],sentiment_context[0],True))


  TotalLines=i

  TotalAspectOccurance=0
  for aspect in ExplicitFeatures:
    TotalAspectOccurance+=ExplicitFeatures[aspect]

  for aspect in ExplicitFeatures:
    ExplicitFeatures[aspect]/=TotalAspectOccurance



  for word in Collection:
      BackgroundProbability[word]=Collection[word]/TotalNoOfWords


  for word in Collection:
    for aspect in TopicModel:
      if word not in TopicModel[aspect]:
        TopicModel[aspect][word]=0


  for aspect in TopicModel:
    for word in TopicModel[aspect]:
      TopicModel[aspect][word]=math.log(1+TopicModel[aspect][word])*math.log(1+(TotalLines/IDF[word]))



  for aspect in TopicModel:
    sumOfTFIDF=sum(TopicModel[aspect].values())
    for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True): # why need to sort !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      TopicModel[aspect][word]=(TopicModel[aspect][word]+1)/(sumOfTFIDF+len(Collection))


  with open(Results_Directory+'TopicModel.txt','w') as outputFile:
    for aspect in TopicModel:
      for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True)[:10]:
        outputFile.write(aspect+', '+word+', '+str(TopicModel[aspect][word])+'\n')
      outputFile.write('\n\n\n')
      for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True)[:3]:
        SynonymSet=[]
        syns = wn.synsets(word)
        for s in syns:
           for l in s.lemma_names():
              SynonymSet.append(l)
        for s in SynonymSet:
          if s not in TopicModel[aspect]:
            TopicModel[aspect][s]=TopicModel[aspect][word]

  '''
  for word in sorted(BackgroundProbability, key=BackgroundProbability.get, reverse=True)[:10]:
      print(word, BackgroundProbability[word])
  '''

  with open(Results_Directory+'ExplicitFeatures.txt','w') as outputfile:
    for aspect in sorted(ExplicitFeatures, key=ExplicitFeatures.get, reverse=True):
        outputfile.write(aspect + ' : ' + str(ExplicitFeatures[aspect])+'\n')


  HP=[]
  HPB=[]
  PI=[]
  print("*******",len(Reviews))
  for reviewNum in range(0,len(Reviews)):
    HP.append(list())
    HPB.append(list())
    PI.append(list())
    for lineNum in range(0,len(Reviews[reviewNum])):
      HP[reviewNum].append({})
      HPB[reviewNum].append({})
      PI[reviewNum].append({})
      for word in Reviews[reviewNum][lineNum]:
        HP[reviewNum][lineNum][word]={}
        for aspect in TopicModel:
          HP[reviewNum][lineNum][word][aspect]=0.0
        HPB[reviewNum][lineNum][word]=0.0
      RandomProbabilities=[np.random.dirichlet(np.ones(len(TopicModel)),size=1)[0]]
      myIndex=0
      for aspect in TopicModel:
        PI[reviewNum][lineNum][aspect]=RandomProbabilities[0][myIndex]
        myIndex+=1
      



  max_iter=50
  lambdaB=0.7
  dist_threshold=1e-6

  print("Starting EM------------------------")

  for i in range(max_iter):
       print('iteration: ' +str(i))
       i+=1
       #E-step
       print('E-step')
       for reviewNum in range(0,len(Reviews)):
         for lineNum in range(0,len(Reviews[reviewNum])):
            for word in Reviews[reviewNum][lineNum]:
              mysum=0
              for aspect in TopicModel:
                  mysum+=PI[reviewNum][lineNum][aspect]*TopicModel[aspect][word]
              if mysum == 0:
                # for aspect in self.TopicModel:
                #  print(self.PI[reviewNum][lineNum][aspect], '\t',self.TopicModel[aspect][word])
                print(word)
              for aspect in TopicModel:
                  HP[reviewNum][lineNum][word][aspect]=PI[reviewNum][lineNum][aspect]*TopicModel[aspect][word]/mysum

              HPB[reviewNum][lineNum][word]=(lambdaB*BackgroundProbability[word])/(lambdaB*BackgroundProbability[word]+((1-lambdaB)*mysum))


       print('M-step')

       #M-step
       #print('Computing PI')

       previousPI =copy.deepcopy(PI)
       # previousPI=[]
       # for reviewNum in range(0,len(Reviews)):
       #   previousPI.append(list())
       #   for lineNum in range(0,len(Reviews[reviewNum])):
       #     previousPI[reviewNum].append({})
       #     for aspect in TopicModel:
       #       previousPI[reviewNum][lineNum][aspect]=PI[reviewNum][lineNum][aspect]

         
       for reviewNum in range(0,len(Reviews)):
         for lineNum in range(0,len(Reviews[reviewNum])):
            denom=0
            for aspect in TopicModel:
                for word in Reviews[reviewNum][lineNum]:
                    denom+=Reviews[reviewNum][lineNum][word]*(1-HPB[reviewNum][lineNum][word])*HP[reviewNum][lineNum][word][aspect]
                    
            for aspect in TopicModel:
                nom=0
                for word in Reviews[reviewNum][lineNum]:
                    nom+=Reviews[reviewNum][lineNum][word]*(1-HPB[reviewNum][lineNum][word])*HP[reviewNum][lineNum][word][aspect]
                try:
                  PI[reviewNum][lineNum][aspect]=nom/denom
                except:
                  print(reviewNum,lineNum,aspect,nom,denom)


       dist=0.0
       for reviewNum in range(0,len(Reviews)):
         for lineNum in range(0,len(Reviews[reviewNum])):
            for aspect in TopicModel:
                 dist=dist+math.pow(PI[reviewNum][lineNum][aspect]-previousPI[reviewNum][lineNum][aspect],2)
       print('dist='+str(dist))
       if dist < dist_threshold:
            break



  Precision.append([])
  Recall.append([])
  F_Measure.append([])
  GlobalNDCG.append([])
  ##Test Data Evaluation
  TPDict={}
  FNDict={}
  FPDict={}


  #for MyK in range(0,len(ExplicitFeatures)+1):
  MyK=0
  for ColorProbability in np.arange(0.35, 0.39, 0.05):
    Theta=0.0
    TP=0
    FP=0
    FN=0
    FrequentThreshold=0.0
    ndcgList=[]

    for reviewNum in range(0,len(Reviews)):
      for lineNum in range(0,len(Reviews[reviewNum])):
        if '[u]' in RawReviews[reviewNum][lineNum]:
          ActualImplicitFeatureSet=set([])
          ActualExplicitFeatureSet=set([])
          for item in GroundTruth[reviewNum][lineNum]:
            if item.implicit==True:
              ActualImplicitFeatureSet.add(item.aspect)
            else:
              ActualExplicitFeatureSet.add(item.aspect)
          #print('\n\n\n\n')
          #print(RawReviews[reviewNum][lineNum])
          #print('\n')
          InferredImplicitFeatureSet=set([])



          for aspect in sorted(PI[reviewNum][lineNum], key=PI[reviewNum][lineNum].get, reverse=True):
            if PI[reviewNum][lineNum][aspect]>=ColorProbability \
                    and aspect not in ActualExplicitFeatureSet \
                    and aspect not in Reviews[reviewNum][lineNum] \
                    and ExplicitFeatures[aspect]>=FrequentThreshold \
                    and aspect not in InferredImplicitFeatureSet:
                #print(word, aspect, HP[reviewNum][lineNum][word][aspect],HPB[reviewNum][lineNum][word])
                InferredImplicitFeatureSet.add(aspect)
                  

          #print('\n\n')
          #for aspect in sorted(PI[reviewNum][lineNum], key=PI[reviewNum][lineNum].get, reverse=True)[:15]:
            #if aspect not in ActualExplicitFeatureSet and aspect not in Reviews[reviewNum][lineNum] and ExplicitFeatures[aspect]>=FrequentThreshold:
              #print(aspect,PI[reviewNum][lineNum][aspect])
          
          for aspect in ExplicitFeatures:
            if ExplicitFeatures[aspect]<FrequentThreshold or aspect in Reviews[reviewNum][lineNum]:
              try:
                ActualImplicitFeatureSet.remove(aspect)
              except:
                emni=1
          #print('\nActual:\n--------------')
          #for aspect in ActualImplicitFeatureSet:
            #print(aspect)

          if len(ActualImplicitFeatureSet)==0:
            continue
          print(RawReviews[reviewNum][lineNum],'\t',InferredImplicitFeatureSet)
          TP+=len(InferredImplicitFeatureSet.intersection(ActualImplicitFeatureSet))
          FP+=len(InferredImplicitFeatureSet - ActualImplicitFeatureSet)
          FN+=len(ActualImplicitFeatureSet - InferredImplicitFeatureSet)

          for aspect in InferredImplicitFeatureSet.intersection(ActualImplicitFeatureSet):
            if aspect not in TPDict:
              TPDict[aspect]=1
            else:
              TPDict[aspect]+=1

          for aspect in InferredImplicitFeatureSet - ActualImplicitFeatureSet:
            if aspect not in FPDict:
              FPDict[aspect]=1
            else:
              FPDict[aspect]+=1

          for aspect in ActualImplicitFeatureSet - InferredImplicitFeatureSet:
            if aspect not in FNDict:
              FNDict[aspect]=1
            else:
              FNDict[aspect]+=1
          
          IdealFeatureList={}
          InferredFeatureList={}
          for aspect in PI[reviewNum][lineNum]:
            if aspect in ActualImplicitFeatureSet:
              IdealFeatureList[aspect]=1
              InferredFeatureList[aspect]=PI[reviewNum][lineNum][aspect]
            elif aspect not in ActualExplicitFeatureSet:
              InferredFeatureList[aspect]=PI[reviewNum][lineNum][aspect]
              IdealFeatureList[aspect]=0
          currentNDCG=NDCG.compute_NDCG(IdealFeatureList,InferredFeatureList)
          if currentNDCG!=-1:
            ndcgList.append(currentNDCG)
    try:  
      Precision[GlobalIteration].append(TP/(TP+FP))
    except:
      Precision[GlobalIteration].append(0)
    try:
      Recall[GlobalIteration].append(TP/(TP+FN))
    except:
      Recall[GlobalIteration].append(0)
    try:
      F_Measure[GlobalIteration].append((2*Precision[GlobalIteration][MyK]*Recall[GlobalIteration][MyK])/(Precision[GlobalIteration][MyK]+Recall[GlobalIteration][MyK]))
    except:
      F_Measure[GlobalIteration].append(0)

    print('P=='+str(ColorProbability))
    print(Precision[GlobalIteration][MyK], Recall[GlobalIteration][MyK], F_Measure[GlobalIteration][MyK])
    print('NDCG='+str(sum(ndcgList)/len(ndcgList)))
    print('###########################\n')
    GlobalNDCG[GlobalIteration].append(sum(ndcgList)/len(ndcgList))
    MyK+=1


with open(Results_Directory+'Summary.csv','w') as Summaryfile:
  Summaryfile.write('P,Precision,Recall,F_measure,NDCG\n')
  col_avg_Precision = [ sum(x)/len(Precision) for x in zip(*Precision)]
  col_avg_Recall = [ sum(x)/len(Recall) for x in zip(*Recall)]
  col_avg_F_Measure = [ sum(x)/len(F_Measure) for x in zip(*F_Measure)]
  col_avg_GlobalNDCG = [ sum(x)/len(GlobalNDCG) for x in zip(*GlobalNDCG)]
  for i in range(0,len(col_avg_Precision)):
    Summaryfile.write(str(i*0.05)+','+str(col_avg_Precision[i])+','+str(col_avg_Recall[i])+','+str(col_avg_F_Measure[i])+','+str(col_avg_GlobalNDCG[i])+'\n')


'''
print('TPS')
for word in sorted(TPDict, key=TPDict.get, reverse=True)[:10]:
    print(word, TPDict[word])
print('\n\n\n\nFPS')
for word in sorted(FPDict, key=FPDict.get, reverse=True)[:10]:
    print(word, FPDict[word])
print('\n\n\n\nFNS')
for word in sorted(FNDict, key=FNDict.get, reverse=True)[:10]:
    print(word, FNDict[word])
'''

for aspect in ExplicitFeatures:
  if ExplicitFeatures[aspect]>FrequentThreshold:
    if aspect not in TPDict:
      TPDict[aspect]=0
    if aspect not in FPDict:
      FPDict[aspect]=0
    if aspect not in FNDict:
      FNDict[aspect]=0
  
  
FeatureWisePrecision={}
FeatureWiseRecall={}
FeatureWiseFmeasure={}


for aspect in TPDict:
  try:
    FeatureWisePrecision[aspect]=TPDict[aspect]/(TPDict[aspect]+FPDict[aspect])
  except:
    FeatureWisePrecision[aspect]=0
  try:
    FeatureWiseRecall[aspect]=TPDict[aspect]/(TPDict[aspect]+FNDict[aspect])
  except:
    FeatureWiseRecall[aspect]=0
  try:
    FeatureWiseFmeasure[aspect]=(2*FeatureWisePrecision[aspect]*FeatureWiseRecall[aspect])/(FeatureWisePrecision[aspect]+FeatureWiseRecall[aspect])
  except:
    FeatureWiseFmeasure[aspect]=0


with open(Results_Directory+'FeatureWiseAnalysis.csv','w') as outputfile:
  outputfile.write('Precision\n')
  for aspect in sorted(FeatureWisePrecision, key=FeatureWisePrecision.get, reverse=True):
      outputfile.write(aspect+','+str(FeatureWisePrecision[aspect])+'\n')

  outputfile.write('\n\n\nRecall\n')
  for aspect in sorted(FeatureWiseRecall, key=FeatureWiseRecall.get, reverse=True):
      outputfile.write(aspect+','+str(FeatureWiseRecall[aspect])+'\n')

  outputfile.write('\n\n\nF-measure\n')
  for aspect in sorted(FeatureWiseFmeasure, key=FeatureWiseFmeasure.get, reverse=True):
      outputfile.write(aspect+','+str(FeatureWiseFmeasure[aspect])+'\n')






