import Stemming
import nltk
import math
import numpy as np, numpy.random
from nltk.corpus import wordnet as wn
import NDCG


Data_Directory='./Annotated Data/Router/'
DataFile='Router.final'
Results_Directory='./Results/Router/Naive Bayes/'
EquivalenceFile='Selected Features.txt'
#EquivalenceFile='All_ExplicitFeatures.txt'

punctuation = "'.,:;!?0123456789"
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
    input_string1 = input_string1.replace(item, '')
  return input_string1



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

implicitfeatureOccurrenecs=0
NoOfReviews=-1
i=0
j=0
for line in TrainingData:
  i+=1
  TotalNoOfWords+=len(line.split())
  if (line.startswith('[t]') or line.startswith('*')):
    NoOfReviews+=1
    Reviews.append(list())
    RawReviews.append(list())
    GroundTruth.append(list())
    GroundTruth[NoOfReviews].append(list())
    j=-1
  else:      
    
    GroundTruth[NoOfReviews].append(list())
    review_with_aspect=line.replace('(','').replace(')','').replace('-','').split('##')

    review_Sentence=remove_punctuation(review_with_aspect[1])
    if review_Sentence.strip()=='':
      continue
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
          implicitfeatureOccurrenecs+=1
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
    TopicModel[aspect][word]=math.log(1+TopicModel[aspect][word])



for aspect in TopicModel:
  sumOfTFIDF=sum(TopicModel[aspect].values())
  for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True):
    TopicModel[aspect][word]=(TopicModel[aspect][word]+1)/(sumOfTFIDF+len(Collection))
    

with open(Results_Directory+'TopicModel.txt','w') as outputFile:
  for aspect in TopicModel:
    for word in sorted(TopicModel[aspect], key=TopicModel[aspect].get, reverse=True)[:10]:
      outputFile.write(aspect+', '+word+', '+str(TopicModel[aspect][word])+'\n')
    outputFile.write('\n\n\n')


'''
for word in sorted(BackgroundProbability, key=BackgroundProbability.get, reverse=True)[:10]:
    print(word, BackgroundProbability[word])
'''

with open(Results_Directory+'ExplicitFeatures.txt','w') as outputfile:
  for aspect in sorted(ExplicitFeatures, key=ExplicitFeatures.get, reverse=True):
      outputfile.write(aspect + ' : ' + str(ExplicitFeatures[aspect])+'\n')



PI=[]
for reviewNum in range(0,len(Reviews)):
  PI.append(list())
  for lineNum in range(0,len(Reviews[reviewNum])):
    PI[reviewNum].append({})
    mySum=0
    for aspect in TopicModel:
      PI[reviewNum][lineNum][aspect]=ExplicitFeatures[aspect]
      for word in Reviews[reviewNum][lineNum]:
        PI[reviewNum][lineNum][aspect]=PI[reviewNum][lineNum][aspect]*TopicModel[aspect][word]
      mySum+=PI[reviewNum][lineNum][aspect]
    for aspect in TopicModel:
      PI[reviewNum][lineNum][aspect]/=mySum
    

TPDict={}
FNDict={}
FPDict={}
##Test Data Evaluation
with open(Results_Directory+'Summary.csv','w') as Summaryfile:
  Summaryfile.write('Theta,Precision,Recall,F_measure,NDCG\n')
  MyK=0
  for Theta in np.arange(0.05, 0.09, 0.05):  
    TP=0
    FP=0
    FN=0
    FrequentThreshold=0.0
    ndcgList=[]

    NoOFImplicitSentences=0
    for reviewNum in range(0,len(Reviews)):
      for lineNum in range(0,len(Reviews[reviewNum])):
        if '[u]' in RawReviews[reviewNum][lineNum]:
          NoOFImplicitSentences+=1
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
            if PI[reviewNum][lineNum][aspect]>=Theta and aspect not in ActualExplicitFeatureSet and ExplicitFeatures[aspect]>=FrequentThreshold:
              #print(aspect, PI[reviewNum][lineNum][aspect])
              InferredImplicitFeatureSet.add(aspect)

          for aspect in ExplicitFeatures:
            if ExplicitFeatures[aspect]<FrequentThreshold:
              try:
                ActualImplicitFeatureSet.remove(aspect)
              except:
                emni=1
          #print('\nActual:\n--------------')
          #for aspect in ActualImplicitFeatureSet:
            #print(aspect)
          if len(ActualImplicitFeatureSet)==0:
            continue
          
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
      Precision=TP/(TP+FP)
    except:
      Precision=0
    try:
      Recall=TP/(TP+FN)
    except:
      Recall=0
    try:
      F_Measure=(2*Precision*Recall)/(Precision+Recall)
    except:
      F_Measure=0

    print('Theta=='+str(Theta))
    print(Precision, Recall, F_Measure)
    print('NDCG='+str(sum(ndcgList)/len(ndcgList)))
    print('###########################\n')
    Summaryfile.write(str(Theta)+','+str(Precision)+','+str(Recall)+','+str(F_Measure)+','+str(sum(ndcgList)/len(ndcgList))+'\n')
    with open(Results_Directory+'NDCG List Theta='+str(Theta)+'.ndcg','w') as outputfile:
      for ndcg in ndcgList:
        outputfile.write(str(ndcg)+'\n')
    MyK+=1
        
print('Total implicit occurrence: '+str(implicitfeatureOccurrenecs))
print('Sentences with implicit feature: '+str(NoOFImplicitSentences))

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





