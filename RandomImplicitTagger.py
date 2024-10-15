import random

EquivalentAspects={}
with open('Selected Features.txt','r') as inputfile:
    for line in inputfile:
      aspects=line.strip().split(',')
      for aspect in aspects:
        EquivalentAspects[aspect]=aspects[0]


FeatureLineMapping={}
lineNum=0
with open('iPod.data','r') as inputfile:
    for line in inputfile:
        lineNum+=1
        if (line.startswith('[t]') or line.startswith('*')):
            continue
        else:
            outputline=''
            segments=line.split('##')
            if len(segments[0])==0:
                continue
            for aspect in segments[0].split(','):
                if aspect.split('[')[0] in FeatureLineMapping:
                    FeatureLineMapping[aspect.split('[')[0]].append(lineNum)
                else:
                    FeatureLineMapping[aspect.split('[')[0]]=[]
                    FeatureLineMapping[aspect.split('[')[0]].append(lineNum)
        
'''
for aspect in sorted(FeatureLineMapping, key=FeatureLineMapping.get, reverse=True):
    print(aspect,len(FeatureLineMapping[aspect]))
'''

LinesToDelete={}
LineMappedtoAspect={}

for aspect in sorted(FeatureLineMapping, key=FeatureLineMapping.get, reverse=True):
    #LinesToDelete[aspect]=random.sample(FeatureLineMapping[aspect],(int)(len(FeatureLineMapping[aspect])*0.2))
    LinesToDelete[aspect]=random.sample(FeatureLineMapping[aspect],30)
    for lineNum in LinesToDelete[aspect]:
        if lineNum not in LineMappedtoAspect:
            LineMappedtoAspect[lineNum]=[]
        LineMappedtoAspect[lineNum].append(aspect)

for aspect in sorted(LinesToDelete, key=LinesToDelete.get, reverse=True):
    print(aspect,len(FeatureLineMapping[aspect]),len(LinesToDelete[aspect]))


lineNum=0
with open('iPod.final','w') as outputfile:
    with open('iPod.data','r') as inputfile:
        for line in inputfile:
            lineNum+=1
            if (line.startswith('[t]') or line.startswith('*')):
                outputfile.write(line)
            else:
                outputline=''
                segments=line.split('##')
                sentence=segments[1]
                if len(segments[0])==0 or lineNum not in LineMappedtoAspect:
                    outputfile.write(line)
                    continue
                i=0
                for aspect in segments[0].split(','):
                    if i>0:
                        outputline+=','
                    i+=1
                    if aspect.split('[')[0] in LineMappedtoAspect[lineNum]:
                        outputline+=aspect+'[u]'
                        sentence=sentence.replace(' '+aspect.split('[')[0],' ')
                        sentence=sentence.replace(aspect.split('[')[0]+' ',' ')
                    else:
                        outputline+=aspect
                outputline=outputline+'##'+sentence
                outputfile.write(outputline)
            

















        


