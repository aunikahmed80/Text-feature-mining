
EquivalentAspects={}
with open('Selected Features.txt','r') as inputfile:
    for line in inputfile:
      aspects=line.strip().split(',')
      for aspect in aspects:
        EquivalentAspects[aspect]=aspects[0]

with open('iPod.data','w') as outputfile:
    with open('iPod.Large','r') as inputfile:
        for line in inputfile:
            if (line.startswith('[t]') or line.startswith('*')):
                outputfile.write(line)
            else:
                outputline=''
                segments=line.split('##')
                i=0
                for aspect in EquivalentAspects:
                    if aspect in segments[1].split() and EquivalentAspects[aspect] not in outputline:
                        if i>0:
                          outputline+=','
                        i+=1
                        outputline+=EquivalentAspects[aspect]+'[@]'
                outputline=outputline+'##'+segments[1]
                outputfile.write(outputline)
                        
                        
            
