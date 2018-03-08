import pylab as plt
import numpy as np
import pandas as pd
w = 0.2

# 
# inconsistent = np.array([[71,46,23,1],[101,76,60,7],[104,99,83,22],[105,81,65,22]])
# consistent = np.array([[76,58,33,5],[108,101,90,56],[103,110,92,70],[102,75,65,57]])#1000 epoch


# inconsistent = np.array([[64,32,14,0],[96,67,52,6],[92,78,54,14],[91,56,33,11]])#5000 epoch
# consistent = np.array([[65,24,9,1],[110,101,89,52],[102,106,87,67],[92,71,62,54]])




inconsistent=np.array([[31,12,6,0],
[44,25,15,3],
[49,34,26,4],
[44,21,10,2]])
 
consistent=np.array([[41,20,12,0],#L1
[52,53,43,15],
[52,54,46,32],
[48,39,37,29]])

inconsistent[:,0]=inconsistent[:,0]-inconsistent[:,2]
inconsistent[:,1]=inconsistent[:,1]-inconsistent[:,2]
inconsistent[:,2]=inconsistent[:,2]-inconsistent[:,3]

consistent[:,0]=consistent[:,0]-consistent[:,2]
consistent[:,1]=consistent[:,1]-consistent[:,2]
consistent[:,2]=consistent[:,2]-consistent[:,3]


# itrList = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000];

differentLayers_consistent=np.array([[19,29,9,6],
                          [28,43,23,10],
                          [55,40,36,21],
                          [48,39,37,29]])

differentLayers_consistent[:,0]=differentLayers_consistent[:,0]-differentLayers_consistent[:,2]
differentLayers_consistent[:,1]=differentLayers_consistent[:,1]-differentLayers_consistent[:,2]
differentLayers_consistent[:,2]=differentLayers_consistent[:,2]-differentLayers_consistent[:,3]


differentLayers_inconsistent=np.array([[11,25,2,1],
                          [25,23,7,0],
                          [46,28,18,0],
                          [44,21,10,2]]);

differentLayers_inconsistent[:,0]=differentLayers_inconsistent[:,0]-differentLayers_inconsistent[:,2]
differentLayers_inconsistent[:,1]=differentLayers_inconsistent[:,1]-differentLayers_inconsistent[:,2]
differentLayers_inconsistent[:,2]=differentLayers_inconsistent[:,2]-differentLayers_inconsistent[:,3]


differentStruct=np.array([[47,41,33,15],
                          [48,39,37,29]]);
                          
differentStruct[:,0]=differentStruct[:,0]-differentStruct[:,2]
differentStruct[:,1]=differentStruct[:,1]-differentStruct[:,2]
differentStruct[:,2]=differentStruct[:,2]-differentStruct[:,3]


df_inconsistent = pd.DataFrame(inconsistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_consistent = pd.DataFrame(consistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_layers_consist = pd.DataFrame(differentLayers_consistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_layers_inconsist = pd.DataFrame(differentLayers_inconsistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_structs = pd.DataFrame(differentStruct, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])


ax = plt.subplot(3,2,1)
df_inconsistent.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Layer')
plt.title("Inconsistent V+A")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )


ax=plt.subplot(3,2,2)
df_consistent.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Layer')
plt.title("Consistent V+A")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )



ax=plt.subplot(3,2,3)
df_layers_inconsist.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Number of Layers')
plt.title("Networks with Different Number of Layers (inconsist)")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('1 Layer', '2 Layers','3 Layers', '4 Layers') )




ax=plt.subplot(3,2,4)
df_layers_consist.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Number of Layers')
plt.title("Networks with Different Number of Layers (consist)")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('1 Layer', '2 Layers','3 Layers', '4 Layers') )







ax=plt.subplot(3,2,5)
df_structs.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Different Structures')
plt.title("Networks with Different Structures")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('2 stage framework', 'mixed framework') )
# 
# 
# 
# 
# 
# ind = np.arange(len(itrList));
# 
# inconsistent = np.transpose(inconsistent);
# consistent = np.transpose(consistent);
# 
# ax= plt.subplot(1,2,1)
# for i in range(4):
#     ax.bar(ind+w*i,inconsistent[i],width=w*0.9,color=plt.cm.tab10(i),align='center')
# ax.set_xticks(ind+w)
# # ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )
# ax.set_xticklabels(itrList);
# 
# plt.ylabel('Number of Cells')
# plt.xlabel('Number of Layer')
# plt.title("Inconsistent V+A")
# plt.ylim([-2,124])
# 
# 
# 
# 
# ax = plt.subplot(1,2,2)
# for i in range(4):
#     ax.bar(ind+w*i,consistent[i],width=w*0.9,color=plt.cm.tab10(i),align='center')
#     
# ax.set_xticks(ind+w)
# ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )
# plt.ylabel('Number of Cells')
# plt.xlabel('Number of Layer')
# plt.title("Consistent V+A")
# plt.ylim([-2,124])

plt.show()











