import pylab as plt
import numpy as np
import pandas as pd
w = 0.2

# 
# inconsistent = np.array([[71,46,23,1],[101,76,60,7],[104,99,83,22],[105,81,65,22]])
# consistent = np.array([[76,58,33,5],[108,101,90,56],[103,110,92,70],[102,75,65,57]])#1000 epoch


# inconsistent = np.array([[64,32,14,0],[96,67,52,6],[92,78,54,14],[91,56,33,11]])#5000 epoch
# consistent = np.array([[65,24,9,1],[110,101,89,52],[102,106,87,67],[92,71,62,54]])




differentLayers_inconsistent=np.array([[12.0,21.0,4.0,1.0],
                       [20.0,17.0,7.0,0.0],
                       [28.0,7.0,22.0,0.0],
                       [33.0,9.0,13.0,2.0]])
 
differentLayers_consistent=np.array([[10.0,24.0,3.0,7.0],
                     [3.0,16.0,17.0,13.0],
                     [16.0,3.0,15.0,26.0],
                     [12.0,3.0,9.0,29.0]])



# itrList = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000];



differentStruct=np.array([[14.0,9.0,16.0,17.0],
                          [12.0,3.0,9.0,29.0]]);
                          


# df_inconsistent = pd.DataFrame(inconsistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
# df_consistent = pd.DataFrame(consistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
# df_layers_consist = pd.DataFrame(differentLayers_consistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
# df_layers_inconsist = pd.DataFrame(differentLayers_inconsistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
# df_structs = pd.DataFrame(differentStruct, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])


df_layers_consist = pd.DataFrame(differentLayers_consistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_layers_inconsist = pd.DataFrame(differentLayers_inconsistent, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])
df_structs = pd.DataFrame(differentStruct, columns=['V-only', 'A-only', 'V+A(inconsistent)', 'V+A(consistent)'])


# ax = plt.subplot(3,2,1)
# df_inconsistent.plot(kind='bar', stacked=True,ax=plt.gca());
# plt.ylabel('Number of Cells')
# plt.xlabel('Layer')
# plt.title("Inconsistent V+A")
# plt.ylim([-2,66])
# # ax.set_xticklabels(itrList);
# ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )
# 
# 
# ax=plt.subplot(3,2,2)
# df_consistent.plot(kind='bar', stacked=True,ax=plt.gca());
# plt.ylabel('Number of Cells')
# plt.xlabel('Layer')
# plt.title("Consistent V+A")
# plt.ylim([-2,66])
# # ax.set_xticklabels(itrList);
# ax.set_xticklabels( ('Layer 1','Layer 2','Layer 3','Layer 4') )



ax=plt.subplot(1,3,1)
df_layers_inconsist.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Number of Layers')
plt.title("Networks with Different Number of Layers (inconsist)")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('1 Layer', '2 Layers','3 Layers', '4 Layers') )




ax=plt.subplot(1,3,2)
df_layers_consist.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Number of Layers')
plt.title("Networks with Different Number of Layers (consist)")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('1 Layer', '2 Layers','3 Layers', '4 Layers') )







ax=plt.subplot(1,3,3)
df_structs.plot(kind='bar', stacked=True,ax=plt.gca());
plt.ylabel('Number of Cells')
plt.xlabel('Different Structures')
plt.title("Networks with Different Structures")
plt.ylim([-2,66])
# ax.set_xticklabels(itrList);
ax.set_xticklabels( ('2-Stage Framework', 'Mixed-Input Framework') )
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











