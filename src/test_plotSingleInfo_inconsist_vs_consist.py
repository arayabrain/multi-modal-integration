import pickle
import numpy as np
import pylab as plt

nObj = 10;

for l in [3]:# range(4):
# for l in range(4):

#     pkl_file = open('data/singleCellInfo_inconsistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_vs_inconsistent_V-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_V = pickle.load(pkl_file)
    pkl_file.close();
    
    #    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_vs_inconsistent_A-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_A = pickle.load(pkl_file)
    pkl_file.close();
    
#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
#     data_oneModalityAtTime = pickle.load(pkl_file)
#     pkl_file.close();

    IRs = data_V[0]
    IRs_flattened_V_inconsist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_V_inconsist[obj] = IRs[obj].flatten();    
    IRs_sorted_V_inconsist = np.sort(IRs_flattened_V_inconsist*-1)*-1;

    
    IRs = data_V[1]
    IRs_flattened_V_consist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_V_consist[obj] = IRs[obj].flatten();    
    IRs_sorted_V_consist = np.sort(IRs_flattened_V_consist*-1)*-1;

    IRs = data_A[0]
    IRs_flattened_A_inconsist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_A_inconsist[obj] = IRs[obj].flatten();    
    IRs_sorted_A_inconsist = np.sort(IRs_flattened_A_inconsist*-1)*-1;

    
    IRs = data_A[1]
    IRs_flattened_A_consist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_A_consist[obj] = IRs[obj].flatten();    
    IRs_sorted_A_consist = np.sort(IRs_flattened_A_consist*-1)*-1;


       
    # plt.subplot(4,1,1)
    # plt.plot(np.transpose(IRs_sorted_untrained));
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("untrained network")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    # 
    # plt.subplot(4,1,2)
    # plt.plot(np.transpose(IRs_sorted_trained_inconsistent)); 
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("trained network (inconsistent)")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    # plt.subplot(4,1,3)   
    # plt.plot(np.transpose(IRs_sorted_trained_consistent)); 
    # plt.ylabel("single cell info [bit]")
    # plt.xlabel("cell rank")
    # plt.title("trained network (consistent)")
    # plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    
#     plt.subplot(4,2,2*(4-l)-1)   
    plt.subplot(1,2,1) 
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_V_inconsist,axis=0)*-1)*-1),'--',label='V_inconsistant')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_V_consist,axis=0)*-1)*-1),'-',label='V_consistant')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("Single Cell Information Analysis (Vision)")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    plt.legend()

#     plt.subplot(4,2,2*(4-l))   
    plt.subplot(1,2,2) 
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_A_inconsist,axis=0)*-1)*-1),'--',label='V_inconsistant')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_A_consist,axis=0)*-1)*-1),'-',label='V_consistant')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("Single Cell Information Analysis (Audio)")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    plt.legend()


# plt.subplot(2,1,2)
# plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)

plt.subplots_adjust(hspace=1.0)
plt.show()


