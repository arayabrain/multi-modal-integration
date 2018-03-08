import pickle
import numpy as np
import pylab as plt

nObj = 10;

for l in [3]:# range(4):
# for l in range(4):

#     pkl_file = open('data/singleCellInfo_inconsistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_V-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_consistent_V = pickle.load(pkl_file)
    pkl_file.close();

    pkl_file = open('data/singleCellInfo_consistent_A-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_consistent_A = pickle.load(pkl_file)
    pkl_file.close();

    
    #    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_inconsistent_V-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_inconsistent_V = pickle.load(pkl_file)
    pkl_file.close();
    
    #    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_inconsistent_A-Only_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_inconsistent_A = pickle.load(pkl_file)
    pkl_file.close();
    
    
    pkl_file = open('data/singleCellInfo_inconsistent_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_inconsistent = pickle.load(pkl_file)
    pkl_file.close();
    
    #    pkl_file = open('data/mutualInfo_bin50_l'+str(i+1)+'.pkl', 'rb')
    pkl_file = open('data/singleCellInfo_consistent_bin10_l'+str(l+1)+'.pkl', 'rb')
    data_consistent = pickle.load(pkl_file)
    pkl_file.close();
    
    
    
#     pkl_file = open('data/singleCellInfo_consistent_oneModalityAtTime_bin10_l'+str(l+1)+'.pkl', 'rb')
#     data_oneModalityAtTime = pickle.load(pkl_file)
#     pkl_file.close();

    IRs = data_inconsistent_V[0]
    IRs_flattened_V_untrained = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_V_untrained[obj] = IRs[obj].flatten();    
    IRs_sorted_V_untrained = np.sort(IRs_flattened_V_untrained*-1)*-1;


    IRs = data_inconsistent_V[1]
    IRs_flattened_V_inconsist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_V_inconsist[obj] = IRs[obj].flatten();    
    IRs_sorted_V_inconsist = np.sort(IRs_flattened_V_inconsist*-1)*-1;

    
    IRs = data_consistent_V[1]
    IRs_flattened_V_consist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_V_consist[obj] = IRs[obj].flatten();    
    IRs_sorted_V_consist = np.sort(IRs_flattened_V_consist*-1)*-1;





    IRs = data_inconsistent_A[0]
    IRs_flattened_A_untrained = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_A_untrained[obj] = IRs[obj].flatten();    
    IRs_sorted_A_untrained = np.sort(IRs_flattened_A_untrained*-1)*-1;

    IRs = data_inconsistent_A[1]
    IRs_flattened_A_inconsist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_A_inconsist[obj] = IRs[obj].flatten();    
    IRs_sorted_A_inconsist = np.sort(IRs_flattened_A_inconsist*-1)*-1;

    
    IRs = data_consistent_A[1]
    IRs_flattened_A_consist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_A_consist[obj] = IRs[obj].flatten();    
    IRs_sorted_A_consist = np.sort(IRs_flattened_A_consist*-1)*-1;






    IRs = data_inconsistent[0]
    IRs_flattened_untrained = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_untrained[obj] = IRs[obj].flatten();    
    IRs_sorted_untrained = np.sort(IRs_flattened_untrained*-1)*-1;


    IRs = data_consistent[1]
    IRs_flattened_consist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_consist[obj] = IRs[obj].flatten();    
    IRs_sorted_consist = np.sort(IRs_flattened_consist*-1)*-1;


    IRs = data_inconsistent[1]
    IRs_flattened_inconsist = np.zeros((nObj,64));
    for obj in range(nObj):
        IRs_flattened_inconsist[obj] = IRs[obj].flatten();    
    IRs_sorted_inconsist = np.sort(IRs_flattened_inconsist*-1)*-1;

    
       
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
    
    
    #     plt.subplot(4,2,2*(4-l))   
    plt.subplot(2,2,1) 
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1),':',label='All_untrained')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_inconsist,axis=0)*-1)*-1),'--',label='All_inconsistant')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_consist,axis=0)*-1)*-1),'-',label='All_consistant')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("(a) Single Cell Information Analysis (Entire Testing Set)")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    plt.legend()
    
#     plt.subplot(4,2,2*(4-l)-1)   
    plt.subplot(2,2,3) 
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_V_untrained,axis=0)*-1)*-1),':',label='V_untrained')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_V_inconsist,axis=0)*-1)*-1),'--',label='V_inconsistant')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_V_consist,axis=0)*-1)*-1),'-',label='V_consistant')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("(b) Single Cell Information Analysis (Vision Only)")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    plt.legend()

#     plt.subplot(4,2,2*(4-l))   
    plt.subplot(2,2,4) 
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_A_untrained,axis=0)*-1)*-1),':',label='A_untrained')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_A_inconsist,axis=0)*-1)*-1),'--',label='A_inconsistant')
    plt.plot(np.transpose(np.sort(np.max(IRs_flattened_A_consist,axis=0)*-1)*-1),'-',label='A_consistant')
    plt.ylabel("single cell info [bit]")
    plt.xlabel("cell rank")
#     plt.title("Layer " + str(l+1))
    plt.title("(c) Single Cell Information Analysis (Audio Only)")
    plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    plt.legend()


# plt.subplot(2,1,2)
# plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)

plt.subplots_adjust(hspace=1.0)
plt.show()


