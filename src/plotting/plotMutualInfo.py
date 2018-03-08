import pylab as plt
import numpy as np
import pickle

# itrList = [0,1000,2000,3000,4000,5000];
layerList =['Layer 1','Layer 2','Layer 3','Layer 4'];
    
for i in range(4):
# for i in [4-1]:
    pkl_file = open('data/mutualInfo_bin20_l'+str(i+1)+'.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
     
     
    IV_untrained = data[0];
    IA_untrained = data[1];    
    IV_trained = data[2];
    IA_trained = data[3];
    
    MI_untrained = np.concatenate((IV_untrained.flatten(),IA_untrained.flatten()))
    MI_trained = np.concatenate((IV_trained.flatten(),IA_trained.flatten()))
    
    
    pkl_file = open('data/mutualInfo_bin20_shuffled_l'+str(i+1)+'.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close();
     
     
    IV_shuffled = data[0];
    IA_shuffled = data[1];    
    
    MI_shuffled = np.concatenate((IV_shuffled.flatten(),IA_shuffled.flatten()))
    
    
    
    # ## plot the info
    plt.subplot(4,1,4-i);
    plt.plot(-np.sort(-MI_untrained.flatten()),':',label="MI_untrained");
    plt.plot(-np.sort(-MI_shuffled.flatten()),'--',label="MI_shuffled");
    plt.plot(-np.sort(-MI_trained.flatten()),label="MI_trained");
    plt.title("Mutual Information (Layer "+ str(i+1) + ")")
    plt.xlabel("s-r pair rank");
    plt.ylabel("Information [bit]")
    # plt.ylim((max(IV_trained.max(),IV_untrained.max())*-0.1,max(IV_trained.max(),IV_untrained.max())*1.1));
    plt.ylim([-0.1,1.1])
    plt.legend();
    
    
#     # ## plot the info (one modality at time)
#     plt.subplot(2,1,1);
#     plt.plot(-np.sort(-IV_trained.flatten()),label="IV_trained");
#     plt.plot(-np.sort(-IV_untrained.flatten()),label="IV_untrained");
#     plt.title("Visual Input Unit x Encoded Unit")
#     plt.xlabel("s-r pair rank");
#     plt.ylabel("Mutual Information [bit]")
#     # plt.ylim((max(IV_trained.max(),IV_untrained.max())*-0.1,max(IV_trained.max(),IV_untrained.max())*1.1));
#     plt.ylim([-0.1,1.1])
#     plt.legend();
#            
#     plt.subplot(2,1,2);
#     plt.plot(-np.sort(-IA_trained.flatten()),label="IA_trained");
#     plt.plot(-np.sort(-IA_untrained.flatten()),label="IA_untrained");
#     plt.title("Audio Input Unit x Encoded Unit")
#     plt.xlabel("s-r pair rank");
#     plt.ylabel("Mutual Information [bit]")
#     # plt.ylim((max(IA_trained.max(),IA_untrained.max())*-0.1,max(IA_trained.max(),IA_untrained.max())*1.1));
#     plt.ylim([-0.1,1.1])
    
    
plt.legend();
plt.subplots_adjust(hspace=1.0)
plt.show()




## display the info. table  
plt.subplot(3,2,1);
plt.gray()
# im = plt.imshow(IV_untrained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IV_untrained,axis=1),(28,28)),vmin=0,vmax=1);
plt.colorbar(im)
plt.title("Visual inputs (Untrained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
plt.subplot(3,2,2);
# im = plt.imshow(IA_untrained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IA_untrained,axis=1),(14,56)),vmin=0,vmax=1);

plt.colorbar(im)
plt.title("Audio inputs (Untrained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")



plt.subplot(3,2,3);
plt.gray()
# im = plt.imshow(IV_untrained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IV_shuffled,axis=1),(28,28)),vmin=0,vmax=1);
plt.colorbar(im)
plt.title("Visual inputs (Shuffled)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
plt.subplot(3,2,4);
# im = plt.imshow(IA_untrained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IA_shuffled,axis=1),(14,56)),vmin=0,vmax=1);

plt.colorbar(im)
plt.title("Audio inputs (Shuffled)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
    
plt.subplot(3,2,5)
# im = plt.imshow(IV_trained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IV_trained,axis=1),(28,28)),vmin=0,vmax=1);
plt.colorbar(im)
plt.title("Visual inputs (Trained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
    
plt.subplot(3,2,6);
# im = plt.imshow(IA_trained,vmin=0,vmax=1, aspect="auto");
im = plt.imshow(np.reshape(np.mean(IA_trained,axis=1),(14,56)),vmin=0,vmax=1);
plt.colorbar(im)
plt.title("Audio inputs (Trained)");
# plt.xlabel("Encoded Units");
# plt.ylabel("Input Units")
plt.show()
  

plt.gray();
for i in range(8):
    for j in range(8):
        index = i*8+j;
        plt.subplot(8,8,index+1);
        # im = plt.imshow(IV_trained,vmin=0,vmax=1, aspect="auto");
        im = plt.imshow(np.reshape(IV_trained[:,index],(28,28)),vmin=0,vmax=1);
        plt.colorbar(im)
#         plt.title("Visual inputs (Trained)");
#         plt.xlabel("Encoded Units");
#         plt.ylabel("Input Units")
plt.show()