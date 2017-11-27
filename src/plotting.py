from params import *
import matplotlib.gridspec as gridspec

def plotResults(model,input1,input2):
    decoded_imgs = model.predict([input1, input2]);
    for i in range(10):
        plt.gray()
        plt.subplot(4,10,i+1)
        plt.imshow(input1[i*50])
        plt.title(i)
        plt.subplot(4,10,10+i+1)
        plt.imshow(input2[i*50])
        plt.subplot(4,10,20+i+1)
        plt.imshow(decoded_imgs[0][i*50])
        plt.subplot(4,10,30+i+1)
        plt.imshow(decoded_imgs[1][i*50])
    plt.show()
    
def plotActivityOfCellsWithMaxInfo(IRs,results, title="Firing Properties of the cells with the highest single cell information"):
    fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    resultNorm=results-np.min(results);
    resultNorm=resultNorm/np.max(resultNorm)
    nObj = np.shape(results)[0];
    nTrans = np.shape(results)[1];
    
    for objIndex in range(nObj):
        # objIndex = 0;
    
        pts_max = np.unravel_index(np.argmax(IRs[objIndex]), IRs[0].shape)
#         print(IRs[objIndex,pts_max[0],pts_max[1],pts_max[2]])
    
        # frTable = resultNorm[:,:,pts[0,0],pts[0,1],pts[0,2]]
        # frTable = np.int32(resultNorm[:,:,pts_max[0],pts_max[1],pts_max[2]]/0.3)
        frTable = resultNorm[:,:,pts_max[0],pts_max[1],pts_max[2]];
        frTableNorm = frTable-np.min(frTable);
        frTableNorm = frTableNorm/np.max(frTableNorm);
    
        gs = gridspec.GridSpec(nObj, 5);
    
        plt.subplot(gs[objIndex,:4])
        plt.imshow(1-frTableNorm, interpolation='nearest',aspect='auto',vmin=0, vmax=1);
        plt.title("Obj " + str(objIndex) + " : " + "{:10.3f}".format(IRs[objIndex,pts_max[0],pts_max[1],pts_max[2]]) + " bit; cell:"+str(pts_max));
        # print(frTable)
        plt.subplot(gs[objIndex,4])
        plt.barh(range(nObj),np.sum(frTable,axis=1),height=0.8)
        plt.xlim((0,nTrans))
        plt.gca().invert_yaxis()
        plt.margins(y=0)
        cur_axes = plt.gca()
#         cur_axes.axes.get_yaxis().set_ticklabels([])
#         plt.axis('off')
        plt.gray()
        plt.suptitle(title)
    
    plt.show()
    
    # fr = [];
    # for pts_cpy in pts:
    #     fr.append(results_comb[:,:,pts_cpy[0],pts_cpy[1],pts_cpy[2]])
    
    # print(fr)
    
