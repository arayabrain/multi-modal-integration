from params import *
from sklearn.decomposition import PCA
from blaze.expr.expressions import label
from sklearn.linear_model import LogisticRegression


# from statsmodels.graphics.tukeyplot import results
def singleCellInfoAnalysis(results_untrained,results_trained,plotOn=True,thresholdMode = False, nBins=3,threshold = 0.7):
    if thresholdMode:
        nBins=2;
    nObj = np.shape(results_untrained)[0];
    nTrans = np.shape(results_untrained)[1];
    nRow = np.shape(results_untrained)[2]
    nCol = np.shape(results_untrained)[3]
    nDep = np.shape(results_untrained)[4]
    

    IRs_list = [];#np.zeros((nObj,nRow,nCol,nDep));#I(R,s) single cell information
    IRs_weighted_list = []#np.zeros((nObj,nRow,nCol,nDep));#I(R,s) single cell information
    IRs_flattened_list = []
    IRs_sorted_list = []#np.zeros((nObj,nRow*nCol*nDep));



    for results in [results_untrained, results_trained]:
        #normalise
        resultNorm=results-np.min(results);
        resultNorm=resultNorm/np.max(resultNorm)
        
        #binning
        binned = np.zeros((nRow,nCol,nDep,nObj,nBins));
        
        for row in range(nRow):
            for col in range(nCol):
                for dep in range(nDep):
                    for s in range(nObj):
                        for t in range(nTrans):
                            if thresholdMode:
                                b = 1 if resultNorm[s,t,row,col,dep]>threshold else 0;
                            else:
#                                 if np.max(resultNorm[s,:,row,col,dep])>0.01:
#                                     b = int(min(np.floor(resultNorm[s,t,row,col,dep]/np.max(resultNorm[s,:,row,col,dep])*nBins),nBins-1))
#                                 else:  
                                b = int(min(np.floor(resultNorm[s,t,row,col,dep]*nBins),nBins-1))
                            binned[row,col,dep,s,b]=binned[row,col,dep,s,b]+1
                        
                        
        sumPerBin = np.zeros((nRow,nCol,nDep,nBins));
        sumPerObj = nTrans;
        sumPerCell = nTrans*nObj;
        IRs = np.zeros((nObj,nRow,nCol,nDep));#I(R,s) single cell information
        IRs_weighted = np.zeros((nObj,nRow,nCol,nDep));#I(R,s) single cell information
        pq_r = np.zeros(nObj)#prob(s') temporally used in the decoing process
        Ps = 1./nObj   #Prob(s) 
        
        
        for row in range(nRow):
            for col in range(nCol):
                for dep in range(nDep):
                    for b in range(nBins):
                        for obj in range(nObj):
                            sumPerBin[row,col,dep,b]+=binned[row,col,dep,obj,b];
                    for obj in range(nObj):    
                        for b in range(nBins):
                            Pr = sumPerBin[row,col,dep,b]/sumPerCell;
                            Prs = binned[row,col,dep,obj,b]/sumPerObj;
#                             if(Pr!=0 and Prs!=0):
                            if(Pr!=0 and Prs!=0 and Pr<Prs):
                                IRs[obj,row,col,dep]+=(Prs*(np.log2(Prs/Pr)));
                                IRs_weighted[obj,row,col,dep]+=(Prs*(np.log2(Prs/Pr)))*(b/(nBins-1));
                            
        IRs_flattened = np.zeros((nObj,nRow*nCol*nDep));
        for obj in range(nObj):
            IRs_flattened[obj] = IRs[obj].flatten();    
        IRs_sorted = np.sort(IRs_flattened*-1)*-1;
        
        IRs_list.append(IRs);
        IRs_weighted_list.append(IRs_weighted);
        IRs_flattened_list.append(IRs_flattened);
        IRs_sorted_list.append(IRs_sorted);
    
    if plotOn:
        labelList = [];
        for o in range(nObj):
            labelList.append(str(o));
                
        plt.subplots_adjust(wspace=0.4, hspace=1.5)
        plt.subplot(3,1,1)
        plt.plot(np.transpose(IRs_sorted_list[0]));
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.legend(labelList)
        
        plt.subplot(3,1,2)
        plt.plot(np.transpose(IRs_sorted_list[1])); 
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("trained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.legend(labelList)
        plt.subplot(3,1,3)   
#         plt.plot(np.transpose([np.sort(np.max(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.max(IRs_flattened_list[1],axis=0)*-1)*-1]));
        plt.plot(np.transpose([np.sort(np.max(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.max(IRs_flattened_list[1],axis=0)*-1)*-1]));
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained v trained network (max vals are taken)")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    
        # plt.subplot(2,1,2)
        # plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)
        
        plt.show()
    return (IRs_list,IRs_weighted_list);







def countCellsWithSelectivity(infoList1, infoList2, results, plotOn=True,infoThreshold = 1.0):
    ## count number of cells developed to be selective to a stimulus    
    nObj = np.shape(results)[0];
    nTrans = np.shape(results)[1];
    
    
    indexShape = np.shape(infoList2[0][0]);
       
    #counting a number of cells that is selective to any of the visual stimulus and to any of the audio stimulus         
    cond_1_untrained = np.zeros(indexShape,dtype=bool) 
    cond_1_trained = np.zeros(indexShape,dtype=bool) 
    cond_2_untrained = np.zeros(indexShape,dtype=bool)
    cond_2_trained = np.zeros(indexShape,dtype=bool) 
    
    infoThreshold1=0.9629458012826149#np.percentile(infoList1[1],80);
    infoThreshold2=0.9440938439990594#np.percentile(infoList2[1],80);
    
#     #take max info of each cell
#     IRs_flattened = np.zeros((nObj,indexShape[0]));
#     for obj in range(nObj):
#         IRs_flattened[obj] = infoList1[1][obj].flatten();    
# 
#     IR_max=np.max(IRs_flattened,axis=0);
            
    count_1_trained = 0;
    count_1_untrained = 0;
    
    for s in range(nObj):
        cond_2_untrained = (cond_2_untrained | (infoList1[0][s]>infoThreshold1)) # check if cells are selective to any of the visual inpputs
        cond_2_trained = (cond_2_trained | (infoList1[1][s]>infoThreshold1))

        cond_1_untrained = (cond_1_untrained|(infoList2[0][s]>infoThreshold2)) # check if cells are selective to any of the auditory inputs
        cond_1_trained = (cond_1_trained| (infoList2[1][s]>infoThreshold2))
        

    count_1_untrained = len(infoList2[0][0,cond_1_untrained & cond_2_untrained]); # number of cells that is selective to at least one visual input and one auditory input
    count_1_trained = len(infoList2[1][0,cond_1_trained & cond_2_trained]);
    
    
    
#     ## (testing addition) plot cells that shows selectivity to the signals from only one modality 
#     if plotOn:
#         fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
#         plt.subplots_adjust(wspace=0.4, hspace=1.0)
#         resultNorm=results-np.min(results);
#         resultNorm=resultNorm/np.max(resultNorm)
# 
#         
#         import matplotlib.gridspec as gridspec
#         maxSubplot = 10;
#         plotIndex = 0;
#         gs = gridspec.GridSpec(maxSubplot, 5);
#         
#         for s in range(nObj):
#             # objIndex = 0;
#             cond_1 = (infoList2[1][s]>infoThreshold2)
#             cond_2 = (infoList1[1][s]>infoThreshold1)
#             
#             
#             pts=np.argwhere((cond_1 ^ cond_2) & (cond_1_trained ^ cond_2_trained));    
#             plt.gray()
#             
#             for p in range(len(pts)):
#                 frTable = resultNorm[:,:,pts[p,0],pts[p,1],pts[p,2]];
#                 frTableNorm = frTable-np.min(frTable);
#                 frTableNorm = frTableNorm/np.max(frTableNorm);
#             
#                 plt.subplot(gs[plotIndex,:2])
#                 plt.imshow(1-frTableNorm[:,:50], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
#                 plt.title("Obj " + str(np.argmax(infoList1[1][:,pts[p,0],pts[p,1],pts[p,2]])) + ": V - " + "{:10.3f}".format(np.max(infoList1[1][:,pts[p,0],pts[p,1],pts[p,2]]))  + " bit; cell:"+str(pts[p]));
#                 
#                 plt.subplot(gs[plotIndex,2:4])
#                 plt.imshow(1-frTableNorm[:,50:100], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
# #                 plt.imshow(frTableNorm, interpolation='nearest',aspect='auto',vmin=0, vmax=1);
#                 plt.title("Obj " + str(np.argmax(infoList2[1][:,pts[p,0],pts[p,1],pts[p,2]])) + ": A - " + "{:10.3f}".format(np.max(infoList2[1][:,pts[p,0],pts[p,1],pts[p,2]]))  + " bit; cell:"+str(pts[p]));
#                 # print(frTable)
#                 
#                 plt.subplot(gs[plotIndex,4])
#                 plt.barh(range(nObj),np.sum(frTable,axis=1),height=0.8,color='k')
#                 plt.xlim((0,nTrans))
#                 plt.gca().invert_yaxis()
#                 plt.margins(y=0)
#                 cur_axes = plt.gca()
#             #         cur_axes.axes.get_yaxis().set_ticklabels([])
#             #         plt.axis('off')
#                 
#                 plt.suptitle("plot cells where info about both A and V are relatively high")
#                 plotIndex+=1;
#                 if plotIndex>=maxSubplot:
#                     plt.show()
#                     plt.clf()
#                     fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
# #                     plt.figure.figsize=(18, 16);
# #                     plt.figure.dpi = 70;
#                     plt.subplots_adjust(wspace=0.4, hspace=0.6)
#                     plotIndex=0;
#         plt.show();
    
#     ## (testing addition) plot only inconsistent
#     if plotOn:
#         fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
#         plt.subplots_adjust(wspace=0.4, hspace=1.0)
#         resultNorm=results-np.min(results);
#         resultNorm=resultNorm/np.max(resultNorm)
# 
#         
#         import matplotlib.gridspec as gridspec
#         maxSubplot = 10;
#         plotIndex = 0;
#         gs = gridspec.GridSpec(maxSubplot, 5);
#         
#         for s in range(nObj):
#             # objIndex = 0;
#             cond_1 = (infoList2[1][s]>infoThreshold1)
#             cond_2 = (infoList1[1][s]>infoThreshold2)
#             
#             
#             pts=np.argwhere((cond_1 ^ cond_2) & (cond_1_trained & cond_2_trained));    
#             plt.gray()
#             
#             for p in range(len(pts)):
#                 frTable = resultNorm[:,:,pts[p,0],pts[p,1],pts[p,2]];
#                 frTableNorm = frTable-np.min(frTable);
#                 frTableNorm = frTableNorm/np.max(frTableNorm);
#             
#                 plt.subplot(gs[plotIndex,:2])
#                 plt.imshow(1-frTableNorm[:,:50], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
#                 plt.title("Obj " + str(np.argmax(infoList1[1][:,pts[p,0],pts[p,1],pts[p,2]])) + ": V - " + "{:10.3f}".format(np.max(infoList1[1][:,pts[p,0],pts[p,1],pts[p,2]]))  + " bit; cell:"+str(pts[p]));
#                 
#                 plt.subplot(gs[plotIndex,2:4])
#                 plt.imshow(1-frTableNorm[:,50:100], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
# #                 plt.imshow(frTableNorm, interpolation='nearest',aspect='auto',vmin=0, vmax=1);
#                 plt.title("Obj " + str(np.argmax(infoList2[1][:,pts[p,0],pts[p,1],pts[p,2]])) + ": A - " + "{:10.3f}".format(np.max(infoList2[1][:,pts[p,0],pts[p,1],pts[p,2]]))  + " bit; cell:"+str(pts[p]));
#                 # print(frTable)
#                 
#                 plt.subplot(gs[plotIndex,4])
#                 plt.barh(range(nObj),np.sum(frTable,axis=1),height=0.8,color='k')
#                 plt.xlim((0,nTrans))
#                 plt.gca().invert_yaxis()
#                 plt.margins(y=0)
#                 cur_axes = plt.gca()
#             #         cur_axes.axes.get_yaxis().set_ticklabels([])
#             #         plt.axis('off')
#                 
#                 plt.suptitle("plot cells where info about both A and V are relatively high")
#                 plotIndex+=1;
#                 if plotIndex>=maxSubplot:
#                     plt.show()
#                     plt.clf()
#                     fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
# #                     plt.figure.figsize=(18, 16);
# #                     plt.figure.dpi = 70;
#                     plt.subplots_adjust(wspace=0.4, hspace=0.6)
#                     plotIndex=0;
#         plt.show();
    


    #counting a number of cells that is selective to at least one consistent stimulus
    cond_trained = np.zeros(indexShape,dtype=bool)
    cond_untrained = np.zeros(indexShape,dtype=bool)

    count_2_trained = 0;
    count_2_untrained = 0;
    for s in range(nObj):
        cond_1_untrained = (infoList1[0][s]>infoThreshold1)
        cond_1_trained = (infoList1[1][s]>infoThreshold1)
        
        cond_2_untrained = (infoList2[0][s]>infoThreshold2)
        cond_2_trained = (infoList2[1][s]>infoThreshold2)
        
        cond_untrained= (cond_untrained | (cond_1_untrained & cond_2_untrained));
        cond_trained = (cond_trained | (cond_1_trained & cond_2_trained));
        
    count_2_untrained = len(infoList1[0][0,cond_untrained]);    
    count_2_trained = len(infoList1[1][0,cond_trained]);    
            
            
    countSelectiveCells1=np.zeros((4,));
    countSelectiveCells1[0]=len(infoList1[0][0,np.max(infoList1[1],axis=0)>infoThreshold1])-count_1_untrained;
    countSelectiveCells1[1]=len(infoList2[0][0,np.max(infoList2[1],axis=0)>infoThreshold2])-count_1_untrained;
    countSelectiveCells1[2]=count_1_untrained-count_2_untrained;
    countSelectiveCells1[3]=count_2_untrained;
    
    
    
    print("** results of untrained network **")    
    print("number of cells carry info>"+ str(infoThreshold1)+" about at least one Visual Input category (untrained): "+str(countSelectiveCells1[0]));
    print("number of cells carry info>"+ str(infoThreshold2)+" about at least one Audio Input category (untrained): "+str(countSelectiveCells1[1]));
    print("number of cells carry info>"+ str(infoThreshold1)+" about at least one V and info>"+  str(infoThreshold2)+ " about at least one A Input categories (can be inconsistent) (untrained): " + str(countSelectiveCells1[2]));
    print("number of cells carry info>"+ str(infoThreshold1) + " about at least one V and info>"+ str(infoThreshold2)+" about at least one A Input stimulus (consistent) (untrained): " + str(countSelectiveCells1[3]));
    print("["+str(countSelectiveCells1[0]) + "," + str(countSelectiveCells1[1]) + "," + str(countSelectiveCells1[2]) + "," +   str(countSelectiveCells1[3]) + "]");
    
    countSelectiveCells2=np.zeros((4,))
    countSelectiveCells2[0]=len(infoList1[1][0,np.max(infoList1[1],axis=0)>infoThreshold1])-count_1_trained;
    countSelectiveCells2[1]=len(infoList2[1][0,np.max(infoList2[1],axis=0)>infoThreshold2])-count_1_trained;
    countSelectiveCells2[2]=count_1_trained-count_2_trained;
    countSelectiveCells2[3]=count_2_trained;
    print("** results of trained network **")
    print("number of cells carry info>"+ str(infoThreshold1)+" about at least one Visual Input category (trained): "+str(countSelectiveCells2[0]));
    print("number of cells carry info>"+ str(infoThreshold2)+" about at least one Audio Input category (trained): "+str(countSelectiveCells2[1]));
    print("number of cells carry info>"+ str(infoThreshold1)+" about at least one V and info>"+  str(infoThreshold2)+ " about at least one A Input categories (can be inconsistent) (trained): " + str(countSelectiveCells2[2]));
    print("number of cells carry info>"+ str(infoThreshold1) + " about at least one V and info>"+ str(infoThreshold2)+" about at least one A Input stimulus (consistent) (trained): " + str(countSelectiveCells2[3]));
    print("["+str(countSelectiveCells2[0]) + "," + str(countSelectiveCells2[1]) + "," + str(countSelectiveCells2[2]) + "," +   str(countSelectiveCells2[3]) + "]");
    
    

    
    if plotOn:
        plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
        plt.subplots_adjust(wspace=0.4, hspace=1.0)
        resultNorm=results-np.min(results);
        resultNorm=resultNorm/np.max(resultNorm)

        
        import matplotlib.gridspec as gridspec
        maxSubplot = 10;
        plotIndex = 0;
        gs = gridspec.GridSpec(maxSubplot, 5);
        
        for s in range(nObj):
            # objIndex = 0;
        
            cond_1 = (infoList1[1][s]>infoThreshold1)
            cond_2 = (infoList2[1][s]>infoThreshold2)
            pts=np.argwhere(cond_1 & cond_2);    
            plt.gray()
            
            for p in range(len(pts)):
                frTable = resultNorm[:,:,pts[p,0],pts[p,1],pts[p,2]];
                frTableNorm = frTable-np.min(frTable);
                frTableNorm = frTableNorm/np.max(frTableNorm);
            
                plt.subplot(gs[plotIndex,:2])
                plt.imshow(1-frTableNorm[:,:50], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
                plt.title("Obj " + str(s) + ": V - " + "{:10.3f}".format(infoList1[1][s,pts[p,0],pts[p,1],pts[p,2]])  + " bit; cell:"+str(pts[p]));
                
                plt.subplot(gs[plotIndex,2:4])
                plt.imshow(1-frTableNorm[:,50:100], interpolation='nearest',aspect='auto',vmin=0, vmax=1);
#                 plt.imshow(frTableNorm, interpolation='nearest',aspect='auto',vmin=0, vmax=1);
                plt.title("Obj " + str(s) + ": A - " + "{:10.3f}".format(infoList2[1][s,pts[p,0],pts[p,1],pts[p,2]])  + " bit; cell:"+str(pts[p]));
                # print(frTable)
                
                plt.subplot(gs[plotIndex,4])
                plt.barh(range(nObj),np.sum(frTable,axis=1),height=0.8,color='k')
                plt.xlim((0,nTrans))
                plt.gca().invert_yaxis()
                plt.margins(y=0)
                cur_axes = plt.gca()
            #         cur_axes.axes.get_yaxis().set_ticklabels([])
            #         plt.axis('off')
                
                plt.suptitle("plot cells where info about both A and V are relatively high")
                plotIndex+=1;
                if plotIndex>=maxSubplot:
                    plt.subplots_adjust(wspace=0.4, hspace=0.6)
                    plt.show()
#                     plt.clf()
#                     plt.figure.figsize=(18, 16);
#                     plt.figure.dpi = 70;
                    
                    plotIndex=0;
                    plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
                    
        plt.show();
        
    return countSelectiveCells2;   




        
def runPCA(results):
    plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    N = 3
    nObj = 10;
    nTrans = 50;

#     comp1=0;
#     comp2=1;
    
#     comp1 = [0,0,1]
#     comp2 = [1,2,2]
    comp1 = [1]
    comp2 = [2]
    
#     col_tab10 = plt.cm.get_cmap('tab10').colors;
    
    for i in range(len(comp1)):
    
        ## V+A
        pca = PCA(n_components=N)
    #     trans = pca.fit(results)
        trans = pca.fit_transform(results)
        
        ax=plt.subplot(len(comp1),3,1+len(comp1)*i);
    #     'b', 'g', 'r', 'c', 'm', 'y', 'k', 
        r,g,b = (0,0,0)
        plt.plot(trans[50*0:50*(0+1),comp1[i]], trans[50*0:50*(0+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-0')
        plt.plot(trans[50*10:50*(10+1),comp1[i]], trans[50*10:50*(10+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-0')
        
        r,g,b = (0,0.9,0)
        plt.plot(trans[50*1:50*(1+1),comp1[i]], trans[50*1:50*(1+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-1')
        plt.plot(trans[50*11:50*(11+1),comp1[i]], trans[50*11:50*(11+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-1')
        
        r,g,b = (1,0,0)
        plt.plot(trans[50*2:50*(2+1),comp1[i]], trans[50*2:50*(2+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-2')
        plt.plot(trans[50*12:50*(12+1),comp1[i]], trans[50*12:50*(12+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-2')
        
        r,g,b = (0,0.9,0.9)
        plt.plot(trans[50*3:50*(3+1),comp1[i]], trans[50*3:50*(3+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-3')
        plt.plot(trans[50*13:50*(13+1),comp1[i]], trans[50*13:50*(13+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-3')
        
        r,g,b = (0,0,1)
        plt.plot(trans[50*4:50*(4+1),comp1[i]], trans[50*4:50*(4+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-4')
        plt.plot(trans[50*14:50*(14+1),comp1[i]], trans[50*14:50*(14+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-4')
        
        r,g,b = (1,0,1)
        plt.plot(trans[50*5:50*(5+1),comp1[i]], trans[50*5:50*(5+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-5')
        plt.plot(trans[50*15:50*(15+1),comp1[i]], trans[50*15:50*(15+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-5')
        
        r,g,b = (0.9,0.9,0)
        plt.plot(trans[50*6:50*(6+1),comp1[i]], trans[50*6:50*(6+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-6')
        plt.plot(trans[50*16:50*(16+1),comp1[i]], trans[50*16:50*(16+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-6')
        
        r,g,b = (0.5,0,0)
        plt.plot(trans[50*7:50*(7+1),comp1[i]], trans[50*7:50*(7+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-7')
        plt.plot(trans[50*17:50*(17+1),comp1[i]], trans[50*17:50*(17+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-7')
       
        r,g,b = (0.5,0,1)
        plt.plot(trans[50*8:50*(8+1),comp1[i]], trans[50*8:50*(8+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-8')
        plt.plot(trans[50*18:50*(18+1),comp1[i]], trans[50*18:50*(18+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-8')
    
        r,g,b = (0.5,0.5,0.5)
        plt.plot(trans[50*9:50*(9+1),comp1[i]], trans[50*9:50*(9+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-9')
        plt.plot(trans[50*19:50*(19+1),comp1[i]], trans[50*19:50*(19+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-9')
        
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.xlabel('Component '+str(comp1[i]))
        plt.ylabel('Component '+str(comp2[i]))
#         plt.tick_params(labelbottom='off',labelleft='off')
    #     plt.xlim([-4,4])
    #     plt.ylim([-4,4])
#         plt.legend()
        plt.title('V+A')
        
        
        
        ## V only
        pca = PCA(n_components=N)
    #     trans = pca.fit(results)
        trans = pca.fit_transform(results[:nObj*nTrans]);
    
        ax=plt.subplot(len(comp1),3,2+len(comp1)*i);
    #     'b', 'g', 'r', 'c', 'm', 'y', 'k', 
        r,g,b = (0,0,0)
        plt.plot(trans[50*0:50*(0+1),comp1[i]], trans[50*0:50*(0+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-0')
        
        r,g,b = (0,0.9,0)
        plt.plot(trans[50*1:50*(1+1),comp1[i]], trans[50*1:50*(1+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-1')
        
        r,g,b = (1,0,0)
        plt.plot(trans[50*2:50*(2+1),comp1[i]], trans[50*2:50*(2+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-2')
        
        r,g,b = (0,0.9,0.9)
        plt.plot(trans[50*3:50*(3+1),comp1[i]], trans[50*3:50*(3+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-3')
        
        r,g,b = (0,0,1)
        plt.plot(trans[50*4:50*(4+1),comp1[i]], trans[50*4:50*(4+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-4')
        
        r,g,b = (1,0,1)
        plt.plot(trans[50*5:50*(5+1),comp1[i]], trans[50*5:50*(5+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-5')
        
        r,g,b = (0.9,0.9,0)
        plt.plot(trans[50*6:50*(6+1),comp1[i]], trans[50*6:50*(6+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-6')
        
        r,g,b = (0.5,0,0)
        plt.plot(trans[50*7:50*(7+1),comp1[i]], trans[50*7:50*(7+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-7')
       
        r,g,b = (0.5,0,1)
        plt.plot(trans[50*8:50*(8+1),comp1[i]], trans[50*8:50*(8+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-8')
    
        r,g,b = (0.5,0.5,0.5)
        plt.plot(trans[50*9:50*(9+1),comp1[i]], trans[50*9:50*(9+1),comp2[i]], '+', markersize=5, color=(r, g, b, 1), mfc='none', label='v-9')
        
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.xlabel('Component '+str(comp1[i]))
        plt.ylabel('Component '+str(comp2[i]))
#         plt.tick_params(labelbottom='off',labelleft='off')
    #     plt.xlim([-4,4])
    #     plt.ylim([-4,4])
        plt.legend()
        plt.title('V only')
        
        
        
        
        ## A only
        pca = PCA(n_components=N)
    #     trans = pca.fit(results)
        trans = pca.fit_transform(results[nObj*nTrans:nObj*nTrans*2]);
    
        ax=plt.subplot(len(comp1),3,3+len(comp1)*i);
    #     'b', 'g', 'r', 'c', 'm', 'y', 'k', 
        r,g,b = (0,0,0)
        plt.plot(trans[50*0:50*(0+1),comp1[i]], trans[50*0:50*(0+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-0')
        
        r,g,b = (0,0.9,0)
        plt.plot(trans[50*1:50*(1+1),comp1[i]], trans[50*1:50*(1+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-1')
        
        r,g,b = (1,0,0)
        plt.plot(trans[50*2:50*(2+1),comp1[i]], trans[50*2:50*(2+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-2')
        
        r,g,b = (0,0.9,0.9)
        plt.plot(trans[50*3:50*(3+1),comp1[i]], trans[50*3:50*(3+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-3')
        
        r,g,b = (0,0,1)
        plt.plot(trans[50*4:50*(4+1),comp1[i]], trans[50*4:50*(4+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-4')
        
        r,g,b = (1,0,1)
        plt.plot(trans[50*5:50*(5+1),comp1[i]], trans[50*5:50*(5+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-5')
        
        r,g,b = (0.9,0.9,0)
        plt.plot(trans[50*6:50*(6+1),comp1[i]], trans[50*6:50*(6+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-6')
        
        r,g,b = (0.5,0,0)
        plt.plot(trans[50*7:50*(7+1),comp1[i]], trans[50*7:50*(7+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-7')
       
        r,g,b = (0.5,0,1)
        plt.plot(trans[50*8:50*(8+1),comp1[i]], trans[50*8:50*(8+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-8')
    
        r,g,b = (0.5,0.5,0.5)
        plt.plot(trans[50*9:50*(9+1),comp1[i]], trans[50*9:50*(9+1),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none', label='a-9')
        
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt.xlabel('Component '+str(comp1[i]))
        plt.ylabel('Component '+str(comp2[i]))
#         plt.tick_params(labelbottom='off',labelleft='off')
    #     plt.xlim([-4,4])
    #     plt.ylim([-4,4])
        plt.legend()
        plt.title('A only')
    
    
    plt.show();
    
    
    
    
    
def runPCAAboutUnits(results_untrained,results_trained,infV,infA):
    infoThreshold = 1.0;
    N = 3
    nObj = 10;
    nTrans = 50;
    comp1=[0];
    comp2=[1];
    i = 0;
    
    shape = np.shape(infV);
    
    
    pca = PCA(n_components=N)
    trans_untrained = pca.fit_transform(results_untrained)
    trans_trained = pca.fit_transform(results_trained)

    
    infV_max_untrained = np.reshape(np.max(infV[0],axis=0),(shape[2],));
    infA_max_untrained = np.reshape(np.max(infA[0],axis=0),(shape[2],));
    infV_max_untrained/=np.log2(10);
    infA_max_untrained/=np.log2(10);
    
    infV_max_trained = np.reshape(np.max(infV[1],axis=0),(shape[2],));
    infA_max_trained = np.reshape(np.max(infA[1],axis=0),(shape[2],));
    infV_max_trained/=np.log2(10);
    infA_max_trained/=np.log2(10);


    
    plt.subplot(1,2,1);
    for c in range(shape[2]):
#         r,g,b = (1 if infV_max[c]>infoThreshold else 0,1 if infA_max[c]>infoThreshold else 0,0)
        r = infV_max_untrained[c];
        g = infA_max_untrained[c];
        b = 0;
                
        plt.plot(trans_untrained[c,comp1[i]],trans_untrained[c,comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none')

    plt.xlabel('Component '+str(comp1[i]))
    plt.ylabel('Component '+str(comp2[i]))
    plt.title('Untrained Network')


    plt.subplot(1,2,2);
    for c in range(shape[2]):
#         r,g,b = (1 if infV_max[c]>infoThreshold else 0,1 if infA_max[c]>infoThreshold else 0,0)
        r = infV_max_trained[c];
        g = infA_max_trained[c];
        b = 0;
        
        plt.plot(trans_trained[c,comp1[i]],trans_trained[c,comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none')

        
 
        
    plt.xlabel('Component '+str(comp1[i]))
    plt.ylabel('Component '+str(comp2[i]))
    plt.legend()
    plt.title('Trained Network: R:V, G:A, Y:V+A')
    
    plt.show();
    

    
#     N = 3
#     nObj= 10;
#     nTrans = 50;
#     pca = PCA(n_components=N)
# #     trans = pca.fit(results)
#     nCombVar = 3;
# 
#     shape = np.shape(results);
#     
#     nCells = shape[0] 
# 
# #     resultRev= np.zeros((nCells*3,nObj*nTrans));
# #     for v in range(nCombVar):
# #         resultRev[(v*nCells):(v+1)*nCells]=results[:,v*nObj*nTrans:(v+1)*nObj*nTrans]
#     
# 
#     trans = pca.fit_transform(result)
#     comp1=0;
#     comp2=1;
#     
#     infV_max = np.reshape(np.max(infV[1],axis=0),(shape[2],));
#     infA_max = np.reshape(np.max(infA[1],axis=0),(shape[2],));
#  
#      
# 
#     r,g,b = (1,0,0)
#     plt.plot(trans[0*(nCells):(0+1)*(nCells),comp1[i]],trans[0*(nCells):(0+1)*(nCells),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none')
#         
#     r,g,b = (0,1,0)
#     plt.plot(trans[1*(nCells):(1+1)*(nCells),comp1[i]],trans[1*(nCells):(1+1)*(nCells),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none')
# 
#     r,g,b = (1,1,0)
#     plt.plot(trans[2*(nCells):(2+1)*(nCells),comp1[i]],trans[2*(nCells):(2+1)*(nCells),comp2[i]], 'o', markersize=5, color=(r, g, b, 1), mfc='none')
# 
#     plt.xlabel('Component '+str(comp1))
#     plt.ylabel('Component '+str(comp2))
# #     plt.xlim([-4,4])
# #     plt.ylim([-4,4])
#     plt.legend()
#     plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
#     
#     plt.show();
    


def mutualInfo(S,R,nBins=5):
    nStim = np.shape(S)[0];
    nUnitS=np.shape(S)[1];
    nUnitR=np.shape(R)[1];

    I = np.zeros((nUnitS,nUnitR));
    
    ## create bin matrix
    PsrTable = np.zeros((nUnitS,nUnitR));
    
    
    ## normalise
    S = S-np.min(S);
    R = R-np.min(R);
    
    S = S/np.max(S);
    R = R/np.max(R);

    for s in range(nUnitS): # for each input pixel
#         print(str(s)+"/"+str(nUnitS));
        for r in range(nUnitR): # for cell in a target layer     
            s_cond_list =[]
            r_cond_list =[]

            Psr=np.zeros((nBins,nBins));
            Ps = np.zeros((nBins,));
            Pr = np.zeros((nBins,));
            
            ## binning the values to create Ps and Pr
            for b in range(nBins):
                if b==0:
                    s_cond =  S[:,s] <= (b+1)*(1/nBins);
                    r_cond = R[:,r] <= (b+1)*(1/nBins);
                elif b==nBins-1:
                    s_cond =  S[:,s] > b*(1/nBins);
                    r_cond = R[:,r] > b*(1/nBins);
                else:
                    s_cond =  (S[:,s] > (b)*(1/nBins))  & (S[:,s] <=(b+1)*(1/nBins)); 
                    r_cond =  (R[:,r] > (b)*(1/nBins)) & (R[:,r] <=(b+1)*(1/nBins));  
                Ps[b] = np.count_nonzero(s_cond); 
                s_cond_list.append(s_cond);   
                    
                Pr[b] = np.count_nonzero(r_cond);
                r_cond_list.append(r_cond);
                
            Ps/=nStim;
            Pr/=nStim;

            ## create P(s,r)
            for b_s in range(nBins):
                for b_r in range(nBins):
                    s_cond = s_cond_list[b_s];
                    r_cond = r_cond_list[b_r];
                    Psr[b_s,b_r]= np.count_nonzero(s_cond&r_cond);
                
            Psr/=nStim;
            
            ## calculate information   
            for x in range(nBins):
                for y in range(nBins):
                    if (Psr[x,y]!=0 and Ps[x]*Pr[y]!=0 and Psr[x,y]-(Ps[x]*Pr[y])>0):
                        I[s,r]+=Psr[x,y]*np.log2(Psr[x,y]/(Ps[x]*Pr[y]));


#     print("** finished calculating mutual cell info ** ");    
    return I;
    
    
    
    
    
    