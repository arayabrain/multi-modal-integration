from params import *
from statsmodels.graphics.tukeyplot import results
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
        plt.subplots_adjust(wspace=0.4, hspace=1.5)
        plt.subplot(3,1,1)
        plt.plot(np.transpose(IRs_sorted_list[0]));
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.subplot(3,1,2)
        plt.plot(np.transpose(IRs_sorted_list[1])); 
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("trained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.subplot(3,1,3)   
        plt.plot(np.transpose([np.sort(np.max(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.max(IRs_flattened_list[1],axis=0)*-1)*-1]))
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained v trained network (max vals are taken)")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    
        # plt.subplot(2,1,2)
        # plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)
        
        plt.show()
    return (IRs_list,IRs_weighted_list);













def singleCellInfoAnalysis_avg(results_untrained,results_trained,plotOn=True):
    nBins = 2;
    nObj = np.shape(results_untrained)[0];
#     nTrans = np.shape(results_untrained)[1];
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
                        b = int(min(np.floor(np.mean(resultNorm[s,:,row,col,dep])*nBins),nBins-1))
#                             print("ori: "+ str(resultNorm[s,t,row,col,dep]) + "\nafter: "+ str(b))
                        binned[row,col,dep,s,b]=binned[row,col,dep,s,b]+1
                        
                        
        sumPerBin = np.zeros((nRow,nCol,nDep,nBins));
        sumPerObj = 1;
        sumPerCell = 1*nObj;
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
        plt.subplot(3,1,1)
        plt.plot(np.transpose(IRs_sorted_list[0]));
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.subplot(3,1,2)
        plt.plot(np.transpose(IRs_sorted_list[1])); 
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("trained network")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
        plt.subplot(3,1,3)   
        plt.plot(np.transpose([np.sort(np.max(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.max(IRs_flattened_list[1],axis=0)*-1)*-1]))
        plt.ylabel("single cell info [bit]")
        plt.xlabel("cell rank")
        plt.title("untrained v trained network (max vals are taken)")
        plt.ylim((np.log2(nObj)*-0.05,np.log2(nObj)*1.05))
    
        # plt.subplot(2,1,2)
        # plt.plot(np.sort(np.max(IRs_flattened_untrained,axis=0)*-1)*-1)
        
        plt.show()
    return (IRs_list,IRs_weighted_list);



def countCellsWithSelectivity(infoList1, infoList2, results, plotOn=True,infoThreshold = 0.5):
    ## count number of cells developed to be selective to a stimulus    
    nObj = np.shape(results)[0];
    nTrans = np.shape(results)[1];
    
    
    indexShape = np.shape(infoList2[0][0]);
       
    #counting a number of cells that is selective to any of the visual stimulus and to any of the audio stimulus         
    cond_1_untrained = np.zeros(indexShape,dtype=bool) 
    cond_1_trained = np.zeros(indexShape,dtype=bool) 
    cond_2_untrained = np.zeros(indexShape,dtype=bool)
    cond_2_trained = np.zeros(indexShape,dtype=bool) 

            
    count_1_trained = 0;
    count_1_untrained = 0;
    
    for s in range(nObj):
        cond_1_untrained = (cond_1_untrained|(infoList2[0][s]>infoThreshold))
        cond_1_trained = (cond_1_trained| (infoList2[1][s]>infoThreshold))
        
        cond_2_untrained = (cond_2_untrained | (infoList1[0][s]>infoThreshold))
        cond_2_trained = (cond_2_trained | (infoList1[1][s]>infoThreshold))
        
        
    count_1_untrained = len(infoList2[0][0,cond_1_untrained & cond_2_untrained]);
    count_1_trained = len(infoList2[1][0,cond_1_trained & cond_2_trained]);
        


    #counting a number of cells that is selective to at least one consistent stimulus
    cond_trained = np.zeros(indexShape,dtype=bool)
    cond_untrained = np.zeros(indexShape,dtype=bool)

    count_2_trained = 0;
    count_2_untrained = 0;
    for s in range(nObj):
        cond_1_untrained = (infoList1[0][s]>infoThreshold)
        cond_1_trained = (infoList1[1][s]>infoThreshold)
        
        cond_2_untrained = (infoList2[0][s]>infoThreshold)
        cond_2_trained = (infoList2[1][s]>infoThreshold)
        
        cond_untrained= (cond_untrained | (cond_1_untrained & cond_2_untrained));
        cond_trained = (cond_trained | (cond_1_trained & cond_2_trained));
        
    count_2_untrained = len(infoList1[0][0,cond_untrained]);    
    count_2_trained = len(infoList1[1][0,cond_trained]);    
            
    print("** results of untrained network **")    
    print("number of cells carry info>0.5 about at least one Visual Input category (untrained): "+str(len(infoList1[0][0,np.max(infoList1[0],axis=0)>infoThreshold])));
    print("number of cells carry info>0.5 about at least one Audio Input category (untrained): "+str(len(infoList2[0][0,np.max(infoList2[0],axis=0)>infoThreshold])));
    print("number of cells carry info>0.5 about at least one V and one A Input categories (can be inconsistent) (untrained): " + str(count_1_untrained));
    print("number of cells carry info>0.5 about at least one consistent V and A Input stimulus (untrained): " + str(count_2_untrained));

    print("** results of trained network **")
    print("number of cells carry info>0.5 about at least one Visual Input category (untrained): "+str(len(infoList1[1][0,np.max(infoList1[1],axis=0)>infoThreshold])));
    print("number of cells carry info>0.5 about at least one Audio Input category (untrained): "+str(len(infoList2[1][0,np.max(infoList2[1],axis=0)>infoThreshold])));
    print("number of cells carry info>0.5 about at least one V and one A Input categories (can be inconsistent) (untrained): " + str(count_1_trained));
    print("number of cells carry info>0.5 about at least one consistent V and A Input stimulus (untrained): " + str(count_2_trained));
    
    if plotOn:
        fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        resultNorm=results-np.min(results);
        resultNorm=resultNorm/np.max(resultNorm)

        
        import matplotlib.gridspec as gridspec
        maxSubplot = 10;
        plotIndex = 0;
        gs = gridspec.GridSpec(maxSubplot, 5);
        
        for s in range(nObj):
            # objIndex = 0;
        
            cond_1 = (infoList1[1][s]>infoThreshold)
            cond_2 = (infoList2[1][s]>infoThreshold)
            pts=np.argwhere(cond_1 & cond_2);    
        
            for p in range(len(pts)):
                frTable = resultNorm[:,:,pts[p,0],pts[p,1],pts[p,2]];
                frTableNorm = frTable-np.min(frTable);
                frTableNorm = frTableNorm/np.max(frTableNorm);
            
                plt.subplot(gs[plotIndex,:4])
                plt.imshow(1-frTableNorm, interpolation='nearest',aspect='auto',vmin=0, vmax=1);
                plt.title("Obj " + str(s) + ": V - " + "{:10.3f}".format(infoList1[1][s,pts[p,0],pts[p,1],pts[p,2]]) + "; A - " + "{:10.3f}".format(infoList2[1][s,pts[p,0],pts[p,1],pts[p,2]])  + " bit; cell:"+str(pts[p]));
                # print(frTable)
                plt.subplot(gs[plotIndex,4])
                plt.barh(range(nObj),np.sum(frTable,axis=1),height=0.8)
                plt.xlim((0,nTrans))
                plt.gca().invert_yaxis()
                plt.margins(y=0)
                cur_axes = plt.gca()
            #         cur_axes.axes.get_yaxis().set_ticklabels([])
            #         plt.axis('off')
                plt.gray()
                plt.suptitle("plot cells where info about both A and V are relatively high")
                plotIndex+=1;
                if plotIndex>=maxSubplot:
                    plt.show()
                    plt.clf()
                    fig=plt.figure(figsize=(18, 16), dpi= 70, facecolor='w', edgecolor='k')
#                     plt.figure.figsize=(18, 16);
#                     plt.figure.dpi = 70;
                    plt.subplots_adjust(wspace=0.4, hspace=0.6)
                    plotIndex=0;
        plt.show();