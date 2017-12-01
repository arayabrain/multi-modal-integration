from params import *
from sklearn.decomposition import PCA
from blaze.expr.expressions import label

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
        plt.plot(np.transpose([np.sort(np.mean(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.max(IRs_flattened_list[1],axis=0)*-1)*-1]));
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
        plt.plot(np.transpose([np.sort(np.mean(IRs_flattened_list[0],axis=0)*-1)*-1,np.sort(np.mean(IRs_flattened_list[1],axis=0)*-1)*-1]))
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
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one Visual Input category (untrained): "+str(len(infoList1[0][0,np.max(infoList1[0],axis=0)>infoThreshold])));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one Audio Input category (untrained): "+str(len(infoList2[0][0,np.max(infoList2[0],axis=0)>infoThreshold])));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one V and one A Input categories (can be inconsistent) (untrained): " + str(count_1_untrained));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one consistent V and A Input stimulus (untrained): " + str(count_2_untrained));

    print("** results of trained network **")
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one Visual Input category (trained): "+str(len(infoList1[1][0,np.max(infoList1[1],axis=0)>infoThreshold])));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one Audio Input category (trained): "+str(len(infoList2[1][0,np.max(infoList2[1],axis=0)>infoThreshold])));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one V and one A Input categories (can be inconsistent) (trained): " + str(count_1_trained));
    print("number of cells carry info>"+ str(infoThreshold)+" about at least one consistent V and A Input stimulus (trained): " + str(count_2_trained));
    print("["+str(len(infoList1[1][0,np.max(infoList1[1],axis=0)>infoThreshold])) + "," + str(len(infoList2[1][0,np.max(infoList2[1],axis=0)>infoThreshold])) + "," + str(count_1_trained) + "," +   str(count_2_trained) + "]");
    
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
        
        
        
def runPCA(results):
    
    N = 3
    nObj = 10;
    nTrans = 50;
    pca = PCA(n_components=N)
#     trans = pca.fit(results)
    trans = pca.fit_transform(results)
    comp1=0;
    comp2=1;
    
#     'b', 'g', 'r', 'c', 'm', 'y', 'k', 
    r,g,b = (0,0,0)
    plt.plot(trans[50*0:50*(0+1),comp1], trans[50*0:50*(0+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-0')
    plt.plot(trans[50*10:50*(10+1),comp1], trans[50*10:50*(10+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-0')
    
    r,g,b = (0,0,1)
    plt.plot(trans[50*1:50*(1+1),comp1], trans[50*1:50*(1+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-1')
    plt.plot(trans[50*11:50*(11+1),comp1], trans[50*11:50*(11+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-1')
    
    r,g,b = (0,1,0)
    plt.plot(trans[50*2:50*(2+1),comp1], trans[50*2:50*(2+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-2')
    plt.plot(trans[50*12:50*(12+1),comp1], trans[50*12:50*(12+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-2')
    
    r,g,b = (0,1,1)
    plt.plot(trans[50*3:50*(3+1),comp1], trans[50*3:50*(3+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-3')
    plt.plot(trans[50*13:50*(13+1),comp1], trans[50*13:50*(13+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-3')
    
    r,g,b = (1,0,0)
    plt.plot(trans[50*4:50*(4+1),comp1], trans[50*4:50*(4+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-4')
    plt.plot(trans[50*14:50*(14+1),comp1], trans[50*14:50*(14+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-4')
    
    r,g,b = (1,0,1)
    plt.plot(trans[50*5:50*(5+1),comp1], trans[50*5:50*(5+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-5')
    plt.plot(trans[50*15:50*(15+1),comp1], trans[50*15:50*(15+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-5')
    
    r,g,b = (1,1,0)
    plt.plot(trans[50*6:50*(6+1),comp1], trans[50*6:50*(6+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-6')
    plt.plot(trans[50*16:50*(16+1),comp1], trans[50*16:50*(16+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-6')
    
    r,g,b = (0.5,0,0)
    plt.plot(trans[50*7:50*(7+1),comp1], trans[50*7:50*(7+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-7')
    plt.plot(trans[50*17:50*(17+1),comp1], trans[50*17:50*(17+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-7')
   
    r,g,b = (0.5,0,1)
    plt.plot(trans[50*8:50*(8+1),comp1], trans[50*8:50*(8+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-8')
    plt.plot(trans[50*18:50*(18+1),comp1], trans[50*18:50*(18+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-8')

    r,g,b = (0.5,0.5,0.5)
    plt.plot(trans[50*9:50*(9+1),comp1], trans[50*9:50*(9+1),comp2], '*', markersize=7, color=(r, g, b, 1), alpha=0.5, label='v-9')
    plt.plot(trans[50*19:50*(19+1),comp1], trans[50*19:50*(19+1),comp2], 'o', markersize=7, color=(r, g, b, 1), alpha=0.5, label='a-9')
    

    plt.xlabel('x_values')
    plt.ylabel('y_values')
#     plt.xlim([-4,4])
#     plt.ylim([-4,4])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
    
    plt.show();



def mutualInfo(S,R):
    # assuming the responses are either over the mean value or under.
    
    nStim = np.shape(S)[0];
    nUnitS=np.shape(S)[1];
    nUnitR=np.shape(R)[1];

    I = np.zeros((nUnitS,nUnitR));
    
    ## create bin matrix
    PsrTable = np.zeros((nUnitS,nUnitR));
    s_mean = np.mean(S);
    r_mean = np.mean(R);

    for s in range(nUnitS):
        for r in range(nUnitR):
            s_cond = S[:,s]>s_mean;
            r_cond = R[:,r]>r_mean;
            Psr=np.zeros((2,2));
            Ps = np.zeros((2,));
            Pr = np.zeros((2,));
            
            
            Ps[1] = np.count_nonzero(s_cond);
            Ps[0] = nStim - Ps[1];
            Pr[1] = np.count_nonzero(r_cond);
            Pr[0] = nStim - Pr[1];
            Psr[1,1]= np.count_nonzero(s_cond&r_cond);
            Psr[0,1] = Pr[1]-Psr[1,1];
            Psr[1,0] = Ps[1]-Psr[1,1];
            Psr[0,0]= nStim-Psr[0,1]-Psr[1,0]-Psr[1,1];
 
            Psr/=nStim;
            Ps/=nStim;
            Pr/=nStim;
            
            for x in range(2):
                for y in range(2):
#                     if (Psr[x,y]-(Ps[x]*Pr[y])<0):
#                         print("Psr["+str(x)+","+str(y) + "]=" + str(Psr[x,y]))
#                         print("Ps["+str(x)+"]*Pr["+str(y)+"]="+str(Ps[x])+"*"+str(Pr[y]))
                    if (Psr[x,y]!=0 and Ps[x]*Pr[y]!=0 and Psr[x,y]-(Ps[x]*Pr[y])>0):
                        I[s,r]+=Psr[x,y]*np.log2(Psr[x,y]/(Ps[x]*Pr[y]));


    print("** finished calculating mutual cell info ** ");    
    return I;
            
            
            
    