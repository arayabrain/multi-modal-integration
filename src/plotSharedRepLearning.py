# import pylab as plt
# 
# v2a_consist = [0.898, 0.896]
# v2a_inconsist = [0.116, 0.676]
# 
# a2v_consist = [0.066, 0.748]
# a2v_inconsist = [0.552, 0.592]
# 
# labels = ["Inconsistent","Consistent"]
# 
# plt.subplot(1,2,1)
# plt.plot(labels,v2a_consist, '--',label="Test with visual inputs")
# plt.plot(labels,v2a_inconsist, label="Test with audio inputs")
# plt.title("Shared Rep. Learning with Visual Input")
# plt.ylim([0,1])
# plt.gca().invert_xaxis()
# # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
# 
# plt.subplot(1,2,2)
# plt.plot(labels,a2v_consist, '--',label="Test with visual inputs")
# plt.plot(labels,a2v_inconsist, label="Test with audio inputs")
# plt.title("Share Representation Learning with Audio Input")
# plt.ylim([0,1])
# plt.gca().invert_xaxis()
# 
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
# plt.subplots_adjust(hspace=1.0)
# 
# 
# plt.show()


import pylab as plt
import numpy as np

v2a_consist = np.array([[0.925999999046, 0.16600000003],
 [0.934000000954, 0.27200000003],
 [0.924000000954, 0.286],
 [0.932000000954, 0.346]])*100

v2a_inconsist = np.array([[0.922000000954, 0.092],
 [0.937999999046, 0.074],
 [0.929999999046, 0.126],
 [0.932000000954, 0.1000]])*100



a2v_consist = np.array([[0.278, 0.736],
 [0.48000000003,0.640000000477],
 [0.56000000003,0.604],
 [0.44200000003,0.592]])*100

a2v_inconsist = np.array([[0.068,0.714000000477],
 [0.044,0.667999999046],
 [0.0780000000596,0.644000000477],
 [0.072,0.628000000954]])*100



labels = ["Inconsistent","Consistent"]
layers = ['1 Layer','2 Layers','3 Layers','4 Layers']
plt.suptitle("Shared Rep. Learning");


plt.subplot(2,2,1)
plt.plot(layers, np.transpose(v2a_inconsist)[0], '--', label="Test with visual inputs")
plt.plot(layers,np.transpose(v2a_inconsist)[1], label="Test with audio inputs")
plt.title("(a1) Visual Training (Inconsistent Inputs)");
plt.ylim([0,100])
plt.xlabel("Number of Layers")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");

plt.subplot(2,2,2)
plt.plot(layers, np.transpose(v2a_consist)[0],'--', label="Test with visual inputs")
plt.plot(layers,np.transpose(v2a_consist)[1], label="Test with audio inputs")
plt.title("(a2) Visual Training (Consistent Inputs)");
plt.ylim([0,100])
plt.xlabel("Number of Layers")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
# plt.gca().invert_xaxis()


plt.subplot(2,2,3)
plt.plot(layers, np.transpose(a2v_inconsist)[0], '--', label="Test with visual inputs")
plt.plot(layers,np.transpose(a2v_inconsist)[1], label="Test with audio inputs")
plt.title("(b1) Audio Training (Inconsistent Inputs)");
plt.ylim([0,100])
plt.xlabel("Number of Layers")
# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");

plt.subplot(2,2,4)
plt.plot(layers, np.transpose(a2v_consist)[0],'--', label="Test with visual inputs")
plt.plot(layers,np.transpose(a2v_consist)[1], label="Test with audio inputs")
plt.title("(b2) Audio Training (Consistent Inputs)");
plt.ylim([0,100])
plt.xlabel("Number of Layers")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
# plt.gca().invert_xaxis()






# plt.subplot(2,1,2)
# plt.plot(labels,a2v_consist, label="Test with visual inputs")
# plt.plot(labels,a2v_inconsist, label="Test with audio inputs")
# plt.title("Share Representation Learning with Audio Input")
# plt.ylim([0,1])

# plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");
plt.subplots_adjust(hspace=2.0)
# plt.subplots_adjust(wspace=2.0)

plt.show()
