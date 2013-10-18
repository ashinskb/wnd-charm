# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:45:21 2013

@author: ashinskybg
"""

#!/usr/bin/env python
from scipy import stats
import numpy as np
#from pylab import plot,show
from scipy import seterr
#import random
import random as rand
import pylab
#from pylab import plot,show
import math
import matplotlib.pyplot as plt

import os

from pychrm.FeatureSet import *

class MultipleRegressionFeatureWeights( ContinuousFeatureWeights ):

		#================================================================
	@classmethod
	def NewFromFeatureSet( cls, training_set ):
		"""Calculate regression parameters and correlation statistics that fully define
		a continuous classifier.

		At present the feature weights are proportional the Pearson correlation coefficient
		for each given feature."""
		
		super( MultipleRegressionFeatureWeights, cls ).NewFromFeatureSet( training_set )
		
		#pseudocode:
		#iterate over all features:
		# get predicted values for each image
		# do regression of ground truth to 
		
		
def normalize_by_columns ( full_stack, mins = None, maxs = None ):
	"""This is a global function to normalize a matrix by columns.
	If numpy 1D arrays of mins and maxs are provided, the matrix will be normalized against these ranges
	Otherwise, the mins and maxs will be determined from the matrix, and the matrix will be normalized
	against itself. The mins and maxs will be returned as a tuple.
	Out of range matrix values will be clipped to min and max (including +/- INF)
	zero-range columns will be set to 0.
	NANs in the columns will be set to 0.
	The normalized output range is hard-coded to 0-100
	"""
# Edge cases to deal with:
#   Range determination:
#     1. features that are nan, inf, -inf
#        max and min determination must ignore invalid numbers
#        nan -> 0, inf -> max, -inf -> min
#   Normalization:
#     2. feature values outside of range
#        values clipped to range (-inf to min -> min, max to inf -> max) - leaves nan as nan
#     3. feature ranges that are 0 result in nan feature values
#     4. all nan feature values set to 0

# Turn off numpy warnings, since e're taking care of invalid values explicitly
	oldsettings = np.seterr(all='ignore')
	if (mins is None or maxs is None):
		# mask out NANs and +/-INFs to compute min/max
		full_stack_m = np.ma.masked_invalid (full_stack, copy=False)
		maxs = full_stack_m.max (axis=0)
		mins = full_stack_m.min (axis=0)

	# clip the values to the min-max range (NANs are left, but +/- INFs are taken care of)
	full_stack.clip (mins, maxs, full_stack)
	# remake a mask to account for NANs and divide-by-zero from max == min
	full_stack_m = np.ma.masked_invalid (full_stack, copy=False)

	# Normalize
	full_stack_m -= mins
	full_stack_m /= (maxs - mins)
	# Left over NANs and divide-by-zero from max == min become 0
	# Note the deep copy to change the numpy parameter in-place.
	full_stack[:] = full_stack_m.filled (0) * 100.0

	# return settings to original
	np.seterr(**oldsettings)

	return (mins,maxs)

def ypred_leave_one_out (train, test, percent) :
    
    #split to ground truth (y) and feature values (X)
    ytrain, Xtrain = np.hsplit(train, np.array([1]))
    ytest, Xtest = np.hsplit(test, np.array([1]))
    ytrain = np.concatenate(ytrain)
    ytest = np.concatenate(ytest)
    
    #normalize feature matrices
    (mins,maxs) = normalize_by_columns(Xtrain)
    
    #reduce features
    a = maxs-mins
    boolean_array = np.bool_(maxs-mins < 2.2204460492503131e-16)
    #print np.where(boolean_array)
    Xtrain = Xtrain.compress(np.logical_not(boolean_array), axis=1)
    training_n, reduced_feature_n = Xtrain.shape
    
    
    
    seterr(invalid='print',divide='print')
    idx = 0
    C = [(0), (0), (0)]
    E = np.zeros(training_n)
    for idx in range(0, reduced_feature_n):
        xi = Xtrain.T[idx]
        #print xi.shape, idx
        #print ytrain
        #print xi
        ytrain, xi
        slope, intercept, r_value, p_value, std_err = stats.linregress(xi,ytrain)
        B = [(slope), (intercept), (r_value)]
        #print 'r value', r_value
        #print 'slope', slope
        #print 'intercept', intercept
        C = np.c_[ C, B ]
        #plot(xi,line,'r-',xi,y,'o')
        #show()    
        count = 0
        D = [[(5)]]
        for count in range(0,training_n):
            y_pred = slope*xi[count]+intercept
            D = np.append(D, np.array(y_pred))
        D = np.delete(D,0,0)
        E = np.vstack([E, D])
        idx =+ 1
    C = np.delete(C,0,1)
    E = np.delete(E,0,0)
    E = E.T
    C[2] = np.square(C[2])
    
    #reduce features based on R
    new_feature_number = math.ceil((percent/100)*reduced_feature_n)   
    #new_feature_number = 60
    number_of_features_cut = reduced_feature_n - new_feature_number
    index = C[2].argsort()[:number_of_features_cut]
    C = np.delete(C, index, 1)    
    
    
    #normalize test matrices
    normalize_by_columns(Xtest, mins, maxs)
    Xtest = Xtest.compress(np.logical_not(boolean_array), axis=1)
    Xtest = np.delete(Xtest, index, 1)
    Xtrain = np.delete(Xtrain, index, 1)
    
    ##############################
    
    #weight the train and test matrices
    w = C[2]
    Xtrain = w*Xtrain
    Xtest = w*Xtest
    
    
    #add ones to feature matrix for least squares
    Xtrain = np.hstack([np.ones((training_n, 1)), Xtrain])
    Xtest = np.hstack([np.ones((test_n, 1)), Xtest])
    
  
    
    #least squares linear regression
    a = np.linalg.lstsq(Xtrain, ytrain)[0]
    ypred = np.array(np.dot(Xtest,a))
    ytest = np.array(ytest)
    return ypred, new_feature_number

   
def mult_leave_out (random, percent):
    i = 1
    J = [[(5)]]
    for row in random:
        test,train = np.vsplit(random, np.array([i]))
        if i == 1:
            y_all, new_feature_number = ypred_leave_one_out(train, test, percent)
        if i != 1:
            if i < sample_n:
                train2,test = np.vsplit(test, np.array([i-1])) 
                train = np.vstack((train2,train))
                y_all, new_feature_number= ypred_leave_one_out(train, test, percent)
            if i > training_n:
                train,test = np.vsplit(random, np.array([training_n]))
                y_all, new_feature_number = ypred_leave_one_out(train, test, percent)
    
        J = np.append(J, np.array(y_all))
        i = i + 1
    J = np.delete(J,0,0)
    ground_truth = random[:,0]
    
    #print J, ground_truth
    slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth,J)
    #print r_value
    #print np.square(r_value)
    return r_value, std_err, p_value, slope, intercept, ground_truth, J, new_feature_number


############################################
#only inputs here are your y ("ground truth" text file) and the pickled fit files of the fof generated from pychrm)
ground_truths = np.genfromtxt("/Users/ashinskybg/Desktop/Everything/Human_Dataset/Regression_Programs/y_ground_truth_values.txt",delimiter="\t").T

dir_w_fofs = '/Users/ashinskybg/Desktop/Everything/Human_Dataset/Everything_from_Vanessa/all_the_fofs'

intermediate_calculation_folder = 'intermediate_calculations' 
graph_destination_subfolder = dir_w_fofs + os.sep + 'Features_vs_Accuracy_Graphs'



# look in desired folder for any file named '.fof' and use it to make a feature set
all_files = [ f for f in os.listdir( dir_w_fofs ) if os.path.isfile( os.path.join( dir_w_fofs, f ) ) ]
all_fof_files = [ f for f in all_files if f.endswith( '.fof' ) ]

# This induces the debugger, insert this line anywhere you want to see what logic is doing
#import pdb; pdb.set_trace()

# Make directory for graphs to go, ignore error if directory already exists
try:
	os.makedirs( graph_destination_subfolder )
except OSError as exception:
	import errno
	if exception.errno != errno.EEXIST:
		raise

for fof_file_path in all_fof_files: 
    fof_path, fof_file = os.path.split( fof_file_path )
    new_fs = FeatureSet_Continuous.NewFromFileOfFiles( fof_file, '-l' )
    #new_fs = FeatureSet_Continuous.NewFromPickleFile(fit_file_path)
    print "read in", fof_file

    feature_matrix = new_fs.data_matrix
    #x1 = new_fs.data_matrix
    #x2 = new_fs2.data_matrix
    #x3 = new_fs3.data_matrix
    #x = np.hstack((x1,x2))
    #x = np.hstack((x,x3))
    
    sample_n, feature_n = feature_matrix.shape    # the number of observations
    # Randomly split to a training and test set
    together = np.insert( feature_matrix, 0, ground_truths, axis=1 )
    
    # But why do you even use random.sample here if you're still choosing all the samples?
    # Is it your goal to scramble the sample order? -Chris
    random = np.array( rand.sample( together, sample_n ) )
    # random = np.array(random.sample(together, sample_n))
    training_n = sample_n - 1
    test_n = sample_n - training_n # test_n will always be 1?
    # percent_values = np.linspace(1, 100, num=2)  # use more common range( 1, 101 )
    percent_values = range(1, 101, 20)
    accuracy_matrix = [[(5)]]
    
    
    for percent in percent_values:
        r_value, std_err, p_value, slope, intercept, ground_truth, J, new_feature_number = mult_leave_out(random, percent)
        foo = np.square(J-ground_truth)
        sumxy = sum(J*ground_truth)
        sumx = sum(ground_truth)
        sumy = sum(J)
        sumxsq = sum(ground_truth*ground_truth)
        sumysq = sum(J*J)
        rtop = ((sample_n*sumxy)-(sumx*sumy))
        rbottom = np.sqrt(((sample_n*sumxsq)-(sumx*sumx))*((sample_n*sumysq)-(sumy*sumy)))
        r = rtop/rbottom
        y = J
        ymean = np.mean(y)
        x = ground_truth
        xmean = np.mean(x)
        ydiff = y-ymean
        xdiff = x-xmean
        ssy = sum(ydiff*ydiff)
        ssx = sum(xdiff*xdiff)
        sterrest = np.sqrt(((1-r*r)*ssy)/(36*ssx))
        print J, ground_truth
        
        print 'percent', percent, 'feature number', new_feature_number, 'r', r_value, 'std_err', std_err, 'p_value', p_value, 'RMS', np.sqrt((sum(foo))/20)
    #uncomment below to graph observed vs predicted for each feature number:
        #line = slope*ground_truth+intercept
        #plot(ground_truth,line,'r-',ground_truth,J,'o')
        #pylab.xlim(0.5, 5)
        #pylab.ylim(0.5, 5)
        #show()    
        points = zip(ground_truth, J)
        print points
        pred_cutoff = slope*2.5+intercept
        print pred_cutoff
        true_positive = [(0,0)]
        true_negative = [(0,0)]
        false_positive = [(0,0)]
        false_negative = [(0,0)]
        for point in points:
            #print point
            if (point[0] >= 2.5 and point[1] >= pred_cutoff):
               true_positive = np.append(true_positive, point) 
            if (point[0] < 2.5 and point[1] < pred_cutoff):
                true_negative = np.append(true_negative, point)
            if point[0] < 2.5 and point[1] >= pred_cutoff:
               false_positive = np.append(false_positive, point)
            if point[0] >= 2.5 and point[1] < pred_cutoff:
               false_negative = np.append(false_negative, point)
        
        true_positive_num = (len(true_positive)-2.0)/2.0
        true_negative_num = (len(true_negative)-2.0)/2.0
        false_positive_num = (len(false_positive)-2.0)/2.0
        false_negative_num = (len(false_negative)-2.0)/2.0
        #print true_positive_num, true_negative_num, false_negative_num, false_positive_num
        sensitivity = true_positive_num/(true_positive_num+false_negative_num)
        specificity = true_negative_num/(true_negative_num+false_positive_num)
        accuracy = (true_positive_num+true_negative_num)/(true_positive_num+false_positive_num+false_negative_num+true_negative_num)
        #print 'sensitivity', sensitivity, 'specificity', specificity, 'accuracy', accuracy
        accuracy_matrix = np.append(accuracy_matrix, np.array(accuracy))
    accuracy_matrix = np.delete(accuracy_matrix,0,0)
    #print accuracy_matrix
    line = slope*ground_truth+intercept
    
    # Get a figure
    feat_accuracy_fig = plt.figure()
    # Add a standard plot area to figure (111 is the standard location on the plot area)
    main_axes = feat_accuracy_fig.add_subplot(111)
    main_axes.set_title('Features_vs_Accuracy' + '_' + fof_file)
    main_axes.set_xlabel('Feature Percentage')
    main_axes.set_ylabel('Accuracy')
    main_axes.plot(percent_values,accuracy_matrix)
    feat_accuracy_fig.savefig(graph_destination_subfolder + os.sep + fof_file + '_' + 'Features_vs_Accuracy.png')
    #plot(percent_values,accuracy_matrix)
    #pylab.xlim(0, 100)
    #pylab.ylim(0, 1)
    #plt.title(
    #plt.xlabel('Feature Percentage')
    #plt.ylabel('Accuracy')
    #plt.savefig(str(f)[12::] + '_' + 'Features_vs_Accuracy.png')
    
    # !!!!!!!!! Bail after first graph!
    break
