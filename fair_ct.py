import numpy as np
from scipy import stats
import pandas as pd
from graphviz import Digraph
import os

class node(object):

    def __init__(self, depth=-1, prev=None, left=None, right=None):
        self.depth = depth
        self.prev = prev
        self.left = left
        self.right = right

class fair_ct(object):

    def __init__(
            self, max_depth=10, min_samples_leaf=10, min_samples_size=10,\
            min_improve=0.0, verbos_split = False):
        self.done_fit = False
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        #self.min_samples_split = min_samples_split
        self.min_samples_size = min_samples_size
        self.min_improve = min_improve

        self.node_count = 0
        self.node_list = []

        self.verbos_split = verbos_split
    
    def _calc_effect(self, y, t):
        #print("len y = ", len(y))
        #print("len t = ", len(t))
        n0 = np.sum(t==0)
        n1 = np.sum(t==1)
        if(n0!=0):
            y0_list = y[t==0]
            mean0 = np.mean(y0_list)
            std0 = (1./(n0))**0.5*np.std(y0_list)
        else:
            mean0 = 0 
            std0 = 0 
        
        if(n1!=0):
            y1_list = y[t==1]
            mean1 = np.mean(y1_list)
            std1 = (1./(n1))**0.5*np.std(y1_list)
        else:
            mean1 = 0 
            std1 = 0 
            
        if(n0!=0 and n1!=0): 
            effect = mean1 - mean0
            std = (std0**2 + std1**2)**0.5
        else: 
            effect = np.nan
            std = np.nan
            #t, p = stats.ttest_ind(y0, y1, equal_var=False)
        return (effect, mean0, mean1), (std, std0, std1), (n0, n1)

    def fit(self,\
            x_train ,y_train, t_train,\
            x_est, y_est, t_est):
        
        n_train = x_train.shape[0]
        n_est = x_est.shape[0]
        n_feature = x_train.shape[1]
        p_treat = np.mean(t_train)
        
        ## record some information into class
        self.n_feature = n_feature
        self.n_train = n_train
        self.n_est = n_est 
        self.p_treat = p_treat

        ## check dimensions of input data
        assert(x_train.shape == (n_train, n_feature))
        assert(y_train.shape == (n_train,))
        assert(t_train.shape == (n_train,))
        assert(x_est.shape == (n_est, n_feature))
        assert(y_est.shape == (n_est,))
        assert(t_est.shape == (n_est,))

        self.root = self._rec_split(\
                0,\
                x_train, y_train, t_train,\
                x_est, y_est, t_est,\
                n_train, n_est, p_treat)

        self.done_fit = True

    def valid(self,\
            x_valid, y_valid, t_valid):
        n_valid = x_valid.shape[0]
        n_feature = x_valid.shape[1]

        ## check valid data has the same dim as train
        assert(n_valid == self.n_train)
        assert(n_feature == self.n_feature)

        ## check dim matches
        assert(y_valid.shape == (n_valid,))
        assert(t_valid.shape == (n_valid,))

        valid_loss = self._rec_valid_loss(\
                0, self.root,\
                x_valid, y_valid, t_valid,\
                self.n_train, self.n_est, self.p_treat)

        return valid_loss

    def _rec_valid_loss(self, depth, node_i,\
            x_valid_s, y_valid_s, t_valid_s,\
            n_train, n_est, p_treat):
        if(node_i.left == None):
            n_treat_i = np.sum(t_valid_s)

            ## protect against imbalance data
            if(n_treat_i<2 or (t_valid_s.shape[0]-n_treat_i)<2): return np.inf

            return self._calc_loss_node(y_valid_s, t_valid_s, n_train, n_est, p_treat)
        
        ## apply the mask
        mask_valid_left = (x_valid_s[:, node_i.split_feature] <= node_i.split_val)
        mask_valid_right = ~mask_valid_left

        loss_left = self._rec_valid_loss(depth+1, node_i.left,\
                x_valid_s[mask_valid_left], y_valid_s[mask_valid_left], t_valid_s[mask_valid_left],\
                n_train, n_est, p_treat)

        loss_right = self._rec_valid_loss(depth+1, node_i.right,\
                x_valid_s[mask_valid_right], y_valid_s[mask_valid_right], t_valid_s[mask_valid_right],\
                n_train, n_est, p_treat)
        
        return loss_left + loss_right
    
    def _get_split(self,\
            x_train_s, y_train_s, t_train_s,\
            x_est_s, y_est_s, t_est_s,\
            n_train, n_est, p_treat):
       
        if(False):
            ## a place-holder, return median value of first feature as split
            return {"feature":0, "val":np.median(x_train_s[:,0]) }

        n_feature = x_train_s.shape[1]
        
        opt_split = {"feature":None, "val":None}
        opt_gain = None

        for i in range(n_feature):
            feature_val = np.unique(x_train_s[:, i])
            
            for val in feature_val:
                mask_train_left = x_train_s[:,i]<=val
                mask_train_right = ~mask_train_left
                mask_est_left = x_est_s[:,i]<=val
                mask_est_right = ~mask_est_left
                
                ## filter the mask by the minimum size constrains
                if(np.sum(mask_train_left)<self.min_samples_leaf): continue
                if(np.sum(mask_train_right)<self.min_samples_leaf): continue
                if(np.sum(mask_est_left)<self.min_samples_leaf): continue
                if(np.sum(mask_est_right)<self.min_samples_leaf): continue

                ## filter the mask by the minimum sample size
                t_train_left = t_train_s[mask_train_left]
                t_train_right = t_train_s[mask_train_right]
                t_est_left = t_est_s[mask_est_left]
                t_est_right = t_est_s[mask_est_right]

                if(np.sum(t_train_left)<self.min_samples_size): continue
                if(np.sum(1-t_train_left)<self.min_samples_size): continue
                if(np.sum(t_train_right)<self.min_samples_size): continue
                if(np.sum(1-t_train_right)<self.min_samples_size): continue
                
                if(np.sum(t_est_left)<self.min_samples_size): continue
                if(np.sum(1-t_est_left)<self.min_samples_size): continue
                if(np.sum(t_est_right)<self.min_samples_size): continue
                if(np.sum(1-t_est_right)<self.min_samples_size): continue               

                ## calculate the objective function of a trail split
                obj_parent = self._calc_loss_node(y_train_s, t_train_s, n_train, n_est, p_treat)
                obj_left = self._calc_loss_node(y_train_s[mask_train_left], t_train_s[mask_train_left],\
                        n_train, n_est, p_treat)
                obj_right = self._calc_loss_node(y_train_s[mask_train_right], t_train_s[mask_train_right],\
                        n_train, n_est, p_treat)
                obj_gain = obj_left + obj_right - obj_parent
                if(obj_gain<self.min_improve): continue
                
                ## Here the obj_gain should be maximized
                if(opt_gain == None or obj_gain>opt_gain):
                    opt_gain = obj_gain
                    opt_split["feature"] = i
                    opt_split["val"] = val
        if(self.verbos_split): 
            print("opt_gain = ", opt_gain)
            print("opt_split = ", opt_split)

        return opt_split
                
    def _calc_loss_node(self, y_s, t_s, n_train, n_est, p_treat):
        n_samples = y_s.shape[0]
        mask_t_0 = t_s==0
        mask_t_1 = t_s==1
        tau = np.mean(y_s[mask_t_1]) - np.mean(y_s[mask_t_0])
        s2_0 = np.var(y_s[mask_t_0])
        s2_1 = np.var(y_s[mask_t_1])

        return 1./n_train*n_samples*tau**2 \
                - (1./n_train+1./n_est)*(s2_1/p_treat + s2_0/(1.-p_treat))

    def _register_node(self, node_i):
        node_i.index = self.node_count
        self.node_count += 1
        self.node_list.append(node_i)

    def _rec_split(self,\
            depth,\
            x_train_s, y_train_s, t_train_s,\
            x_est_s, y_est_s, t_est_s,\
            n_train, n_est, p_treat):

        ## terminate condition
        if(depth > self.max_depth):
            return None

        ## creat a node and calc effect
        node_i = node(depth=depth)
        self._register_node(node_i)
        node_i.effect = self._calc_effect(y_est_s, t_est_s)
        node_i.samples = (x_train_s.shape[0], x_est_s.shape[0])

        opt_split = self._get_split(
                x_train_s, y_train_s, t_train_s,\
                x_est_s, y_est_s, t_est_s,\
                n_train, n_est, p_treat)

        ## record split information
        node_i.split_feature = opt_split["feature"]
        node_i.split_val = opt_split["val"]

        if(opt_split["feature"]==None): 
            return node_i

        ## apply the opt_split, get left and right sub data
        mask_train_left = (x_train_s[:, opt_split["feature"]] <=  opt_split["val"])
        mask_train_right = ~mask_train_left
        mask_est_left = (x_est_s[:, opt_split["feature"]] <=  opt_split["val"])
        mask_est_right = ~mask_est_left
 
        
        node_i.left = self._rec_split(depth+1,\
                x_train_s[mask_train_left], y_train_s[mask_train_left], t_train_s[mask_train_left],\
                x_est_s[mask_est_left], y_est_s[mask_est_left], t_est_s[mask_est_left],\
                n_train, n_est, p_treat)


        node_i.right = self._rec_split(depth+1,\
                x_train_s[mask_train_right], y_train_s[mask_train_right], t_train_s[mask_train_right],\
                x_est_s[mask_est_right], y_est_s[mask_est_right], t_est_s[mask_est_right],\
                n_train, n_est, p_treat)
       
        return node_i
    
    def _rec_leaf_node_info(self, node_i, x_est_s, y_est_s, t_est_s, z_est_s):
        if(node_i.left == None):
            ## return the information of leaf node
            assert(node_i.right==None)
            effect_est = self._calc_effect(y_est_s, t_est_s)
            effect_z_list = {}
            for zi in np.unique(z_est_s):
                mask_zi = (z_est_s == zi)
                effect_zi = self._calc_effect(y_est_s[mask_zi], t_est_s[mask_zi])
                effect_z_list[zi] = effect_zi

            tmp = [{"index":node_i.index, "effect":effect_est, "effect_z_list":effect_z_list}]
            node_i.z_info = tmp[0]
            return tmp
        else:
            ## split the est samples
            mask_left = (x_est_s[:,node_i.split_feature] <= node_i.split_val)
            mask_right = ~mask_left
            res_left = self._rec_leaf_node_info(node_i.left,\
                    x_est_s[mask_left], y_est_s[mask_left], t_est_s[mask_left], z_est_s[mask_left])
            res_right = self._rec_leaf_node_info(node_i.right,\
                    x_est_s[mask_right], y_est_s[mask_right], t_est_s[mask_right], z_est_s[mask_right])
            return res_left + res_right

    def get_leaf_node_info(self, x_est, y_est, t_est, z_est):
        return self._rec_leaf_node_info(self.root, x_est, y_est, t_est, z_est)
        

    def _rec_traversal(self, node_i):
        if(node_i == None): return
        if(node_i.left == None): 
            print("Leaf", end = " ")
            print("effect = (%.2f, %.2f, %.2f, %2f, %.2f, %.2f), samples = (%d, %d), "%\
                (node_i.effect[0]+node_i.effect[1]+node_i.samples))
        else: 
            print("Regular", end = " ")
            #print(node_i.effect)
            print("effect = (%.2f, %.2f, %.2f, %2f, %.2f, %.2f), samples = (%d, %d), "%\
                (node_i.effect[0]+node_i.effect[1]+node_i.samples), end = "")
            print("split = (%d, %.2f)"%(node_i.split_feature, node_i.split_val))

            self._rec_traversal(node_i.left)
            self._rec_traversal(node_i.right)


    def traversal(self):
        self._rec_traversal(self.root)

    def _rec_predict(self, node, x):
        if(node.left == None): return node.effect, node.index
        if(x[node.split_feature]<=node.split_val): return self._rec_predict(node.left, x)
        return self._rec_predict(node.right, x)

    def predict(self, X, ifindex=False):
        X = np.array(X)
        n_data = X.shape[0]
        tau = np.zeros(n_data)
        index = [] 
        for i in range(n_data):
            effect_i, index_i = self._rec_predict(self.root, X[i])
            tau[i] = effect_i[0][0]
            index.append(index_i)
        if(ifindex): return tau, index
        else: return tau
    
    def _make_node_text(self, node_i, feature_name, ifshowz):
        effect = node_i.effect 
        tau = effect[0][0]
        n = effect[2][0] + effect[2][1]
        s = ""
        if(node_i.left != None): s += "%s <= %.3g"%(feature_name[node_i.split_feature], node_i.split_val) + "\n"
        s += "tau = %.2g"%(tau)+"\n"+"n = %d"%(n)+"\n"
        s += "r = %.2g"%(effect[2][1]*1./n) + "\n"
        
        ## leaf node
        if(ifshowz):
            try:
                effect_z_list = node_i.z_info["effect_z_list"]
                s += "------\n"
                for zi in effect_z_list:
                    effect_zi = effect_z_list[zi]
                    tau_zi = effect_zi[0][0]
                    n_zi = effect_zi[2][0] + effect_zi[2][1]
                    r_zi = effect_zi[2][1]*1./n_zi
                    s += "--[z = %d]--"%(zi) + "\n"
                    s += "tau = %.2g"%(tau_zi)+"\n"+"n = %d"%(n_zi)+"\n"
                    s += "r = %.2g"%(r_zi) + "\n"

            except AttributeError:
                pass

        return s


    def _rec_show_tree(self, node_i, node_idx, dot, dot_info, feature_name, ifshowz):

        node_label = "%d"%(node_idx)
        node_text = node_label
        
        #if(node_i.left != None): feature = node_i.split_feature
        #else: feature = -1
        dot.node(node_label, self._make_node_text(node_i, feature_name, ifshowz), style="filled", fillcolor="#B8F0B2")

        if(node_i.left != None):
            dot_info[0]+=1
            left_idx = dot_info[0]
            left_label = "%d"%(left_idx)
            left_text = left_label
            dot.node(left_label, left_text)
            dot.edge(node_label, left_label)
            self._rec_show_tree(node_i.left, left_idx, dot, dot_info, feature_name, ifshowz)
        if(node_i.right != None):
            dot_info[0] += 1
            right_idx = dot_info[0]
            right_label = "%d"%(right_idx)
            right_text = right_label
            dot.node(right_label, right_text)
            dot.edge(node_label, right_label)
            self._rec_show_tree(node_i.right, right_idx, dot, dot_info, feature_name, ifshowz)

    
    def show_tree(self, fname, feature_name = None, ifshowz=True):
        
        fname_split = os.path.splitext(fname)

        fname_pref, fname_suf = fname_split[0], fname_split[1][1:]
        dot = Digraph(node_attr={'shape': 'box'}, format=fname_suf)
        dot.node("0", "root")
        
        if(feature_name == None):
            feature_name = []
            for i in range(self.n_feature):
                feature_name.append("feature_%d"%(i))
        
        assert(len(feature_name) == self.n_feature)

        self._rec_show_tree(self.root, 0, dot, [0], feature_name, ifshowz)
        dot.render(fname_pref, view=False)
        return dot
