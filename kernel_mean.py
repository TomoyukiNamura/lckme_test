"""Module :mod:`sklearn.kernel_ridge` implements kernel ridge regression."""

# Authors: Mathieu Blondel <mathieu@mblondel.org>
#          Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

import numpy as np
from scipy import linalg
import copy

from .base import BaseEstimator, RegressorMixin
from .metrics.pairwise import pairwise_kernels,euclidean_distances
from .linear_model.ridge import _solve_cholesky_kernel
from .utils import check_array, check_X_y
from .utils.validation import check_is_fitted
from .kernel_approximation import RBFSampler,Nystroem 

def calcPairwiseDist(X, method='method1'):
    # ユークリッド距離行列を計算
    matrix_pairdist = euclidean_distances(X)
    
    # method1:各データ点で最も距離が近いデータとの距離を計算し，ペアワイズ距離ベクトルとする
    if method=='method1':
        matrix_pairdist = matrix_pairdist + np.eye(matrix_pairdist.shape[0]) * np.max(matrix_pairdist)
        vector_pairdist = np.min(matrix_pairdist,axis=0)
    
    # method2:各データ点で最も距離が近いデータとの距離を計算し，ペアワイズ距離ベクトルとする(データのペア重複は除外)
    elif method=='method2':
        return None
#        matrix_pairdist = np.triu(matrix_pairdist, k=1) + np.tri(matrix_pairdist.shape[0]) * np.max(matrix_pairdist)
        #vector_pairdist = np.min(matrix_pairdist,axis=0)
        
    # method3:各データ点で，全てのデータとの距離を計算し，ペアワイズ距離ベクトルとする
    else:
        matrix_pairdist = np.triu(matrix_pairdist, k=1)
        vector_pairdist = matrix_pairdist[np.tri(matrix_pairdist.shape[0])!=1]
    
    return vector_pairdist


def hsic(emb_x,emb_y):
    '''
    ヒルベルトシュミット独立基準
    '''
    K = emb_x._get_kernel(emb_x.X)
    L = emb_y._get_kernel(emb_y.X)
    m = K.shape[0]
    H = np.eye(m) - np.dot(np.ones(m).reshape(-1,1),np.ones(m).reshape(1,-1))/m
    
    return np.trace(np.dot(np.dot(K,H),np.dot(L,H)))/(m-1)**2
    

def _inv_gram_matrix(emb_X, alpha=1,knn_id=None, sample_weight=None):
    if sample_weight is not None and not isinstance(sample_weight, float):
        sample_weight = check_array(sample_weight, ensure_2d=False)

    if knn_id is None:
        K = emb_X._get_kernel(emb_X.X)
    else:
        K = emb_X._get_kernel(emb_X.X[knn_id,:])
    alpha = np.atleast_1d(alpha)

    n_samples = K.shape[0]
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        print('Not supported!!')
        return None

    if one_alpha:
        # (K - alphaIn)^(-1)
        K.flat[::n_samples + 1] += alpha[0]
        inv_gram_ = linalg.inv(K)

        # 一応Kを元の値にもどす
        K.flat[::n_samples + 1] -= alpha[0]

    else:
        print('Not supported!!')
        return None

    return inv_gram_
        

def _inv_gram_matrix_nystroem(emb_X, subsample_id, alpha=1, n_components=10, knn_id=None, sample_weight=None):
    if sample_weight is not None and not isinstance(sample_weight, float):
        sample_weight = check_array(sample_weight, ensure_2d=False)
    
    K_nm = emb_X._get_kernel(emb_X.X, emb_X.X[subsample_id,:])
    K_mm = emb_X._get_kernel(emb_X.X[subsample_id,:], emb_X.X[subsample_id,:])
    
    alpha = np.atleast_1d(alpha)

    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        print('Not supported!!')
        return None

    if one_alpha:
        # (K - alphaIn)^(-1)
        K_mm.flat[::n_components + 1] *= alpha[0]
        
        inv_gram_ = np.dot(linalg.pinv(np.dot(K_nm.T,K_nm)+K_mm), K_nm.T).T

        # 一応Kを元の値にもどす
        K_mm.flat[::n_components + 1] /= alpha[0]

    else:
        print('Not supported!!')
        return None

    return inv_gram_,subsample_id


def _inv_approximated_gram_matrix(emb_X, transformer, alpha=1, n_components=10, knn_id=None, sample_weight=None):    
    if sample_weight is not None and not isinstance(sample_weight, float):
        sample_weight = check_array(sample_weight, ensure_2d=False)

    K_nm = transformer.transform(emb_X.X)
    K_mm = np.eye(n_components)

    alpha = np.atleast_1d(alpha)

    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        print('Not supported!!')
        return None

    if one_alpha:
        # (K - alphaIn)^(-1)
        K_mm.flat[::n_components + 1] *= n_components*alpha[0]
        
        inv_gram_ = np.dot(linalg.inv(np.dot(K_nm.T,K_nm)+K_mm), K_nm.T).T

        # 一応Kを元の値にもどす
        K_mm.flat[::n_components + 1] /= n_components*alpha[0]

    else:
        print('Not supported!!')
        return None

#    return inv_gram_,subsample_id
    return inv_gram_


def innerProduct(emb_x,emb_y):
    if emb_x.kernel!=emb_y.kernel or emb_x.gamma!=emb_y.gamma:
        print("error: 2 kernel isn't equal")
        return np.nan
        
    return np.dot(np.dot(emb_x.weights, emb_x._get_kernel(emb_x.X,emb_y.X)), emb_y.weights.T).reshape(1)
    
def RKHSnorm(emb_x):
    return np.sqrt(innerProduct(emb_x,emb_x))
    
def RKHSdist(emb_x,emb_y):
    iXX = innerProduct(emb_x,emb_x)
    iYY = innerProduct(emb_y,emb_y)
    iXY = innerProduct(emb_x,emb_y)
    return np.sqrt(iXX + iYY - 2.0 * iXY)
    

class KernelMean(object):
    def __init__(self, X, weights=None, kernel="linear", gamma=None, degree=3, coef0=1,kernel_params=None):
        if X.ndim==1:
            self.X = X.reshape(-1,1)
        else:
            self.X = X.reshape(-1,X.shape[1])

        if weights is not None:
            self.weights = weights.reshape(1, -1)
        else:
            self.weights = (np.ones(self.X.shape[0])/self.X.shape[0]).reshape(1, -1)

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def weighted_sum(self):
        return np.dot(self.weights,self.X)[0]
    
    def estimate(self, new_X):
        return np.dot(self.weights,self._get_kernel(X=self.X, Y=new_X)).reshape(-1,)
    

    """
    def __init__(self,weight,kernel,model):
        self.weight = np.array(weight)
        self.kernel = kernel
        self.model = np.array(model)
       
    # カーネル平均をの計算
    def calcKernelMean(self,x_lange):
        result = []
        
        for i in range(len(x_lange)):
            result.append(np.dot(self.weight, self.kernel(x_lange[i],self.model)))
            
        return result
    """


class ConditionalKernelMean(BaseEstimator, RegressorMixin):
    """Conditional Kernel Mean.
    """
    def __init__(self, alpha=1, method = 'default',n_components=None):
        """
        method:{'default','nw','rss'}
            default: original    
            nw: nadaraya-watoson(conditional kernel density estimation)
            rss: original with random sub sample
        """
        
        self.alpha = alpha
        self.method = method
        self.n_components = n_components

    def fit(self, emb_X, emb_y, sample_weight=None, subsample_id=None):
        """Fit Conditional Kernel Mean model
        """
        
        self.emb_X = copy.deepcopy(emb_X)
        self.emb_y = copy.deepcopy(emb_y)
        
        if self.method == 'rss':
            if subsample_id is None:
                subsample_id = np.random.randint(0,emb_X.X.shape[0],self.n_components)
                
            self.emb_X.X = self.emb_X.X[subsample_id,:]
            self.emb_X.weights = self.emb_X.weights[:,subsample_id]
            
            self.emb_y.X = self.emb_y.X[subsample_id,:]
            self.emb_y.weights = self.emb_y.weights[:,subsample_id]
            
        # Convert data
        #self.emb_X.X, self.emb_y.X = check_X_y(self.emb_X.X, self.emb_y.X, accept_sparse=("csr", "csc"), 
        #    multi_output=True,y_numeric=True)

        if self.method=='nw':
            self.inv_gram_ = np.nan
        
        else:
            self.inv_gram_ = _inv_gram_matrix(self.emb_X, alpha=self.alpha, knn_id=None, sample_weight=sample_weight)
                    
        return self
    

    def predict(self, X):
        """Predict using the Conditional Kernel Mean model
        """

        check_is_fitted(self, ["emb_X", "emb_y", "inv_gram_"])
        
        # Xをリシェイプ
        if X.ndim==1:
            X = X.reshape(-1,1)
        else:
            X = X.reshape(-1,X.shape[1])

        # 入力Xと学習データXのカーネル行列作成
        K_X = self.emb_X._get_kernel(self.emb_X.X, X)

        # 重み計算
        if self.method=='nw':
            weights_y = K_X
            
            for i in range(X.shape[0]):
                weights_y[:,i] = weights_y[:,i]/np.sum(weights_y[:,i])
        
        else:
            weights_y = np.dot(self.inv_gram_, K_X)                 

        result = []
        for i in range(X.shape[0]):
            tmp = copy.deepcopy(self.emb_y)
            tmp.weights = weights_y.T[i].reshape(1, -1)
            result.append(tmp)

        return result        


class NystroemConditionalKernelMean(BaseEstimator, RegressorMixin):
    """Conditional Kernel Mean.
    """
    def __init__(self, alpha=1, n_components=10):
        self.alpha = alpha
        self.n_components = n_components


    def fit(self, emb_X, emb_y, sample_weight=None, subsample_id=None):
        """Fit Conditional Kernel Mean model
        """
        
        self.emb_X = copy.deepcopy(emb_X)
        self.emb_y = copy.deepcopy(emb_y)

        # Convert data
        #self.emb_X.X, self.emb_y.X = check_X_y(self.emb_X.X, self.emb_y.X, accept_sparse=("csr", "csc"), 
        #    multi_output=True,y_numeric=True)

        if subsample_id is None:
            subsample_id = np.random.randint(0,emb_X.X.shape[0],self.n_components)
            
        self.n_components = subsample_id.shape[0]
        self.inv_gram_,subsample_id = _inv_gram_matrix_nystroem(self.emb_X, subsample_id, alpha=self.alpha, n_components=self.n_components, knn_id=None, sample_weight=sample_weight)
        
        self.emb_X.X = self.emb_X.X[subsample_id,:]
        self.emb_X.weights = self.emb_X.weights[:,subsample_id]
        
        return self
    

    def predict(self, X):
        """Predict using the Conditional Kernel Mean model
        """

        check_is_fitted(self, ["emb_X", "emb_y", "inv_gram_"])
        
        # Xをリシェイプ
        if X.ndim==1:
            X = X.reshape(-1,1)
        else:
            X = X.reshape(-1,X.shape[1])

        # 入力Xと学習データXのカーネル行列作成
        K_X = self.emb_X._get_kernel(self.emb_X.X, X)

        # 重み計算
        weights_y = np.dot(self.inv_gram_, K_X)

        result = []
        for i in range(X.shape[0]):
            tmp = copy.deepcopy(self.emb_y)
            tmp.weights = weights_y.T[i].reshape(1, -1)
            result.append(tmp)

        return result        


class ApproximationConditionalKernelMean(BaseEstimator, RegressorMixin):
    """Conditional Kernel Mean.
    """
    def __init__(self, alpha=1, n_components=10, method='Nystroem',random_state=None):
        self.alpha = alpha
        self.n_components = n_components
        self.method = method
        self.random_state = random_state


    def fit(self, emb_X, emb_y, sample_weight=None):
        """Fit Conditional Kernel Mean model
        """
        
        self.emb_X = copy.deepcopy(emb_X)
        self.emb_y = copy.deepcopy(emb_y)

        # Convert data
        #self.emb_X.X, self.emb_y.X = check_X_y(self.emb_X.X, self.emb_y.X, accept_sparse=("csr", "csc"), 
        #    multi_output=True,y_numeric=True)

        if self.method == 'Nystroem':
            self.transformer = Nystroem(kernel=self.emb_X.kernel, gamma=self.emb_X.gamma, coef0=self.emb_X.coef0, degree=self.emb_X.degree, 
                                        kernel_params=self.emb_X.kernel_params, n_components=self.n_components, random_state=self.random_state)
            
        elif self.method == 'RBFSampler':
            self.transformer = RBFSampler(gamma=self.emb_X.gamma, n_components=self.n_components,random_state=self.random_state)
            
        else:
            print('This method is not supported')
            return None

        self.transformer.fit(self.emb_X.X)
        self.inv_gram_ = _inv_approximated_gram_matrix(self.emb_X, transformer=self.transformer, alpha=self.alpha, n_components=self.n_components, knn_id=None, sample_weight=sample_weight)
        
        return self
    

    def predict(self, X):
        """Predict using the Conditional Kernel Mean model
        """

        check_is_fitted(self, ["emb_X", "emb_y", "inv_gram_"])
        
        # Xをリシェイプ
        if X.ndim==1:
            X = X.reshape(-1,1)
        else:
            X = X.reshape(-1,X.shape[1])

        # 入力Xと学習データXのカーネル行列作成
#        K_X = self.emb_X._get_kernel(self.emb_X.X, X)
        K_X = self.transformer.transform(X).T

        # 重み計算
        weights_y = np.dot(self.inv_gram_, K_X)

        result = []
        for i in range(X.shape[0]):
            tmp = copy.deepcopy(self.emb_y)
            tmp.weights = weights_y.T[i].reshape(1, -1)
            result.append(tmp)

        return result   

class LocalConditionalKernelMean(BaseEstimator, RegressorMixin):
    """Lokal Conditional Kernel Mean with k-nearest neighbor.
    """
    def __init__(self, alpha=1,knn_k=10, random_sub_sampling=False):
        self.alpha = alpha
        self.knn_k = knn_k
        self.random_sub_sampling = random_sub_sampling


    def fit(self, emb_X, emb_y, sample_weight=None):
        """Fit Conditional Kernel Mean model
        """
        self.emb_X = copy.deepcopy(emb_X)
        self.emb_y = copy.deepcopy(emb_y)
        self.sample_weight = sample_weight

        # Convert data
        #self.emb_X.X, self.emb_y.X = check_X_y(self.emb_X.X, self.emb_y.X, accept_sparse=("csr", "csc"), 
        #    multi_output=True,y_numeric=True)
    

    def predict(self, X):
        """Predict using the Lokal Conditional Kernel Mean model with k-nearest neighbor
        """
        check_is_fitted(self, ["emb_X", "emb_y"])
        
        # Xをリシェイプ
        if X.ndim==1:
            X = X.reshape(-1,1)
        else:
            X = X.reshape(-1,X.shape[1])

        # self.knn_k > self.emb_X.X.shape[0]の場合
        if self.knn_k > self.emb_X.X.shape[0]:
            self.knn_k = self.emb_X.X.shape[0]


        # 入力Xと学習データXのカーネル行列作成
        K_X = self.emb_X._get_kernel(self.emb_X.X, X)

        result = []
        for i in range(X.shape[0]):
            if self.random_sub_sampling==False:
                # Kの列ごとに，Kの値がもっとも大きいk個のIndexを取得する
                knn_id = np.argsort(K_X[:,i])[::-1][0:self.knn_k]
            
            else:
                knn_id = np.random.choice(np.argsort(K_X[:,i])[::-1],self.knn_k)
            
            # グラム逆行列
            inv_gram_ = _inv_gram_matrix(self.emb_X, alpha=self.alpha,knn_id=knn_id, sample_weight=None)

            # 重み計算
            weights_y = np.dot(inv_gram_, K_X[knn_id,i])

            tmp = copy.deepcopy(self.emb_y)
            tmp.X = tmp.X[knn_id,:]
            tmp.weights = weights_y.T.reshape(1, -1)
            result.append(tmp)

        return result 


class DivideAndConquerCKME(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1 ,n_components=1, n_weaklearners=10):
        self.alpha = alpha
        self.n_components = n_components
        self.n_weaklearners = n_weaklearners
        
    def fit(self, emb_X, emb_y, sample_weight=None, subsample_id=None):
        # n_weaklearners個の弱学習器を学習
        self.weaklearners = []
        for i_weaklearner in range(self.n_weaklearners):
            tmp_model = ConditionalKernelMean(alpha=self.alpha, method='rss',n_components=self.n_components)
            self.weaklearners.append(tmp_model.fit(emb_X, emb_y)) 
        
    def predict(self, X):
        # 各弱学習器で予測
        result_weaklearners = []
        for i_weaklearner in range(self.n_weaklearners):
            result_weaklearners.append(self.weaklearners[i_weaklearner].predict(X))
            
        # 弱学習器分の条件付きカーネル平均の平均を計算
        result = []
        for i in range(X.shape[0]):
            tmp_X = np.array([])
            tmp_weights = np.array([])
            
            for i_weaklearner in range(self.n_weaklearners):
                tmp_X = np.append(tmp_X,result_weaklearners[i_weaklearner][i].X)
                tmp_weights = np.append(tmp_weights,result_weaklearners[i_weaklearner][i].weights)
                
            result.append(KernelMean(X=tmp_X, weights=tmp_weights/self.n_weaklearners, kernel=result_weaklearners[0][0].kernel, gamma=result_weaklearners[0][0].gamma, degree=result_weaklearners[0][0].degree, coef0=result_weaklearners[0][0].coef0,kernel_params=result_weaklearners[0][0].kernel_params))
        
        return result 
