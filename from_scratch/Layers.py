import numpy as np

class Fully_Connected:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dW = 0
        self.dB = 0
        self.cache = 0
        
    def forward(self, inputs):
        self.cache = inputs
        out = np.dot(inputs, self.weights) + self.biases
        return out
    
    def backprop(self,delta):
        self.dW = np.dot(self.cache.T,delta)/delta.shape[0]
        self.dB = np.sum(delta,axis=0)/delta.shape[0]
        delta = np.dot(delta,self.weights.T)
        return delta
        

class ConvLayer():
    def __init__(self,nc_inputs,num_filters,kernel_size,stride=1,padding=0):
        # TODO : add padding, strides
        self.nc_inputs = nc_inputs
        self.num_filters =num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = 0.01 * np.random.randn(num_filters,nc_inputs,kernel_size, kernel_size)
        self.h_out = 0
        self.w_out = 0
        self.cache = 0
        self.dW = 0  
        
    
    def backprop(self,dY):
        """
            Backpropagation of gradient
            dY : global gradient
            return new global grad and store local grad
        """
        grad = np.zeros(self.cache.shape)
        self.dW = np.zeros((self.num_filters,self.nc_inputs,self.kernel_size,self.kernel_size))
        flipped_kernels = np.flip(self.kernels,axis=(2,3))
        for patch_in,i,j in self.generate_patch(self.cache,dY.shape[2],1,self.kernel_size):
            for n_filter in range(self.num_filters):
                for nc in range(self.kernels.shape[1]):
                    r = patch_in[:,nc,:,:]*dY[:,n_filter,:,:]
                    s = np.sum(r,axis = (0,1,2))/grad.shape[0]
                    self.dW[n_filter,nc,i,j] = s
        pad = np.pad(dY,((0,),(0,),(self.kernel_size-1,),(self.kernel_size-1,)))
        for patch_dY,i,j in self.generate_patch(pad,self.kernel_size,1,self.cache.shape[2]):
            for nc in range(self.cache.shape[1]):
                r = np.sum(patch_dY*flipped_kernels[:,nc,:,:],axis = 1)
                s = np.sum(r,axis = (1,2))
                grad[:,nc,i,j] = s
        return grad
        
    def compute_size(self,input_image):
        # Compute output size
        h_in, w_in = input_image.shape[2:4]
        self.h_out = (h_in - (self.kernel_size - 1) -1 )/self.stride +1
        self.h_out = np.floor(self.h_out).astype(int)
        self.w_out = (w_in - (self.kernel_size - 1) -1 )/self.stride +1
        self.w_out = np.floor(self.w_out).astype(int)


    def generate_patch(self,input_image,kernel_size,stride,bound):
        # generate patch which will be multiplied with conv filters
        start_x = 0
        start_y = 0
        for x in range(bound):
            patch_x = start_x
            start_x +=stride
            start_y = 0
            for y in range(bound):
                patch_y = start_y
                start_y += stride
                image_patch = input_image[:,:,patch_x:patch_x+kernel_size,patch_y:patch_y+kernel_size]
                yield image_patch,x,y
    
    def forward(self, inputs):
        # Forward input
        self.compute_size(inputs)
        self.cache = inputs
        feature_maps = np.zeros((inputs.shape[0],self.num_filters,self.h_out,self.w_out))
        # Pour chaque filtre de sortie
        for n_filter in range(self.num_filters):
            for patch,i,j in self.generate_patch(inputs,self.kernel_size,self.stride,self.w_out):
                # On recupère un patch de shape (N,Cin,i,j)
                out_patch = patch*self.kernels[n_filter]
                feature_maps[:,n_filter,i,j] = np.sum(out_patch,axis = (1,2,3))
        return feature_maps


class MaxPooling():
    def __init__(self,size,stride = None,padding = 0):
        if stride==None :
            self.stride = size
        else:
            self.stride = stride
        self.padding = padding
        self.size = size
        self.h_out = 0
        self.w_out = 0
        self.cache = 0
        
    def backprop(self,global_grad):
        """
            Backpropagation of gradient
            global_grad : global gradient
        """
        a = 0
        b = 0
        start_x = 0
        start_y = 0
        inputs = self.cache
        grad = np.zeros(inputs.shape)
        for (patch,i,j) in self.generate_patch(inputs):
            for N in range(inputs.shape[0]):
                for nc in range(inputs.shape[1]):
                    arg = np.argmax(patch[N,nc,:,:])
                    m = np.unravel_index(arg,patch.shape)
                    grad[N,nc,m[2]+i*self.size,m[3]+j*self.size] = global_grad[N,nc,a,b]
            b+=1
            if b==global_grad.shape[2]:
                b=0
                a+=1
        return grad
                            
        
    def compute_size(self,input_image):
        # Compute output size
        h_in, w_in = input_image.shape[2:4]
        self.h_out = (h_in - (self.size - 1) -1 )/self.stride +1
        self.h_out = np.floor(self.h_out).astype(int)
        self.w_out = (w_in - (self.size - 1) -1 )/self.stride +1
        self.w_out = np.floor(self.w_out).astype(int)
    
    
    def generate_patch(self,input_image):
        # generate patch which will be multiplied with conv filters
        start_x = 0
        start_y = 0
        self.compute_size(input_image)
        self.map_max_indices = np.zeros(input_image.shape)
        feature_maps = np.zeros((input_image.shape[0],input_image.shape[1],self.h_out,self.w_out))
        for x in range(self.h_out):
            patch_x = start_x
            start_x += self.size
            start_y = 0
            for y in range(self.w_out):
                patch_y = start_y
                start_y += self.stride
                image_patch = input_image[:,:,patch_x:patch_x+self.size,patch_y:patch_y+self.size]
                yield image_patch,x,y
    
    
    
    def forward(self, inputs):
        # Renvoie la taille des features map de sortie
        self.compute_size(inputs)
        # Crée un tableau vide dans lequel on stock les features map
        feature_maps = np.zeros((inputs.shape[0],inputs.shape[1],self.h_out,self.w_out))
        # Pour chaque on prend le max de chaque patch
        self.cache = inputs
        for (patch,i,j) in self.generate_patch(inputs):
            feature_maps[:,:,i,j] = np.amax(patch,axis = (2,3))
        return feature_maps
        