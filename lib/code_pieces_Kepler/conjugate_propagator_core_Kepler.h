int sid = blockIdx.x*blockDim.x + threadIdx.x;                                                                                                                                
if (sid >= c_threads) return;

for(int i = 0 ; i < c_nSpin*c_nSpin*c_nColor*c_nColor ; i++){
  inOut[i*c_stride + sid] = conj(inOut[i*c_stride + sid]);
 }

