int sid = blockIdx.x*blockDim.x + threadIdx.x;                                                                                                                                
if (sid >= c_threads) return;

inOut[0*c_stride + sid] = conj(inOut[0*c_stride + sid]);
inOut[1*c_stride + sid] = conj(inOut[1*c_stride + sid]);
inOut[2*c_stride + sid] = conj(inOut[2*c_stride + sid]);
inOut[3*c_stride + sid] = conj(inOut[3*c_stride + sid]);
inOut[4*c_stride + sid] = conj(inOut[4*c_stride + sid]);
inOut[5*c_stride + sid] = conj(inOut[5*c_stride + sid]);
inOut[6*c_stride + sid] = conj(inOut[6*c_stride + sid]);
inOut[7*c_stride + sid] = conj(inOut[7*c_stride + sid]);
inOut[8*c_stride + sid] = conj(inOut[8*c_stride + sid]);
inOut[9*c_stride + sid] = conj(inOut[9*c_stride + sid]);
inOut[10*c_stride + sid] = conj(inOut[10*c_stride + sid]);
inOut[11*c_stride + sid] = conj(inOut[11*c_stride + sid]);
