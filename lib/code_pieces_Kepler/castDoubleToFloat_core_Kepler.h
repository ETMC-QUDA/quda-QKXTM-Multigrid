int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    out[(mu*3+c1)*c_stride+sid].x = (float) in[(mu*3+c1)*c_stride+sid].x;
    out[(mu*3+c1)*c_stride+sid].y = (float) in[(mu*3+c1)*c_stride+sid].y;
  }
