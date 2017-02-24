int space_stride = c_localL[0]*c_localL[1]*c_localL[2];
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= space_stride) return;

FLOAT2 projector[4][4];
get_Projector(projector, PARTICLE, PID); // get the correct projector

unsigned short int mu,nu,ku,lu,c1,c2,c3,c1p,c2p,c3p;
FLOAT2 factor;
FLOAT2 spinor[4][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int ic = 0 ; ic < 3 ; ic++){
    spinor[mu][ic].x = 0.; spinor[mu][ic].y = 0.;
  }

#define PROP3D(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + ( (mu*4+nu)*3*3 + a*3 + b ) * space_stride) )


for(int cc1 = 0 ; cc1 < 6 ; cc1++){
  c1 = c_eps[cc1][0];
  c2 = c_eps[cc1][1];
  c3 = c_eps[cc1][2];
  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
    c1p = c_eps[cc2][0];
    c2p = c_eps[cc2][1];
    c3p = c_eps[cc2][2];
    if(c3p == c_c2){
      for(int idx = 0 ; idx < 16 ; idx++){
        mu = c_NTN_indices[idx][0];
        nu = c_NTN_indices[idx][1];
        ku = c_NTN_indices[idx][2];
        lu = c_NTN_indices[idx][3];
	if(lu == c_nu){
	  for(int a = 0 ; a < 4 ; a++)
            for(int b = 0 ; b < 4 ; b++)
              if( norm(projector[b][a]) > 1e-3 ){
		factor = (-1)*c_sgn_eps[cc1]*c_sgn_eps[cc2]*c_NTN_values[idx]*projector[b][a];
		spinor[nu][c3] = spinor[nu][c3] + factor * PROP3D(tex,mu,b,c1,c1p) * PROP3D(tex,a,ku,c2,c2p);
		spinor[nu][c3] = spinor[nu][c3] + factor * PROP3D(tex,mu,ku,c1,c1p) * PROP3D(tex,a,b,c2,c2p);
	      }
	}
      }
    }
  }
 }

for(int mu = 0 ; mu < 4 ; mu++)
  for(int ic = 0 ; ic < 3 ; ic++)
    out[mu*3*c_stride + ic*c_stride + timeslice*space_stride + sid] = spinor[mu][ic];

#undef PROP3D
