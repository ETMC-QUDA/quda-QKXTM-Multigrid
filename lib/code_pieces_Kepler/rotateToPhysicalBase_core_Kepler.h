
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

Float2 P[4][4];
Float2 PT[4][4];
Float2 imag_unit;
imag_unit.x = 0.;
imag_unit.y = 1.;
// we have 9 color

for(int c1 = 0 ; c1 < c_nColor ; c1++)
  for(int c2 = 0 ; c2 < c_nColor ; c2++){

    for(int mu = 0 ; mu < c_nSpin ; mu++)
      for(int nu = 0 ; nu < c_nSpin; nu++)
	P[mu][nu] = inOut[mu*c_nSpin*c_nColor*c_nColor*c_stride + nu*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid];
      
    
    PT[0][0] = 0.5 * (P[0][0] + sign * ( imag_unit * P[0][2] ) + sign * ( imag_unit * P[2][0] ) - P[2][2]);
    PT[0][1] = 0.5 * (P[0][1] + sign * ( imag_unit * P[0][3] ) + sign * ( imag_unit * P[2][1] ) - P[2][3]);
    PT[0][2] = 0.5 * (sign * ( imag_unit * P[0][0] ) + P[0][2] - P[2][0] + sign * ( imag_unit * P[2][2] ));
    PT[0][3] = 0.5 * (sign * ( imag_unit * P[0][1] ) + P[0][3] - P[2][1] + sign * ( imag_unit * P[2][3] ));

    PT[1][0] = 0.5 * (P[1][0] + sign * ( imag_unit * P[1][2] ) + sign * ( imag_unit * P[3][0] ) - P[3][2]);
    PT[1][1] = 0.5 * (P[1][1] + sign * ( imag_unit * P[1][3] ) + sign * ( imag_unit * P[3][1] ) - P[3][3]);
    PT[1][2] = 0.5 * (sign * ( imag_unit * P[1][0] ) + P[1][2] - P[3][0] + sign * ( imag_unit * P[3][2] ));
    PT[1][3] = 0.5 * (sign * ( imag_unit * P[1][1] ) + P[1][3] - P[3][1] + sign * ( imag_unit * P[3][3] ));

    PT[2][0] = 0.5 * (sign * ( imag_unit * P[0][0] ) - P[0][2] + P[2][0] + sign * ( imag_unit * P[2][2] ));
    PT[2][1] = 0.5 * (sign * ( imag_unit * P[0][1] ) - P[0][3] + P[2][1] + sign * ( imag_unit * P[2][3] ));
    PT[2][2] = 0.5 * (sign * ( imag_unit * P[0][2] ) - P[0][0] + sign * ( imag_unit * P[2][0] ) + P[2][2]);
    PT[2][3] = 0.5 * (sign * ( imag_unit * P[0][3] ) - P[0][1] + sign * ( imag_unit * P[2][1] ) + P[2][3]);

    PT[3][0] = 0.5 * (sign * ( imag_unit * P[1][0] ) - P[1][2] + P[3][0] + sign * ( imag_unit * P[3][2] ));
    PT[3][1] = 0.5 * (sign * ( imag_unit * P[1][1] ) - P[1][3] + P[3][1] + sign * ( imag_unit * P[3][3] ));
    PT[3][2] = 0.5 * (sign * ( imag_unit * P[1][2] ) - P[1][0] + sign * ( imag_unit * P[3][0] ) + P[3][2]);
    PT[3][3] = 0.5 * (sign * ( imag_unit * P[1][3] ) - P[1][1] + sign * ( imag_unit * P[3][1] ) + P[3][3]);



    for(int mu = 0 ; mu < c_nSpin ; mu++)
      for(int nu = 0 ; nu < c_nSpin; nu++)
	inOut[mu*c_nSpin*c_nColor*c_nColor*c_stride + nu*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid] = PT[mu][nu];




  }
