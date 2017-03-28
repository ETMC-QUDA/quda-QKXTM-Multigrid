# This python script colors the lattice in a hierarchical way
# Then the elements of the Hadamard matrix are reordered based on the coloring
# And then an element wise multiplication creates the stochastic probing vectors

########## NOTES #############
# There is an unsigned integer "k" running from 1 unitl ...
# From this integer we can specify several important quantities regarding the coloring
# The total number of colors is given by N_{hc} = 2 * 2^{d(k-1)} where d is the number of dimensions
# The distance seperating neighbors carrying the same color is D=2^k
# The size of the elementary coloring is given L_u=2^{k-1}
# A condition must be fulfilled in order to be able to do the coloring for a specific k
# The condition must be that the number of blocks in each direction must be even
# And that Ls%(2Lu)=0 and Lt%(2Lu)=0

import os
import sys
import numpy as np
import argparse
I=np.complex(0,-1)

def fcb(pars,Nc,Lu):
    lc=np.zeros((int(Nc/2),2),dtype=int)
    if pars['d'] == 2:
        for i in range(Lu):
            for j in range(Lu):
                lc[i*Lu+j][0] = i*Lu*2+j*2+0 # put even colors for even blocks
                lc[i*Lu+j][1] = i*Lu*2+j*2+1 # put odd colors for odd blocks
    elif pars['d'] == 3:
        for i in range(Lu):
            for j in range(Lu):
                for k in range(Lu):
                    lc[i*Lu*Lu+j*Lu+k][0] = (i*Lu*Lu+j*Lu+k)*2+0 # put even colors for even blocks
                    lc[i*Lu*Lu+j*Lu+k][1] = (i*Lu*Lu+j*Lu+k)*2+1 # put odd colors for odd blocks
    else:
        for i in range(Lu):
            for j in range(Lu):
                for k in range(Lu):
                    for l in range(Lu):
                        lc[i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l][0] = (i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l)*2+0 # put even colors for even blocks
                        lc[i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l][1] = (i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l)*2+1 # put odd colors for odd blocks
    return lc

def get_ind2Vec(idx,L,d):
    x=np.zeros((d),dtype=np.int)
    if d == 2:
        x[1]=int(idx/L)
        x[0]=idx-x[1]*L
    elif d == 3:
        x[2]=int(idx/(L**2))
        x[1]=int(idx/L)-x[2]*L
        x[0]=idx - x[2]*L**2 - x[1]*L
    else:
        x[3]=int(idx/(L**3))
        x[2]=int(idx/(L**2)) - x[3]*L
        x[1]=int(idx/L) - x[3]*L**2 - x[2]*L
        x[0]=idx-(x[3]*L**3+x[2]*L**2+x[1]*L)
    return x
        
def get_vec2Idx(x,L,d):
    if d == 2:
        idx=x[0]+x[1]*L
    elif  d == 3:
        idx=x[0]+x[1]*L+x[2]*L**2
    else:
        idx=x[0]+x[1]*L+x[2]*L**2+x[3]*L**3
    return idx
        
                
def hch_coloring(pars,Nc,Lu):
    vol=pars['Ls']**(pars['d']-1)*pars['Lt']
    Vc=np.zeros((vol),dtype=np.int) # array carrying the color for each lattice point
    lc=fcb(pars,Nc,Lu)
    for i in range(vol):
        gx=get_ind2Vec(i,pars['Ls'],pars['d']) # find the global position
        bx=(gx/Lu).astype(int) # find the position of the block
        eo=bx.sum() & 1 # find out if the block is even or odd
        lx=gx-Lu*bx
        Vc[i]=lc[get_vec2Idx(lx,Lu,pars['d'])][eo]
    return Vc
        
def check_coloring(pars,Vc,D):

    def brt(x,L):
        y=x
        if y >= L:
            y=y%L
        if y < 0:
            y=y+L
        return y

    if pars['d'] == 2:
        for t in range(pars['Lt']):
            for x in range(pars['Ls']):
                c1=Vc[get_vec2Idx([x,t],pars['Ls'],pars['d'])]
                for dx in range(-D+1,D):
                    for dt in range(-D+1,D):
                        ds=abs(dx)+abs(dt)
                        if ds < D and ds != 0:
                            xn=x+dx
                            xn=brt(xn,pars['Ls'])
                            tn=t+dt
                            tn=brt(tn,pars['Lt'])
                            c2=Vc[get_vec2Idx([xn,tn],pars['Ls'],pars['d'])]                            
                            if c1 == c2:
                                sys.stderr.write('Mistake with coloring: Same color detected between neighbors (%d,%d) and (%d,%d)\n' %(x,t,xn,tn))
    if pars['d'] == 3:
        for t in range(pars['Lt']):
            for y in range(pars['Ls']):
                for x in range(pars['Ls']):
                    c1=Vc[get_vec2Idx([x,y,t],pars['Ls'],pars['d'])]
                    for dx in range(-D+1,D):
                        for dy in range(-D+1,D):
                            for dt in range(-D+1,D):
                                ds=abs(dx)+abs(dy)+abs(dt)
                                if ds < D and ds !=0:
                                    xn=x+dx
                                    xn=brt(xn,pars['Ls'])
                                    yn=y+dy
                                    yn=brt(yn,pars['Ls'])
                                    tn=t+dt
                                    tn=brt(tn,pars['Lt'])
                                    c2=Vc[get_vec2Idx([xn,yn,tn],pars['Ls'],pars['d'])]
                                    if c1 == c2:
                                        sys.stderr.write('Mistake with coloring: Same color detected between neighbors (%d,%d,%d) and (%d,%d,%d)\n' %(x,y,t,xn,yn,tn))
    if pars['d'] == 4:
        for t in range(pars['Lt']):
            print(t)
            for z in range(pars['Ls']):
                for y in range(pars['Ls']):
                    for x in range(pars['Ls']):
                        c1=Vc[get_vec2Idx([x,y,z,t],pars['Ls'],pars['d'])]
                        for dx in range(-D+1,D):
                            for dy in range(-D+1,D):
                                for dz in range(-D+1,D):
                                    for dt in range(-D+1,D):
                                        ds=abs(dx)+abs(dy)+abs(dz)+abs(dt)
                                        if ds < D and ds !=0:
                                            xn=x+dx
                                            xn=brt(xn,pars['Ls'])
                                            yn=y+dy
                                            yn=brt(yn,pars['Ls'])
                                            zn=z+dz
                                            zn=brt(zn,pars['Ls'])
                                            tn=t+dt
                                            tn=brt(tn,pars['Lt'])
                                            c2=Vc[get_vec2Idx([xn,yn,zn,tn],pars['Ls'],pars['d'])]
                                            if c1 == c2:
                                                sys.stderr.write('Mistake with coloring: Same color detected between neighbors (%d,%d,%d,%d) and (%d,%d,%d,%d)\n' %(x,y,t,xn,yn,tn))
                                    

def PrintColoring(pars,Vc):
    if pars['d'] == 2:
        for t in range(pars['Lt']):
            for x in range(pars['Ls']):
                sys.stdout.write('%03d ' % Vc[get_vec2Idx([x,t],pars['Ls'],pars['d'])])
            sys.stdout.write('\n')
            

def HadamardMatrix(n):
    assert n%2==0 # check that the dimensions are even
    # Create the matrix.
    H = np.ones( (n, n), dtype=np.int)
    # Initialize Hadamard matrix of order n.
    i1 = 1
    while i1 < n:
        for i2 in range(i1):
            for i3 in range(i1):
                H[i2+i1,i3]    = H[i2,i3]
                H[i2,i3+i1]    = H[i2,i3]
                H[i2+i1,i3+i1] = -1*H[i2,i3]
        i1 += i1
    return H

def HadamardElements(i,j):
    sum=0
    for k in range(32):
        sum += (i%2)*(j%2)
        i=i>>1
        j=j>>1
    if sum%2 == 0:
        return 1
    else:
        return -1

def RandomSourceZ4(Nv,seed=123456):
    def mapToZ4(x):
        if x<0.25:
            return 1
        elif x >= 0.25 and x < 0.5:
            return -1
        elif x >= 0.5 and x <0.75:
            return I
        else:
            return -I
    Z=np.random.random(Nv)
    Z=np.array([mapToZ4(x) for x in Z])
    return Z


def ProbingSource(Vc,Z,ic,Nc):
    assert ic < Nc
    assert len(Vc) == len(Z)
    HZ=np.array(Z)
    for i in range(len(Vc)):
        cl=Vc[i]
        assert cl < Nc
        HZ[i] *= HadamardElements(cl,ic)
    return HZ
    

def assertions(args):
    pars={}
    pars['d']=int(args['dimensions'])
    pars['Ls']=int(args['Ls'])
    pars['Lt']=int(args['Lt'])
    pars['k']=int(args['k'])
    pars['checkC']=args['checkColors']
    pars['Nsrcs']=int(args['Nsrcs'])
    assert pars['d'] >= 2 and  pars['d'] <= 4 # only 2,3,4 dimensions can be used
    assert pars['Ls']%2 == 0 and pars['Ls'] >= 4
    assert pars['Lt']%2 == 0 and pars['Lt'] >= 4
    assert pars['k'] > 0
    return pars

def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(prog='Hierarchical coloring', description='This code performs the hierarchical coloring of the lattice')
    parser.add_argument('--dimensions',help='The number of dimensions that we want to have',default=2)
    parser.add_argument('--Ls',help='Lattice extent in the spatial directions',default=8)
    parser.add_argument('--Lt',help='Lattice extent in the temporal direction',default=16)
    parser.add_argument('--k',help='The integer number needed to specify the distance (not the distance) i.e k=1 is even-odd coloring',default=1)
    parser.add_argument('--checkColors',help='This option allow us to do a selfconsistent check to see if the coloring is done correctly',action='store_true')
    parser.add_argument('--Nsrcs',help='The number of stochastic sources we want to compute',default=2)
    args = vars(parser.parse_args())
    pars=assertions(args)

    # import variables needed for the coloring
    Nc=2*2**(pars['d']*(pars['k']-1))
    D=2**pars['k']
    Lu=2**(pars['k']-1)
    
    # checks that the Lattice extents are fine for the specific distance coloring
    assert pars['Ls']%(2*Lu) ==0 
    assert pars['Lt']%(2*Lu) ==0

    Vc=hch_coloring(pars,Nc,Lu)
    if pars['d'] == 2:
        PrintColoring(pars,Vc)
    if pars['checkC']:
        check_coloring(pars,Vc,D) # check if any neighbor in distance smaller than D has the same color

    for nsrc in range(pars['Nsrcs']):
        Z=RandomSourceZ4(len(Vc))
        for ic in range(Nc): # do the whole Hadamard basis
            HZ=ProbingSource(Vc,Z,ic,Nc)
            print(HZ)
############################                                                                                                                                                  
if __name__ == "__main__":
    sys.exit(main())

