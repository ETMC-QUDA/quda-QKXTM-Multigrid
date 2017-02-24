//////////////////////////////////         KX        ////////////////////////////////

extern "C" {
#include <lime.h>
}

#ifdef MULTI_GPU
#include <mpi.h>
#endif

static void qcd_swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}
/*  // maybe need it
static void qcd_swap_4(float *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
  {
    j=i; k=j+3;
    swap = *j; *j = *k;  *k = swap;
    j++; k--;
    swap = *j; *j = *k;  *k = swap;
  }
}
*/
static int qcd_isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}

static char* qcd_getParam(char token[],char* params,int len)
{
   int i,token_len=strlen(token);

   for(i=0;i<len-token_len;i++)
   {
      if(memcmp(token,params+i,token_len)==0)
      {
         i+=token_len;
         *(strchr(params+i,'<'))='\0';
         break;
      }
   }
   return params+i;
}

static char* qcd_getParamComma(char token[],char* params,int len)
{
  int i,token_len=strlen(token);

  for(i=0;i<len-token_len;i++)
    {
      if(memcmp(token,params+i,token_len)==0)
        {
          i+=token_len;
          *(strchr(params+i,','))='\0';
          break;
        }
    }
  return params+i;
}


// read ildg confs format
// using Alex modified function

extern Topology *default_topo;

typedef struct
{
  double re;
  double im;
} qcd_complex_16;


static void read_custom_binary_gauge_field (double **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4])
{
/*
read gauge fileld config stored in binary file
*/
  FILE *fid;
  int x1, x2, x3, x4, ln[4] = { 0, 0, 0, 0 };
  unsigned long long	lvol, ixh, iy, mu;
  char tmpVar[20];
  double *U = (double*)malloc(18*sizeof(double));
  double *resEvn[4], *resOdd[4];
  int nvh, iDummy;

  LimeReader *limereader;
  char *lime_type, *lime_data;
  n_uint64_t lime_data_size;
  char dummy;
  int  isDouble;
  int  error_occured=0;
  double dDummy;

#ifdef	MULTI_GPU
  MPI_Offset offset = 0;
  MPI_Datatype subblock;  //MPI-type, 5d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  unsigned long i=0;
  unsigned short chunksize;
  char *ftmp=NULL;
#else
  double *ftmp=NULL;
  unsigned long long	iread, idx;
#endif

  fid=fopen(fname,"r");
  if(fid==NULL)
    {
      errorQuda("Error reading configuration! Could not open path for reading");
    }
	
  if ((limereader = limeCreateReader(fid))==NULL)
    {
      errorQuda("Could not create limeReader");
    }
	
  while(limeReaderNextRecord(limereader)      != LIME_EOF )
    {
      lime_type				       = limeReaderType(limereader);
      
      if(strcmp(lime_type,"ildg-binary-data") == 0)
	break;
      
      if(strcmp(lime_type,"xlf-info")==0)
	{
	      lime_data_size = limeReaderBytes(limereader);
	      lime_data = (char * )malloc(lime_data_size);
	      limeReaderReadData((void *)lime_data,&lime_data_size, limereader);
	      
	      strcpy	(tmpVar, "kappa =");
	      sscanf(qcd_getParamComma(tmpVar,lime_data, lime_data_size),"%lf",&dDummy);    
	      printfQuda("Kappa given is : %f \t Kappa conf is : %f \t check them agree\n", inv_param->kappa , dDummy);
	      
	      strcpy	(tmpVar, "mu =");
	      sscanf(qcd_getParamComma(tmpVar,lime_data, lime_data_size),"%lf",&dDummy);
	      printfQuda("Mu given is : %f \t Mu conf is : %f \t may disagree for heavy quark\n" , inv_param->mu , dDummy);
	      
	      free(lime_data);
	}
	  
	  if(strcmp(lime_type,"ildg-format")==0)
	    {
	      lime_data_size = limeReaderBytes(limereader);
	      lime_data = (char * )malloc(lime_data_size);
	      limeReaderReadData((void *)lime_data,&lime_data_size, limereader);
	      
	      strcpy	(tmpVar, "<precision>");
	      sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&isDouble);    
	      printfQuda("Precision:\t%i bit\n",isDouble);
	      
	      strcpy	(tmpVar, "<lx>");
	      sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	      param->X[0]	 = iDummy/gridSize[0];
	      ln[0]		 = iDummy;
	      
	      strcpy	(tmpVar, "<ly>");
	      sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	      param->X[1]	 = iDummy/gridSize[1];
	      ln[1]		 = iDummy;
	      
	      strcpy	(tmpVar, "<lz>");
	      sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	      param->X[2]	 = iDummy/gridSize[2];
	      ln[2]		 = iDummy;
	      
	      strcpy	(tmpVar, "<lt>");
	      sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	      param->X[3]	 = iDummy/gridSize[3];
	      ln[3]		 = iDummy;
	      
	      printfQuda("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
	      printfQuda("Subvolume:\t%ix%ix%ix%i\n", param->X[0], param->X[1], param->X[2], param->X[3]);
	      
	      free(lime_data);
	    }
	}
      
      // Read 1 byte to set file-pointer to start of binary data
      
#ifdef	MULTI_GPU
      lime_data_size=1;
      limeReaderReadData(&dummy,&lime_data_size,limereader);
      offset = ftell(fid)-1;
#endif
      limeDestroyReader(limereader);
     
  
  MPI_Bcast(&error_occured,1,MPI_INT,0,MPI_COMM_WORLD);
  
  if(error_occured)
    errorQuda("unknown error");
  
  if(isDouble == 32)
    {
      errorQuda("Unsupported precision 32 bits");
    }

  nvh = (param->X[0] * param->X[1] * param->X[2] * param->X[3]) / 2;
  
  for(int dir = 0; dir < 4; dir++)
    {
      resEvn[dir] = gauge[dir];
      resOdd[dir] = gauge[dir] + nvh*18;
    } 

  lvol = ln[0]*ln[1]*ln[2]*ln[3];
  
  if(lvol==0)
    {
      errorQuda("Zero volume");
    }
 
  int x4start = comm_coords(default_topo)[3]*param->X[3];
  int x4end   = x4start + param->X[3];
  int x3start = comm_coords(default_topo)[2]*param->X[2];
  int x3end   = x3start + param->X[2];
  int x2start = comm_coords(default_topo)[1]*param->X[1];
  int x2end   = x2start + param->X[1];
  int x1start = comm_coords(default_topo)[0]*param->X[0];
  int x1end   = x1start + param->X[0];
  
#ifdef	MULTI_GPU
  sizes[0]	 = ln[3];
  sizes[1]	 = ln[2];
  sizes[2]	 = ln[1];
  sizes[3]	 = ln[0];
  sizes[4]	 = 4*3*3*2;
  lsizes[0]	 = param->X[3];
  lsizes[1]	 = param->X[2];
  lsizes[2]	 = param->X[1];
  lsizes[3]	 = param->X[0];
  lsizes[4]	 = sizes[4];
  starts[0]	 = comm_coords(default_topo)[3]*param->X[3];
  starts[1]	 = comm_coords(default_topo)[2]*param->X[2];
  starts[2]	 = comm_coords(default_topo)[1]*param->X[1];
  starts[3]	 = comm_coords(default_topo)[0]*param->X[0];
  starts[4]	 = 0;
  
  strcpy(tmpVar, "native");
	
  MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, tmpVar, MPI_INFO_NULL);

  //load time-slice by time-slice:

  chunksize = 4*3*3*sizeof(qcd_complex_16);
  ftmp = (char*) malloc(((unsigned int) chunksize*nvh*2));
  
  if(ftmp == NULL)
    {
      errorQuda(" Out of memory, couldn't alloc %u bytes", (unsigned int) (chunksize*nvh*2));
     
    }
  else
    printf("%d bytes reserved for gauge fields (%d x %u)\n", chunksize*nvh*2, chunksize, nvh*2);
  
  if(MPI_File_read_all(mpifid, ftmp, 4*3*3*2*nvh*2, MPI_DOUBLE, &status) == 1)
    printf("Error in MPI_File_read_all\n");
  
  if(4*3*3*2*nvh*2*sizeof(double) > 2147483648)
    {
      printf  ("File too large. At least %lu processes are needed to read this file properly.\n", (4*3*3*2*nvh*2*sizeof(double)/2147483648)+1);
      printf  ("If some results are wrong, try increasing the number of MPI processes.\n");
    }
  
  if(!qcd_isBigEndian())
    qcd_swap_8((double*) ftmp,2*4*3*3*nvh*2);
#else
  ftmp = (double*)malloc(lvol*72*sizeof(double));
  
  if(ftmp == NULL)
    {
      errorQuda("Error, could not alloc ftmp");
    }
  
  iread	 = fread(ftmp, sizeof(double), 72*lvol, fid);
  
  if	(iread != 72*lvol)
    {
      errorQuda("Error, could not read proper amount of data");
    }
  
  fclose(fid);
  
  if(!qcd_isBigEndian())      
    qcd_swap_8	((double*) ftmp,72*lvol);
#endif
  
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  
  iy	 = 0;
  ixh	 = 0;

#ifdef	MULTI_GPU
  i = 0;
#endif

  int lx1 = 0, lx2 = 0, lx3 = 0, lx4 = 0;

  for(x4 = x4start; x4 < x4end; x4++) 
    {  // t
      for(x3 = x3start; x3 < x3end; x3++) 
	{  // z
	  for(x2 = x2start; x2 < x2end; x2++) 
	    {  // y
	      for(x1 = x1start; x1 < x1end; x1++) 
		{  // x
		  int oddBit	 = (x1+x2+x3+x4) & 1;
		  
#ifdef	MULTI_GPU
		  iy		 = ((x1-x1start)+(x2-x2start)*param->X[0]+(x3-x3start)*param->X[1]*param->X[0]+(x4-x4start)*param->X[0]*param->X[1]*param->X[2])/2;
#else
		  iy		 = x1+x2*param->X[0]+x3*param->X[1]*param->X[0]+x4*param->X[0]*param->X[1]*param->X[2];
#endif
		  for(mu = 0; mu < 4; mu++)
		    {  
#ifdef	MULTI_GPU
		      memcpy(U, &(ftmp[i]), 18*sizeof(double));    
		      if(oddBit)
			memcpy(&(resOdd[mu][18*iy]), U, 18*sizeof(double));
		      else
			memcpy(&(resEvn[mu][18*iy]), U, 18*sizeof(double));
		      i	+= 144;
#else
		      ixh		 = (lx1+lx2*param->X[0]+lx3*param->X[0]*param->X[1]+lx4*param->X[0]*param->X[1]*param->X[2])/2;
		      idx = mu*18 + 72*iy;
		      double *links_ptr = ftmp+idx;
		      memcpy(U, links_ptr, 18*sizeof(double));
		      if(oddBit)
			memcpy(resOdd[mu]+18*ixh, U, 18*sizeof(double));
		      else
			memcpy(resEvn[mu]+18*ixh, U, 18*sizeof(double));
#endif
		    } 
		  ++lx1;
		} 
	      lx1 = 0;
	      ++lx2;
	    } 
	  lx2 = 0;
	  ++lx3;				
	} 
      lx3 = 0;
      ++lx4;
    }
  free(ftmp);
  
#ifdef	MULTI_GPU
  MPI_File_close(&mpifid);
#endif  
  
  //	Apply BC here
  
  //  applyGaugeFieldScaling<double>((double**)gauge, nvh, param);
  
  return;
}


static void read_custom_binary_gauge_field_smeared(double **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4])
{
  FILE *fid;
  int x1, x2, x3, x4, ln[4] = { 0, 0, 0, 0 };
  unsigned long long	lvol, ixh, iy, mu;
  char tmpVar[20];
  double *U = (double*)malloc(18*sizeof(double));
  double *resEvn[4], *resOdd[4];
  int nvh, iDummy;

  LimeReader *limereader;
  char *lime_type, *lime_data;
  n_uint64_t lime_data_size;
  char dummy;
  int  isDouble;
  int  error_occured=0;
  double dDummy;


#ifdef	MULTI_GPU
  MPI_Offset offset = 0;
  MPI_Datatype subblock;  //MPI-type, 5d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  unsigned long i=0;
  unsigned short chunksize;
  char *ftmp=NULL;
#else
  double *ftmp=NULL;
  unsigned long long	iread, idx;
#endif

  fid=fopen(fname,"r");
  if(fid==NULL)
    {
      errorQuda("Error reading configuration! Could not open path for reading");
    }
	
  if ((limereader = limeCreateReader(fid))==NULL)
    {
      errorQuda("Could not create limeReader");
    }
	
  while(limeReaderNextRecord(limereader)      != LIME_EOF )
    {
      lime_type				       = limeReaderType(limereader);
      
      if(strcmp(lime_type,"ildg-binary-data") == 0)
	break;
      
	  
      if(strcmp(lime_type,"ildg-format")==0)
	{
	  lime_data_size = limeReaderBytes(limereader);
	  lime_data = (char * )malloc(lime_data_size);
	  limeReaderReadData((void *)lime_data,&lime_data_size, limereader);
	  
	  strcpy	(tmpVar, "<precision>");
	  sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&isDouble);    
	  printfQuda("Precision:\t%i bit\n",isDouble);
	  
	  strcpy	(tmpVar, "<lx>");
	  sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	  param->X[0]	 = iDummy/gridSize[0];
	  ln[0]		 = iDummy;
	      
	  strcpy	(tmpVar, "<ly>");
	  sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	  param->X[1]	 = iDummy/gridSize[1];
	  ln[1]		 = iDummy;
	      
	  strcpy	(tmpVar, "<lz>");
	  sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	  param->X[2]	 = iDummy/gridSize[2];
	  ln[2]		 = iDummy;
	      
	  strcpy	(tmpVar, "<lt>");
	  sscanf(qcd_getParam(tmpVar,lime_data, lime_data_size),"%i",&iDummy);
	  param->X[3]	 = iDummy/gridSize[3];
	  ln[3]		 = iDummy;
	  
	  printfQuda("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
	  printfQuda("Subvolume:\t%ix%ix%ix%i\n", param->X[0], param->X[1], param->X[2], param->X[3]);
	  
	  free(lime_data);
	}
    }
  
  // Read 1 byte to set file-pointer to start of binary data
  
#ifdef	MULTI_GPU
      lime_data_size=1;
      limeReaderReadData(&dummy,&lime_data_size,limereader);
      offset = ftell(fid)-1;
#endif
      limeDestroyReader(limereader);
     
  
  MPI_Bcast(&error_occured,1,MPI_INT,0,MPI_COMM_WORLD);
  
  if(error_occured)
    errorQuda("unknown error");
  
  if(isDouble == 32)
    {
      errorQuda("Unsupported precision 32 bits");
    }

  nvh = (param->X[0] * param->X[1] * param->X[2] * param->X[3]) / 2;
  
  for(int dir = 0; dir < 4; dir++)
    {
      resEvn[dir] = gauge[dir];
      resOdd[dir] = gauge[dir] + nvh*18;
    } 

  lvol = ln[0]*ln[1]*ln[2]*ln[3];
  
  if(lvol==0)
    {
      errorQuda("Zero volume");
    }


  int x4start = comm_coords(default_topo)[3]*param->X[3];
  int x4end   = x4start + param->X[3];
  int x3start = comm_coords(default_topo)[2]*param->X[2];
  int x3end   = x3start + param->X[2];
  int x2start = comm_coords(default_topo)[1]*param->X[1];
  int x2end   = x2start + param->X[1];
  int x1start = comm_coords(default_topo)[0]*param->X[0];
  int x1end   = x1start + param->X[0];
  
#ifdef	MULTI_GPU
  sizes[0]	 = ln[3];
  sizes[1]	 = ln[2];
  sizes[2]	 = ln[1];
  sizes[3]	 = ln[0];
  sizes[4]	 = 4*3*3*2;
  lsizes[0]	 = param->X[3];
  lsizes[1]	 = param->X[2];
  lsizes[2]	 = param->X[1];
  lsizes[3]	 = param->X[0];
  lsizes[4]	 = sizes[4];
  starts[0]	 = comm_coords(default_topo)[3]*param->X[3];
  starts[1]	 = comm_coords(default_topo)[2]*param->X[2];
  starts[2]	 = comm_coords(default_topo)[1]*param->X[1];
  starts[3]	 = comm_coords(default_topo)[0]*param->X[0];
  starts[4]	 = 0;

 
  
  strcpy(tmpVar, "native");
	
  MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, tmpVar, MPI_INFO_NULL);

  //load time-slice by time-slice:

  chunksize = 4*3*3*sizeof(qcd_complex_16);
  ftmp = (char*) malloc(((unsigned int) chunksize*nvh*2));
  
  if(ftmp == NULL)
    {
      errorQuda(" Out of memory, couldn't alloc %u bytes", (unsigned int) (chunksize*nvh*2));
     
    }
  else
    printf("%d bytes reserved for gauge fields (%d x %u)\n", chunksize*nvh*2, chunksize, nvh*2);
  
  if(MPI_File_read_all(mpifid, ftmp, 4*3*3*2*nvh*2, MPI_DOUBLE, &status) == 1)
    printf("Error in MPI_File_read_all\n");
  
  if(4*3*3*2*nvh*2*sizeof(double) > 2147483648)
    {
      printf  ("File too large. At least %lu processes are needed to read this file properly.\n", (4*3*3*2*nvh*2*sizeof(double)/2147483648)+1);
      printf  ("If some results are wrong, try increasing the number of MPI processes.\n");
    }
  
  if(!qcd_isBigEndian())
    qcd_swap_8((double*) ftmp,2*4*3*3*nvh*2);
#else
  ftmp = (double*)malloc(lvol*72*sizeof(double));
  
  if(ftmp == NULL)
    {
      errorQuda("Error, could not alloc ftmp");
    }
  
  iread	 = fread(ftmp, sizeof(double), 72*lvol, fid);
  
  if	(iread != 72*lvol)
    {
      errorQuda("Error, could not read proper amount of data");
    }
  
  fclose(fid);
  
  if(!qcd_isBigEndian())      
    qcd_swap_8	((double*) ftmp,72*lvol);
#endif
  
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  
  iy	 = 0;
  ixh	 = 0;

#ifdef	MULTI_GPU
  i = 0;
#endif

  int lx1 = 0, lx2 = 0, lx3 = 0, lx4 = 0;

  for(x4 = x4start; x4 < x4end; x4++) 
    {  // t
      for(x3 = x3start; x3 < x3end; x3++) 
	{  // z
	  for(x2 = x2start; x2 < x2end; x2++) 
	    {  // y
	      for(x1 = x1start; x1 < x1end; x1++) 
		{  // x
		  int oddBit	 = (x1+x2+x3+x4) & 1;
		  
#ifdef	MULTI_GPU
		  iy		 = ((x1-x1start)+(x2-x2start)*param->X[0]+(x3-x3start)*param->X[1]*param->X[0]+(x4-x4start)*param->X[0]*param->X[1]*param->X[2])/2;
#else
		  iy		 = x1+x2*param->X[0]+x3*param->X[1]*param->X[0]+x4*param->X[0]*param->X[1]*param->X[2];
#endif
		  for(mu = 0; mu < 4; mu++)
		    {  
#ifdef	MULTI_GPU
		      memcpy(U, &(ftmp[i]), 18*sizeof(double));    
		      if(oddBit)
			memcpy(&(resOdd[mu][18*iy]), U, 18*sizeof(double));
		      else
			memcpy(&(resEvn[mu][18*iy]), U, 18*sizeof(double));
		      i	+= 144;
#else
		      ixh		 = (lx1+lx2*param->X[0]+lx3*param->X[0]*param->X[1]+lx4*param->X[0]*param->X[1]*param->X[2])/2;
		      idx = mu*18 + 72*iy;
		      double *links_ptr = ftmp+idx;
		      memcpy(U, links_ptr, 18*sizeof(double));
		      if(oddBit)
			memcpy(resOdd[mu]+18*ixh, U, 18*sizeof(double));
		      else
			memcpy(resEvn[mu]+18*ixh, U, 18*sizeof(double));
#endif
		    } 
		  ++lx1;
		} 
	      lx1 = 0;
	      ++lx2;
	    } 
	  lx2 = 0;
	  ++lx3;				
	} 
      lx3 = 0;
      ++lx4;
    }
  free(ftmp);
  
#ifdef	MULTI_GPU
  MPI_File_close(&mpifid);
#endif  
      


}


static void read_custom_binary_gauge_field_andreas(double **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4])
{
  FILE *fid;
  int ln[4] = { 0, 0, 0, 0 };
  unsigned long long	lvol;
  char tmpVar[20];

  MPI_Offset offset = 0;
  MPI_Datatype subblock;  //MPI-type, 5d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[6], lsizes[6], starts[6];
  unsigned long i=0;
  unsigned short chunksize;
  char *ftmp=NULL;

  fid=fopen(fname,"rb");
  if(fid==NULL)
    {
      errorQuda("Error reading configuration! Could not open path for reading");
    }
	

  for(int i = 0 ; i < 4 ; i++)
    ln[i] = param->X[i]*gridSize[i];

  printfQuda("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
  printfQuda("Subvolume:\t%ix%ix%ix%i\n", param->X[0], param->X[1], param->X[2], param->X[3]);

  
  lvol = ln[0]*ln[1]*ln[2]*ln[3];
  
  if(lvol==0)
    {
      errorQuda("Zero volume");
    }


  sizes[0] = 3*3;
  sizes[1] = ln[0];
  sizes[2] = ln[1];
  sizes[3] = ln[2];
  sizes[4] = ln[3];
  sizes[5] = 4*2;

  lsizes[0] = 3*3;
  lsizes[1] = param->X[0];
  lsizes[2] = param->X[1];
  lsizes[3] = param->X[2];
  lsizes[4] = param->X[3];
  lsizes[5] = 4*2;
  
  starts[0] = 0;
  starts[1] = comm_coords(default_topo)[0]*param->X[0];
  starts[2] = comm_coords(default_topo)[1]*param->X[1];
  starts[3] = comm_coords(default_topo)[2]*param->X[2];
  starts[4] = comm_coords(default_topo)[3]*param->X[3];
  starts[5] = 0;


  strcpy(tmpVar, "native");

  offset = ftell(fid);
	
  MPI_Type_create_subarray(6,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, tmpVar, MPI_INFO_NULL);

  //load time-slice by time-slice:

  chunksize = 4*3*3*sizeof(qcd_complex_16);
  ftmp = (char*) malloc(((unsigned int) chunksize*lvol));
  
  if(ftmp == NULL)
    {
      errorQuda(" Out of memory, couldn't alloc %u bytes", (unsigned int) (chunksize*lvol));
     
    }
  else
    printf("%ld bytes reserved for gauge fields (%d x %ld)\n", chunksize*lvol, chunksize, lvol);
  
  if(MPI_File_read_all(mpifid, ftmp, 4*3*3*2*lvol, MPI_DOUBLE, &status) == 1)
    printf("Error in MPI_File_read_all\n");
  
  if(4*3*3*2*lvol*sizeof(double) > 2147483648)
    {
      printf  ("File too large. At least %lu processes are needed to read this file properly.\n", (4*3*3*2*lvol*sizeof(double)/2147483648)+1);
      printf  ("If some results are wrong, try increasing the number of MPI processes.\n");
    }
  
  if(!qcd_isBigEndian())
    qcd_swap_8((double*) ftmp,2*4*3*3*lvol);
  
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  



  qcd_complex_16 *p_gauge[4];

  for(int i = 0 ; i < 4 ; i++)
    p_gauge[i] = (qcd_complex_16*) gauge[i];


  int lx1 = 0, lx2 = 0, lx3 = 0, lx4 = 0;

  for(int ic1 = 0 ; ic1 < 3 ; ic1++)
    for(int ic2 = 0 ; ic2 < 3 ; ic2++)
      for(int ix = 0; ix < param->X[0]; ix++){
	for(int iy = 0; iy < param->X[1]; iy++){
	  for(int iz = 0; iz < param->X[2]; iz++){
	    for(int it = 0; it < param->X[3]; it++){  
		  
	      for(int mu = 0; mu < 4; mu++)
		{  
		  int pos = ix + iy*param->X[0] + iz*param->X[0]*param->X[1] + it*param->X[0]*param->X[1]*param->X[2];
		  p_gauge[mu][pos*3*3+ic1*3+ic2] = *((qcd_complex_16*) &(ftmp[i]));
		  i += sizeof(qcd_complex_16);
		} 
	    } 
	  } 
	} 
      }

  free(ftmp);
  
#ifdef	MULTI_GPU
  MPI_File_close(&mpifid);
#endif  
      


}


void readLimeGauge(void **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4]){

  if( param->cpu_prec == QUDA_DOUBLE_PRECISION )
    read_custom_binary_gauge_field ( (double**) gauge, fname, param, inv_param, gridSize);
  else
    errorQuda("Dont support reading confs lime single precision");
}

void readLimeGaugeSmeared(void **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4]){

  if( param->cpu_prec == QUDA_DOUBLE_PRECISION )
    read_custom_binary_gauge_field_smeared ( (double**) gauge, fname, param, inv_param, gridSize);
  else
    errorQuda("Dont support reading confs lime single precision");
}

void readGaugeAndreas(void **gauge, char *fname, QudaGaugeParam *param, QudaInvertParam *inv_param, int gridSize[4]){

  if( param->cpu_prec == QUDA_DOUBLE_PRECISION )
    read_custom_binary_gauge_field_andreas ( (double**) gauge, fname, param, inv_param, gridSize);
  else
    errorQuda("Dont support reading confs lime single precision");
}


void applyBoundaryCondition(void **gauge, int Vh ,QudaGaugeParam *gauge_param){

  if( gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION )
    applyGaugeFieldScaling<double>((double**)gauge, Vh, gauge_param); 
  else
    errorQuda("boundary condition application implement only for double precision");
}
