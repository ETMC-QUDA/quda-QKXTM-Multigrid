
static void	merge			(double *sort1, int *idx1, int n1, double *sort2, int *idx2, int n2, bool inverse)
{
  int	i1=0, i2=0;
  int	*ord;
  double	*result;

  ord    = (int *)    malloc(sizeof(int)   *(n1+n2)); 
  result = (double *) malloc(sizeof(double)*(n1+n2)); 

  for	(int i=0; i<(n1+n2); i++)
    {
      if	((sort1[i1] >= sort2[i2]) != inverse) {	//LOGICAL XOR
	result[i] = sort1[i1];
	ord[i] = idx1[i1];
	i1++;
      } else {
	result[i] = sort2[i2];
	ord[i] = idx2[i2];
	i2++;
      }

      if	(i1 == n1) {
	for	(int j=i+1; j<(n1+n2); j++,i2++)
	  {
	    result[j] = sort2[i2];
	    ord[j] = idx2[i2];
	  }
	i = n1+n2;
      } else if (i2 == n2) {
	for	(int j=i+1; j<(n1+n2); j++,i1++)
	  {
	    result[j] = sort1[i1];
	    ord[j] = idx1[i1];
	  }
	i = i1+i2;
      }
    }

  for	(int i=0;i<n1;i++)
    {
      idx1[i] = ord[i];
      sort1[i] = result[i];
    }

  for	(int i=0;i<n2;i++)
    {
      idx2[i] = ord[i+n1];
      sort2[i] = result[i+n1];
    }

  free (ord);
  free (result);
}

static void	sort			(double *unsorted, int n, bool inverse, int *idx)
{
  if	(n <= 1)
    return;

  int	n1, n2;

  n1 = n>>1;
  n2 = n-n1;

  double	*unsort1 = unsorted;
  double	*unsort2 = (double *) ((char *) unsorted + n1*sizeof(double));
  int	*idx1	 = idx;
  int	*idx2	 = (int *)    ((char *) idx	 + n1*sizeof(int));

  sort	(unsort1, n1, inverse, idx1);
  sort	(unsort2, n2, inverse, idx2);

  merge	(unsort1, idx1, n1, unsort2, idx2, n2, inverse);
}

static void	mergeAbs		(double *sort1, int *idx1, int n1, double *sort2, int *idx2, int n2, bool inverse)
{
  int	i1=0, i2=0;
  int	*ord;
  double	*result;

  ord    = (int *)    malloc(sizeof(int)   *(n1+n2)); 
  result = (double *) malloc(sizeof(double)*(n1+n2)); 

  for	(int i=0; i<(n1+n2); i++)
    {
      if	((fabs(sort1[i1]) >= fabs(sort2[i2])) != inverse) {	//LOGICAL XOR
	result[i] = sort1[i1];
	ord[i] = idx1[i1];
	i1++;
      } else {
	result[i] = sort2[i2];
	ord[i] = idx2[i2];
	i2++;
      }

      if	(i1 == n1) {
	for	(int j=i+1; j<(n1+n2); j++,i2++)
	  {
	    result[j] = sort2[i2];
	    ord[j] = idx2[i2];
	  }
	i = n1+n2;
      } else if (i2 == n2) {
	for	(int j=i+1; j<(n1+n2); j++,i1++)
	  {
	    result[j] = sort1[i1];
	    ord[j] = idx1[i1];
	  }
	i = i1+i2;
      }
    }

  for	(int i=0;i<n1;i++)
    {
      idx1[i] = ord[i];
      sort1[i] = result[i];
    }

  for	(int i=0;i<n2;i++)
    {
      idx2[i] = ord[i+n1];
      sort2[i] = result[i+n1];
    }

  free (ord);
  free (result);
}

static void	sortAbs			(double *unsorted, int n, bool inverse, int *idx)
{
  if	(n <= 1)
    return;

  int	n1, n2;

	n1 = n>>1;
	n2 = n-n1;

	double	*unsort1 = unsorted;
	double	*unsort2 = (double *) ((char *) unsorted + n1*sizeof(double));
	int	*idx1	 = idx;
	int	*idx2	 = (int *)    ((char *) idx	 + n1*sizeof(int));

	sortAbs	(unsort1, n1, inverse, idx1);
	sortAbs	(unsort2, n2, inverse, idx2);

	mergeAbs(unsort1, idx1, n1, unsort2, idx2, n2, inverse);
}


void quicksort(int n, double arr[], int idx[]){
  double v, td;
  int i, j, l, r, ti, tos, stack[32];

  l = 0; r = n-1; tos = -1;
  for (;;){
    while (r > l){
      v = arr[r]; i = l; j = r-1;
      for (;;){
        while (arr[i] < v) i ++;
        /* j > l prevents underflow */
        while (arr[j] >= v && j > l) j --;
        if (i >= j) break;
        td = arr[i]; arr[i] = arr[j]; arr[j] = td;
        ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
      }
      td = arr[i]; arr[i] = arr[r]; arr[r] = td;
      ti = idx[i]; idx[i] = idx[r]; idx[r] = ti;
      if (i-l > r-i){
        stack[++tos] = l; stack[++tos] = i-1; l = i+1;
      }
      else{
        stack[++tos] = i+1; stack[++tos] = r; r = i-1;
      }
      if(tos > 31) {
        fprintf(stderr,"Error in quicksort! Aborting...!");fflush(stderr);
        exit(31);
      }
    }
    if (tos == -1) break;
    r = stack[tos--]; l = stack[tos--];
  }
}
