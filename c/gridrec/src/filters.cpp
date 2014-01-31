/** File filters.c  Revised 7/7/98   **/
/** Filters redefined so that x=1/2 is the usual
	 cut-off frequency, instead of x=1 as formerly. */
#include "grid.h"

#define LEN_FTBL 7	/* Number of entries in ftbl[] below */

float none(float x){	/* no filter */
	return 1;
}

float shlo(float x){	/* Shepp-Logan filter */
	return abs(sin(pi*x)/pi);
}

float hann(float x){	/* Hann filter */
	return abs(x)*0.5*(1.+cos(2*pi*x));
}

float hamm(float x){	/* Hamming filter */
	return abs(x)*(0.54+0.46*cos(2*pi*x));
}

float ramp(float x){	/* Ramp filter */
	return abs(x);
}

struct {char* name; float (*fp)(float);} fltbl[]= {
        {"none",none},
	{"shlo",shlo},		/* The default choice */
	{"shepp",shlo},
	{"hann",hann},
	{"hamm",hamm},
	{"hamming",hamm},
	{"ramp",ramp},
	{"ramlak",ramp}
};	

float (*get_filter(char *name))(float) {
	int i;
	for(i=0;i<LEN_FTBL;i++){
		if(!strcmp(name,fltbl[i].name)){
			return fltbl[i].fp;
		}
	}
	/* If no match, set default choice */
	printf(" Using default filter: %s.\n",
			fltbl[0].name);
	strcpy(name,fltbl[0].name);
	return fltbl[0].fp;   
}  /*** End get_filter() ***/
	
