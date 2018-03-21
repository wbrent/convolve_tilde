/*

convolve~

Copyright 2010 William Brent

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

version 0.11, March 17, 2018

- using FFTW as of version 0.11

*/

#include "m_pd.h"
#include "fftw3.h"
#include <math.h>
#define MINWIN 64
#define DEFAULTWIN 2048
#define NUMBARKBOUNDS 25

t_float barkBounds[] = {0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500};

static t_class *convolve_tilde_class;

typedef struct _convolve_tilde 
{
    t_object x_obj;
	t_symbol *x_objSymbol;
    t_symbol *x_arrayName;
    t_word *x_vec;
	t_clock *x_clock;
    t_sample *x_irSignalEq;
    int x_startupFlag;
    int x_arraySize;
    int x_numParts;
    t_float x_sr;
    t_float x_n;
    int x_dspTick;
    int x_bufferLimit;
    int x_window;
    int x_windowDouble;
    t_float x_ampScalar;
	t_sample *x_signalBuf;
	t_sample *x_signalBufPadded;
	t_sample *x_invOutFftwOut;
	t_sample *x_nonOverlappedOutput;	
	t_sample *x_finalOutput;
	
    fftwf_complex *x_irFreqDomData;
	fftwf_complex *x_liveFreqDomData;
    fftwf_complex *x_sigBufPadFftwOut;
    fftwf_complex *x_invOutFftwIn;
	fftwf_plan x_sigBufPadFftwPlan;
	fftwf_plan x_invOutFftwPlan;

    t_float x_f;
    
} convolve_tilde;


/* ------------------------ convolve~ -------------------------------- */

static void convolve_tilde_analyze(convolve_tilde *x, t_symbol *arrayName)
{
    t_garray *arrayPtr;
    int i, j, oldNonOverlappedSize, newNonOverlappedSize;
    t_float *fftwIn;
    fftwf_complex *fftwOut;
	fftwf_plan fftwPlan;

	if(x->x_numParts>0)
		oldNonOverlappedSize = (x->x_numParts+1)*x->x_windowDouble;
	else
		oldNonOverlappedSize = 0;
	
    // if this call to _analyze() is issued from _eq(), the incoming arrayName will match x->x_arrayName.
    // if incoming arrayName doesn't match x->x_arrayName, load arrayName and dump its samples into x_irSignalEq
    if(arrayName->s_name != x->x_arrayName->s_name || x->x_startupFlag)
    {
    	int oldArraySize;

		x->x_startupFlag = 0;

    	oldArraySize = x->x_arraySize;
    	
		if(!(arrayPtr = (t_garray *)pd_findbyclass(arrayName, garray_class)))
		{
			if(*arrayName->s_name)
			{
				pd_error(x, "%s: no array named %s", x->x_objSymbol->s_name, arrayName->s_name);

				// resize x_irSignalEq back to 0
				x->x_irSignalEq = (t_sample *)t_resizebytes(
					x->x_irSignalEq,
					oldArraySize*sizeof(t_sample),
					0
				);
		
				x->x_arraySize = 0;
				x->x_vec = 0;
				return;
			}
		}
		else if(!garray_getfloatwords(arrayPtr, &x->x_arraySize, &x->x_vec))
		{
			pd_error(x, "%s: bad template for %s", x->x_objSymbol->s_name, arrayName->s_name);

			// resize x_irSignalEq back to 0
			x->x_irSignalEq = (t_sample *)t_resizebytes(
				x->x_irSignalEq,
				oldArraySize*sizeof(t_sample),
				0
			);

			x->x_arraySize = 0;
			x->x_vec = 0;
			return;
		}
		else
			x->x_arrayName = arrayName;
		
	    // resize x_irSignalEq
		x->x_irSignalEq = (t_sample *)t_resizebytes(
			x->x_irSignalEq,
			oldArraySize*sizeof(t_sample),
			x->x_arraySize*sizeof(t_sample)
		);
	
		// since this is first analysis of arrayName, load it into x_irSignalEq
		for(i=0; i<x->x_arraySize; i++)
			x->x_irSignalEq[i] = x->x_vec[i].w_float;
    }
	else
	{
		t_garray *thisArrayPtr;
		t_word *thisVec;
		int thisArraySize;

		// if we want to assume that a 2nd call to analyze() with the same array name is safe (and we don't have to reload x_vec or update x_arraySize), we have to do some careful safety checks to make sure that the size of x_arrayName hasn't changed since the last time this was called. If it has, we should just abort for now.
		thisArraySize = 0;
		
		thisArrayPtr = (t_garray *)pd_findbyclass(arrayName, garray_class);
		garray_getfloatwords(thisArrayPtr, &thisArraySize, &thisVec);
		
		if(thisArraySize != x->x_arraySize)
		{
			pd_error(x, "%s: size of array %s has changed since previous analysis...aborting. Reload %s with previous IR contents or analyze another array.", x->x_objSymbol->s_name, arrayName->s_name, arrayName->s_name);
			return;
		}
	}

    // count how many partitions there will be for this IR
    x->x_numParts = 0;
    while((x->x_numParts*x->x_window) < x->x_arraySize)
    	x->x_numParts++;

	newNonOverlappedSize = (x->x_numParts+1)*x->x_windowDouble;
    
    // resize time-domain buffer
    x->x_nonOverlappedOutput = (t_sample *)t_resizebytes(
    	x->x_nonOverlappedOutput,
    	oldNonOverlappedSize*sizeof(t_sample),
    	newNonOverlappedSize*sizeof(t_sample)
    );

	// clear time-domain buffer
    for(i=0; i<newNonOverlappedSize; i++)
    	x->x_nonOverlappedOutput[i] = 0.0;

	// free x_irFreqDomData/x_liveFreqDomData and re-alloc to new size based on x_numParts
	fftwf_free(x->x_irFreqDomData);
	fftwf_free(x->x_liveFreqDomData);
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_numParts*(x->x_window+1));
	x->x_liveFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_numParts*(x->x_window+1));
	
	// clear x_liveFreqDomData
    for(i=0; i<x->x_numParts*(x->x_window+1); i++)
    {
    	x->x_liveFreqDomData[i][0] = 0.0;
    	x->x_liveFreqDomData[i][1] = 0.0;
    }

    // set up FFTW input buffer
    fftwIn = (t_float *)t_getbytes(x->x_windowDouble * sizeof(t_float));

	// set up the FFTW output buffer. It is N/2+1 elements long for an N-point r2c FFT
	// fftwOut[i][0] and fftwOut[i][1] refer to the real and imaginary parts of bin i
	fftwOut = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	// FFT plan
	fftwPlan = fftwf_plan_dft_r2c_1d(x->x_windowDouble, fftwIn, fftwOut, FFTW_ESTIMATE);

	// we're supposed to initialize the input array after we create the plan
	for(i=0; i<x->x_windowDouble; i++)
		fftwIn[i] = 0.0;

    // take FFTs of partitions, and store in x_irFreqDomData as chunks of
    // window+1 complex arrays
    for(i=0; i<x->x_numParts; i++)
    {
    	int startSpec, startVec;
    	
    	startSpec = i*(x->x_window+1);
    	startVec = i*x->x_window;
    	
    	// we are analyzing partitions of x_irSignalEq, in case there has been any EQ
		for(j=0; j<x->x_window && (startVec+j)<x->x_arraySize; j++)
			fftwIn[j] = x->x_irSignalEq[startVec+j];
		
		// zero pad
		for(; j<x->x_windowDouble; j++)
			fftwIn[j] = 0.0;
			
		// execute FFT
		fftwf_execute(fftwPlan);

		// copy freq domain data from fft output buffer into
		// larger IR freq domain data buffer
		for(j=0; j<x->x_window+1; j++)
		{
			x->x_irFreqDomData[startSpec+j][0] = fftwOut[j][0];
			x->x_irFreqDomData[startSpec+j][1] = fftwOut[j][1];
		}
	}
				
    t_freebytes(fftwIn, (x->x_windowDouble)*sizeof(t_float));
	fftwf_free(fftwOut);
	fftwf_destroy_plan(fftwPlan); 
	
	post("%s: analysis of IR array %s complete. Array size: %i. Partitions: %i.", x->x_objSymbol->s_name, x->x_arrayName->s_name, x->x_arraySize, x->x_numParts);
}


static void convolve_tilde_eq(convolve_tilde *x, t_symbol *s, int argc, t_atom *argv)
{
    int i, j, windowTriple, windowTripleHalf, *barkBinBounds;
    t_float *fftwIn, *eqArray;
    fftwf_complex *fftwOut;
	fftwf_plan fftwForwardPlan, fftwInversePlan;

	// if no array has been analyzed yet, we can't do the EQ things below
	if(x->x_arrayName == gensym("NOARRAYSPECIFIED"))
		pd_error(x, "%s: no IR array has been analyzed", x->x_objSymbol->s_name);
	else
	{
	// we'll pad with x->x_arraySize zeros before and after the IR signal
	windowTriple = x->x_arraySize*3;

	// FFTW documentation says the output of a r2c forward transform is floor(N/2)+1
	windowTripleHalf = floor(windowTriple/(t_float)2)+1;
	
	eqArray = (t_float *)getbytes(windowTripleHalf*sizeof(t_float));

	// at this point, if we took in 24 bark band scalars instead of the insane x->x_arraySize*3 scalars, we could find the bin bounds for each of the Bark bands and fill eqArray with those.

	// array to hold bin number of each of the 25 bark bounds, plus one more for Nyquist
	barkBinBounds = (int *)getbytes((NUMBARKBOUNDS+1)*sizeof(int));
	
	for(i=0; i<NUMBARKBOUNDS; i++)
		barkBinBounds[i] = floor((barkBounds[i]*windowTriple)/x->x_sr);

	// the upper bound should be the Nyquist bin
	barkBinBounds[NUMBARKBOUNDS] = windowTripleHalf-1;

	if(argc != NUMBARKBOUNDS)
		post("%s: WARNING: \"eq\" message should contain %i frequency band scalars", x->x_objSymbol->s_name, NUMBARKBOUNDS);

	// need to check that argc == NUMBARKBOUNDS
	for(i=0; i<NUMBARKBOUNDS && i<argc; i++)
		for(j=barkBinBounds[i]; j<barkBinBounds[i+1]; j++)
			eqArray[j] = (atom_getfloat(argv+i)<0.0)?0:atom_getfloat(argv+i);
	
	// if there were too few arguments coming in (argc<NUMBARKBOUNDS), fill out the remaining bin scalars in eqArray with 1.0
	for(; i<NUMBARKBOUNDS; i++)
		for(j=barkBinBounds[i]; j<barkBinBounds[i+1]; j++)
			eqArray[j] = 1.0;
			
	// need to set the Nyquist bin too, since the j loop doesn't go to the upper bound inclusive
	eqArray[windowTripleHalf-1] = (atom_getfloat(argv+(NUMBARKBOUNDS-1))<0.0)?0:atom_getfloat(argv+(NUMBARKBOUNDS-1));
	
    // set up FFTW input buffer
    fftwIn = (t_float *)t_getbytes(windowTriple * sizeof(t_float));

	// set up the FFTW output buffer
	fftwOut = (fftwf_complex *)fftwf_alloc_complex(windowTripleHalf);

	// forward FFT plan
	fftwForwardPlan = fftwf_plan_dft_r2c_1d(windowTriple, fftwIn, fftwOut, FFTW_ESTIMATE);

	// inverse FFT plan
	fftwInversePlan = fftwf_plan_dft_c2r_1d(windowTriple, fftwOut, fftwIn, FFTW_ESTIMATE);

	// fill input buffer with zeros
	for(i=0; i<windowTriple; i++)
		fftwIn[i] = 0.0;

	// place actual IR signal in center of buffer
	for(i=0; i<x->x_arraySize; i++)
		fftwIn[x->x_arraySize+i] = x->x_vec[i].w_float;

	// execute forward FFT
	fftwf_execute(fftwForwardPlan);

	// apply bin scalars
	for(i=0; i<windowTripleHalf; i++)
	{
		fftwOut[i][0] *= eqArray[i];
		fftwOut[i][1] *= eqArray[i];
	}
	
	// execute inverse FFT
	fftwf_execute(fftwInversePlan);
	
	// write altered signal to internal memory for analysis
	for(i=0; i<x->x_arraySize; i++)
		x->x_irSignalEq[i] = fftwIn[x->x_arraySize+i]/windowTriple;

    t_freebytes(eqArray, windowTripleHalf*sizeof(t_float));
    t_freebytes(barkBinBounds, (NUMBARKBOUNDS+1)*sizeof(int));
    t_freebytes(fftwIn, windowTriple*sizeof(t_float));
	fftwf_free(fftwOut);
	fftwf_destroy_plan(fftwForwardPlan); 
	fftwf_destroy_plan(fftwInversePlan); 

	post("%s: EQ scalars applied to IR array %s", x->x_objSymbol->s_name, x->x_arrayName->s_name);
	
	// re-run analysis
	convolve_tilde_analyze(x, x->x_arrayName);
	}
}


static void convolve_tilde_window(convolve_tilde *x, t_float w)
{
	int i, iWin, is64, oldWindow, oldWindowDouble;

	iWin = w;
	is64 = (iWin%64)==0;
	
	oldWindow = x->x_window;
	oldWindowDouble = x->x_windowDouble;

	if(is64)
	{
		if(iWin >= MINWIN)
			x->x_window = iWin;
		else
		{
			x->x_window = MINWIN;
			pd_error(x, "%s: requested window size too small. minimum value of %i used instead", x->x_objSymbol->s_name, x->x_window);				
		}
	}
	else
	{
		x->x_window = DEFAULTWIN;
		pd_error(x, "%s: window not a multiple of 64. default value of %i used instead", x->x_objSymbol->s_name, x->x_window);
	}

	// resize time-domain buffer to zero bytes
    x->x_nonOverlappedOutput = (t_sample *)t_resizebytes(
    	x->x_nonOverlappedOutput,
    	((x->x_numParts+1)*x->x_windowDouble)*sizeof(t_sample),
    	0
    );

	// update window-based terms
	x->x_windowDouble = x->x_window*2;
	x->x_ampScalar = 1.0f/x->x_windowDouble;
	x->x_bufferLimit = x->x_window/x->x_n;

	// update window-based memory
	fftwf_free(x->x_irFreqDomData);
	fftwf_free(x->x_liveFreqDomData);
	fftwf_free(x->x_sigBufPadFftwOut);
	fftwf_free(x->x_invOutFftwIn);
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_liveFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_sigBufPadFftwOut = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_invOutFftwIn = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	x->x_signalBuf = (t_sample *)t_resizebytes(x->x_signalBuf, oldWindow*sizeof(t_sample), x->x_window*sizeof(t_sample));
	x->x_finalOutput = (t_sample *)t_resizebytes(x->x_finalOutput, oldWindow*sizeof(t_sample), x->x_window*sizeof(t_sample));
	
	x->x_signalBufPadded = (t_sample *)t_resizebytes(x->x_signalBufPadded, oldWindowDouble*sizeof(t_sample), x->x_windowDouble*sizeof(t_sample));
	x->x_invOutFftwOut = (t_sample *)t_resizebytes(x->x_invOutFftwOut, oldWindowDouble*sizeof(t_sample), x->x_windowDouble*sizeof(t_sample));

	// signalBufPadded plan
	fftwf_destroy_plan(x->x_sigBufPadFftwPlan); 
	fftwf_destroy_plan(x->x_invOutFftwPlan); 
	x->x_sigBufPadFftwPlan = fftwf_plan_dft_r2c_1d(x->x_windowDouble, x->x_signalBufPadded, x->x_sigBufPadFftwOut, FFTW_ESTIMATE);
	x->x_invOutFftwPlan = fftwf_plan_dft_c2r_1d(x->x_windowDouble, x->x_invOutFftwIn, x->x_invOutFftwOut, FFTW_ESTIMATE);

	// init signal buffers
 	for(i=0; i<x->x_window; i++)
 	{
 		x->x_signalBuf[i] = 0.0;
		x->x_finalOutput[i] = 0.0;
	}
	
 	for(i=0; i<x->x_windowDouble; i++)
 	{
 		x->x_signalBufPadded[i] = 0.0;
 		x->x_invOutFftwOut[i] = 0.0;
	}	

	post("%s: partition size: %i", x->x_objSymbol->s_name, x->x_window);

	// set numParts back to zero so that the IR analysis routine is initialized as if it's the first call
	x->x_numParts = 0;
	// reset DSP ticks since we're clearing a new buffer and starting to fill it with signal
	x->x_dspTick = 0;

	// re-run IR analysis routine, but only IF x->arrayName exists
	if(x->x_arrayName == gensym("NOARRAYSPECIFIED"))
		;
	else
		convolve_tilde_analyze(x, x->x_arrayName);
}


static void convolve_tilde_print(convolve_tilde *x)
{
	post("%s: IR array: %s", x->x_objSymbol->s_name, x->x_arrayName->s_name);
	post("%s: array length: %i", x->x_objSymbol->s_name, x->x_arraySize);
	post("%s: window size: %i", x->x_objSymbol->s_name, x->x_window);
	post("%s: number of partitions: %i", x->x_objSymbol->s_name, x->x_numParts);
	post("%s: sampling rate: %i", x->x_objSymbol->s_name, (int)x->x_sr);
	post("%s: block size: %i", x->x_objSymbol->s_name, (int)x->x_n);
}


static void convolve_tilde_flush(convolve_tilde *x)
{
	int i;
	
	// init signal buffers
 	for(i=0; i<x->x_window; i++)
 	{
 		x->x_signalBuf[i] = 0.0;
		x->x_finalOutput[i] = 0.0;
	}
	
 	for(i=0; i<x->x_windowDouble; i++)
 	{
 		x->x_signalBufPadded[i] = 0.0;
 		x->x_invOutFftwOut[i] = 0.0;
	}

	if(x->x_numParts>0)
	{
		// clear time-domain buffer
		for(i=0; i<(x->x_numParts+1)*x->x_windowDouble; i++)
			x->x_nonOverlappedOutput[i] = 0.0;

		// clear x_liveFreqDomData
		for(i=0; i<x->x_numParts*(x->x_window+1); i++)
		{
			x->x_liveFreqDomData[i][0] = 0.0;
			x->x_liveFreqDomData[i][1] = 0.0;
		}
    }

	// reset buffering process
	x->x_dspTick = 0;
}


static void convolve_tilde_initClock(convolve_tilde *x)
{
	x->x_startupFlag = 1;

	// try analyzing at creation if there was a table specified
	if(x->x_arrayName != gensym("NOARRAYSPECIFIED"))
		convolve_tilde_analyze(x, x->x_arrayName);
}


static void *convolve_tilde_new(t_symbol *s, int argc, t_atom *argv)
{
    convolve_tilde *x = (convolve_tilde *)pd_new(convolve_tilde_class);

	int i, is64, window;
	
	outlet_new(&x->x_obj, &s_signal);

	// store the pointer to the symbol containing the object name. Can access it for error and post functions via s->s_name
	x->x_objSymbol = s;

	switch(argc)
	{		
		case 0:
			window = DEFAULTWIN;
			x->x_arrayName = gensym("NOARRAYSPECIFIED");
			break;

		case 1:
			window = atom_getfloat(argv);
			x->x_arrayName = gensym("NOARRAYSPECIFIED");
			break;

		case 2:
			window = atom_getfloat(argv);
			x->x_arrayName = atom_getsymbol(argv+1);
			break;
	
		default:
			pd_error(x, "%s: the only creation argument should be the window/partition size in samples", x->x_objSymbol->s_name);
			window = DEFAULTWIN;
			x->x_arrayName = gensym("NOARRAYSPECIFIED");
			break;
	}
	
	is64 = (window%64)==0;
	if(is64)
	{
		if(window >= MINWIN)
			x->x_window = window;
		else
		{
			x->x_window = MINWIN;
			pd_error(x, "%s: requested window size too small. minimum value of %i used instead", x->x_objSymbol->s_name, x->x_window);				
		}
	}
	else
	{
		x->x_window = DEFAULTWIN;
		pd_error("%s: window not a multiple of 64. default value of %i used instead", x->x_objSymbol->s_name, x->x_window);
	}

    x->x_clock = clock_new(x, (t_method)convolve_tilde_initClock);	
	x->x_arraySize = 0;
	x->x_numParts = 0;
	x->x_sr = 44100;
	x->x_n = 64;
	x->x_dspTick = 0; 
	x->x_windowDouble = x->x_window*2;
	x->x_ampScalar = 1.0f/x->x_windowDouble;
	
	x->x_bufferLimit = x->x_window/x->x_n;

	// these will be resized when analysis occurs
	x->x_nonOverlappedOutput = (t_sample *)getbytes(0);
	x->x_irSignalEq = (t_sample *)getbytes(0);
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_liveFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	x->x_signalBuf = (t_sample *)getbytes(x->x_window*sizeof(t_sample));
	x->x_signalBufPadded = (t_sample *)getbytes(x->x_windowDouble*sizeof(t_sample));
	x->x_invOutFftwOut = (t_sample *)getbytes(x->x_windowDouble*sizeof(t_sample));
	x->x_finalOutput = (t_sample *)getbytes(x->x_window*sizeof(t_sample));

	// set up the FFTW output buffer for signalBufPadded
	x->x_sigBufPadFftwOut = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	// signalBufPadded plan
	x->x_sigBufPadFftwPlan = fftwf_plan_dft_r2c_1d(x->x_windowDouble, x->x_signalBufPadded, x->x_sigBufPadFftwOut, FFTW_ESTIMATE);

	x->x_invOutFftwIn = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	// invOut plan
	x->x_invOutFftwPlan = fftwf_plan_dft_c2r_1d(x->x_windowDouble, x->x_invOutFftwIn, x->x_invOutFftwOut, FFTW_ESTIMATE);

	// init signal buffers
 	for(i=0; i<x->x_window; i++)
 	{
 		x->x_signalBuf[i] = 0.0;
		x->x_finalOutput[i] = 0.0;
	}
	
 	for(i=0; i<x->x_windowDouble; i++)
 	{
 		x->x_signalBufPadded[i] = 0.0;
 		x->x_invOutFftwOut[i] = 0.0;
	}	

    post("%s: version 0.11", x->x_objSymbol->s_name);
    post("%s: partition size %i", x->x_objSymbol->s_name, x->x_window);

	clock_delay(x->x_clock, 0); // wait 0ms before IR analysis to give a control cycle for IR samples to be loaded
	
    return(x);
}


static t_int *convolve_tilde_perform(t_int *w)
{
    int i, p, n, window, windowDouble, numParts;
	t_float ampScalar;
	
    convolve_tilde *x = (convolve_tilde *)(w[1]);

    t_sample *in = (t_float *)(w[2]);
    t_sample *out = (t_float *)(w[3]);
    
    n = w[4];
	window = x->x_window;
	windowDouble = x->x_windowDouble;
	numParts = x->x_numParts;
	ampScalar = x->x_ampScalar;
	
	if(n!=64 || numParts<1)
	{
		for(i=0; i<n; i++)
			out[i] = 0.0;

		return (w+5);		
	};
	
	// buffer most recent block
	for(i=0; i<n; i++)
		x->x_signalBuf[(x->x_dspTick*n)+i] = in[i];
		
	if(++x->x_dspTick >= x->x_bufferLimit)
	{
		x->x_dspTick = 0;
		
		// don't do anything if the IR hasn't been analyzed yet
		if(x->x_numParts>0)
		{

		// copy the signal buffer into the transform IN buffer
 		for(i=0; i<window; i++)
 			x->x_signalBufPadded[i] = x->x_signalBuf[i];
		
		// pad the rest out with zeros
		for(; i<windowDouble; i++)
			x->x_signalBufPadded[i] = 0.0;

		// take FT of the most recent input, padded to double window size
		fftwf_execute(x->x_sigBufPadFftwPlan);
			
 		// multiply against partitioned IR spectra. these need to be complex multiplies
 		// also, sum into appropriate part of the nonoverlapped buffer
		for(p=0; p<numParts; p++)
		{
			int startIdx;
			
			startIdx = p*(window+1);

			for(i=0; i<(window+1); i++)
			{
				t_float realLive, imagLive, realIR, imagIR, real, imag;

				realLive = x->x_sigBufPadFftwOut[i][0];
				imagLive = x->x_sigBufPadFftwOut[i][1];

				realIR = x->x_irFreqDomData[startIdx+i][0];
				imagIR = x->x_irFreqDomData[startIdx+i][1];
				
				// MINUS the imag part because i^2 = -1
				real = (realLive * realIR) - (imagLive * imagIR);
				imag = (imagLive * realIR) + (realLive * imagIR);

				// sum into the live freq domain data buffer
				x->x_liveFreqDomData[startIdx+i][0] += real;
				x->x_liveFreqDomData[startIdx+i][1] += imag;
			}
		}

		// copy the freq dom data from head of the complex summing buffer into the inverse FFT input buffer
		for(i=0; i<(window+1); i++)
		{
			x->x_invOutFftwIn[i][0] = x->x_liveFreqDomData[i][0];
			x->x_invOutFftwIn[i][1] = x->x_liveFreqDomData[i][1];
		}
		
		// execute 
		fftwf_execute(x->x_invOutFftwPlan);
		
		// copy the latest IFFT time-domain result into the SECOND block of x_nonOverlappedOutput. The first block contains time-domain results from last time, which we will overlap-add with
		for(i=0; i<windowDouble; i++)
			x->x_nonOverlappedOutput[windowDouble+i] = x->x_invOutFftwOut[i];
		
		// write time domain output to x->x_finalOutput and reduce gain
		for(i=0; i<window; i++)
		{
			x->x_finalOutput[i] = x->x_nonOverlappedOutput[window+i] +  x->x_nonOverlappedOutput[windowDouble+i];

			x->x_finalOutput[i] *= ampScalar;
		}

		// push the live freq domain data buffer contents backwards
		for(i=0; i<((numParts*(window+1))-(window+1)); i++)
		{
			x->x_liveFreqDomData[i][0] = x->x_liveFreqDomData[(window+1)+i][0];
			x->x_liveFreqDomData[i][1] = x->x_liveFreqDomData[(window+1)+i][1];
		}
		
		// init the newly available chunk at the end
		for(; i<(numParts*(window+1)); i++)
		{
			x->x_liveFreqDomData[i][0] = 0.0;
			x->x_liveFreqDomData[i][1] = 0.0;
		}
		
		// push remaining output buffer contents backwards
		for(i=0; i<(((numParts+1)*windowDouble)-windowDouble); i++)
			x->x_nonOverlappedOutput[i] = x->x_nonOverlappedOutput[windowDouble+i];
		
		// init the newly available chunk at the end
		for(; i<((numParts+1)*windowDouble); i++)
			x->x_nonOverlappedOutput[i] = 0.0;
 		}
	};
	
	// output
	for(i=0; i<n; i++)
		out[i] = x->x_finalOutput[(x->x_dspTick*n)+i];
	
    return (w+5);
}


// could do a check here for whether x->x_n == sp[0]->s_n
// if not, could suspend DSP and throw an error. Block sizes must match
static void convolve_tilde_dsp(convolve_tilde *x, t_signal **sp)
{
	dsp_add(
		convolve_tilde_perform,
		4,
		x,
		sp[0]->s_vec,
		sp[1]->s_vec,
		sp[0]->s_n
	);

	// TODO: could allow for re-blocking and re-calc x_bufferLimit based on new x_n
	if(sp[0]->s_n != x->x_n)
	    pd_error(x, "%s: block size must be 64. DSP suspended.", x->x_objSymbol->s_name);
	
	if(sp[0]->s_sr != x->x_sr)
	{
		x->x_sr = sp[0]->s_sr;
	    post("%s: sample rate updated to %i", x->x_objSymbol->s_name, x->x_sr);
	};
	
	if(x->x_numParts<1)
		pd_error(x, "%s: impulse response analysis not performed yet. output will be zero.", x->x_objSymbol->s_name);
};


static void convolve_tilde_free(convolve_tilde *x)
{	
    t_freebytes(x->x_irSignalEq, x->x_arraySize*sizeof(t_sample));
    t_freebytes(x->x_signalBuf, x->x_window*sizeof(t_sample));
    t_freebytes(x->x_signalBufPadded, x->x_windowDouble*sizeof(t_sample));
    t_freebytes(x->x_invOutFftwOut, x->x_windowDouble*sizeof(t_sample));
    t_freebytes(x->x_finalOutput, x->x_window*sizeof(t_sample));

	// free FFTW stuff
	fftwf_free(x->x_irFreqDomData);
	fftwf_free(x->x_liveFreqDomData);
	fftwf_free(x->x_sigBufPadFftwOut);
	fftwf_free(x->x_invOutFftwIn);
	fftwf_destroy_plan(x->x_sigBufPadFftwPlan); 
	fftwf_destroy_plan(x->x_invOutFftwPlan); 

    if(x->x_numParts>0)
    {
	    t_freebytes(
	    	x->x_nonOverlappedOutput,
	    	(x->x_numParts+1)*x->x_windowDouble*sizeof(t_sample)
	    );
	}
	else
	    t_freebytes(x->x_nonOverlappedOutput, 0);

	clock_free(x->x_clock);
};


void convolve_tilde_setup(void)
{
    convolve_tilde_class = 
    class_new(
    	gensym("convolve~"),
    	(t_newmethod)convolve_tilde_new,
    	(t_method)convolve_tilde_free,
        sizeof(convolve_tilde),
        CLASS_DEFAULT,
        A_GIMME,
		0
    );

    CLASS_MAINSIGNALIN(convolve_tilde_class, convolve_tilde, x_f);

	class_addcreator(
		(t_newmethod)convolve_tilde_new,
		gensym("wbrent/convolve~"),
		A_GIMME,
		0
	);

	class_addmethod(
		convolve_tilde_class,
		(t_method)convolve_tilde_print,
		gensym("print"),
		0
	);

	class_addmethod(
		convolve_tilde_class,
		(t_method)convolve_tilde_window,
		gensym("window"),
		A_DEFFLOAT,
		0
	);
	
	class_addmethod(
		convolve_tilde_class,
		(t_method)convolve_tilde_analyze,
		gensym("analyze"),
		A_SYMBOL,
		0
	);

	class_addmethod(
		convolve_tilde_class,
		(t_method)convolve_tilde_eq,
		gensym("eq"),
		A_GIMME,
		0
	);

	class_addmethod(
		convolve_tilde_class,
		(t_method)convolve_tilde_flush,
		gensym("flush"),
		0
	);
	
    class_addmethod(
    	convolve_tilde_class,
    	(t_method)convolve_tilde_dsp,
    	gensym("dsp"),
    	0
    );
}