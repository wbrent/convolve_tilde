/*

convolve~

Copyright 2010 William Brent

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

version 0.10, March 17, 2018

- using FFTW as of version 0.10

*/

#include "m_pd.h"
#include "fftw3.h"
#include <math.h>
#define MINWIN 64
#define DEFAULTWIN 2048

static t_class *convolve_tilde_class;

typedef struct _convolve_tilde 
{
    t_object x_obj;
	t_symbol *x_objSymbol;
    t_symbol *x_arrayName;
    t_word *x_vec;
    int x_suspendDSP;
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
	t_float *x_eqArray;
	
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
    int i, j, oldSize, newSize;
    t_float *fftwIn;
    fftwf_complex *fftwOut;
	fftwf_plan fftwPlan;

	if(x->x_numParts>0)
		oldSize = (x->x_numParts+1)*x->x_windowDouble;
	else
		oldSize = 0;

    if(!(arrayPtr = (t_garray *)pd_findbyclass(arrayName, garray_class)))
    {
        if(*arrayName->s_name)
        {
        	pd_error(x, "%s: no array named %s", x->x_objSymbol->s_name, arrayName->s_name);
	        x->x_vec = 0;
        	return;
	    }
    }
    else if(!garray_getfloatwords(arrayPtr, &x->x_arraySize, &x->x_vec))
    {
        pd_error(x, "%s: bad template for %s", x->x_objSymbol->s_name, arrayName->s_name);
        x->x_arraySize = 0;
        x->x_vec = 0;
        return;
    }
    else
		x->x_arrayName = arrayName;
    
    // count how many partitions there will be for this IR
    x->x_numParts = 0;
    while((x->x_numParts*x->x_window) < x->x_arraySize)
    	x->x_numParts++;

    newSize = (x->x_numParts+1)*x->x_windowDouble;
    
    // resize time-domain buffer
    x->x_nonOverlappedOutput = (t_sample *)t_resizebytes(
    	x->x_nonOverlappedOutput,
    	oldSize*sizeof(t_sample),
    	newSize*sizeof(t_sample)
    );

	// clear time-domain buffer
    for(i=0; i<newSize; i++)
    	x->x_nonOverlappedOutput[i] = 0.0;

	// free x_irFreqDomData/x_liveFreqDomData and re-alloc to new size based on x_numParts
	fftwf_free(x->x_irFreqDomData);
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_numParts*(x->x_window+1));

	fftwf_free(x->x_liveFreqDomData);
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
    	
		for(j=0; j<x->x_window && (startVec+j)<x->x_arraySize; j++)
			fftwIn[j] = x->x_vec[startVec+j].w_float;
		
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
	
	post("%s: analysis of IR array %s complete. %i partitions", x->x_objSymbol->s_name, x->x_arrayName->s_name, x->x_numParts);
}


static void convolve_tilde_eq(convolve_tilde *x, t_symbol *s, int argc, t_atom *argv)
{
	int i;

	if(argc<(x->x_window+1))
		pd_error(x, "%s: EQ scalar list contains fewer than %i elements. Ignored.", x->x_objSymbol->s_name, x->x_window+1);
	else
	{
		for(i=0; i<(x->x_window+1); i++)
			x->x_eqArray[i] = (atom_getfloat(argv+i)<0)?0:atom_getfloat(argv+i);
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
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_liveFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	x->x_signalBuf = (t_sample *)t_resizebytes(x->x_signalBuf, oldWindow*sizeof(t_sample), x->x_window*sizeof(t_sample));
	x->x_signalBufPadded = (t_sample *)t_resizebytes(x->x_signalBufPadded, oldWindowDouble*sizeof(t_sample), x->x_windowDouble*sizeof(t_sample));
	x->x_invOutFftwOut = (t_sample *)t_resizebytes(x->x_invOutFftwOut, oldWindowDouble*sizeof(t_sample), x->x_windowDouble*sizeof(t_sample));
	x->x_finalOutput = (t_sample *)t_resizebytes(x->x_finalOutput, oldWindow*sizeof(t_sample), x->x_window*sizeof(t_sample));

	x->x_eqArray = (t_float *)t_resizebytes(x->x_eqArray, (oldWindow+1)*sizeof(t_float), (x->x_window+1)*sizeof(t_float));

	// set up the FFTW output buffer for signalBufPadded
	fftwf_free(x->x_sigBufPadFftwOut);
	x->x_sigBufPadFftwOut = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	// signalBufPadded plan
	fftwf_destroy_plan(x->x_sigBufPadFftwPlan); 
	x->x_sigBufPadFftwPlan = fftwf_plan_dft_r2c_1d(x->x_windowDouble, x->x_signalBufPadded, x->x_sigBufPadFftwOut, FFTW_ESTIMATE);

	fftwf_free(x->x_invOutFftwIn);
	x->x_invOutFftwIn = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	// invOut plan
	fftwf_destroy_plan(x->x_invOutFftwPlan); 
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

	// initialize EQ array to flat
 	for(i=0; i<(x->x_window+1); i++)
 		x->x_eqArray[i] = 1.0; 		

	post("%s: partition size: %i. EQ scalars reset to 1.0", x->x_objSymbol->s_name, x->x_window);

	// set numParts back to zero so that the IR analysis routine is initialized as if it's the first call
	x->x_numParts = 0;
	// reset DSP ticks since we're clearing a new buffer and starting to fill it with signal
	x->x_dspTick = 0;

	// re-run IR analysis routine, but only IF x->arrayName exists
	if(x->x_arrayName==gensym("NOARRAYSPECIFIED"))
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
	
	x->x_suspendDSP = 0;
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
	x->x_irFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);
	x->x_liveFreqDomData = (fftwf_complex *)fftwf_alloc_complex(x->x_window+1);

	x->x_signalBuf = (t_sample *)getbytes(x->x_window*sizeof(t_sample));
	x->x_signalBufPadded = (t_sample *)getbytes(x->x_windowDouble*sizeof(t_sample));
	x->x_invOutFftwOut = (t_sample *)getbytes(x->x_windowDouble*sizeof(t_sample));
	x->x_finalOutput = (t_sample *)getbytes(x->x_window*sizeof(t_sample));

	x->x_eqArray = (t_float *)getbytes((x->x_window+1)*sizeof(t_float));

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

	// initialize EQ array to flat
 	for(i=0; i<(x->x_window+1); i++)
 		x->x_eqArray[i] = 1.0; 		

    post("%s: version 0.10", x->x_objSymbol->s_name);
    post("%s: partition size %i", x->x_objSymbol->s_name, x->x_window);

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
		
	if(x->x_suspendDSP)
	{
		for(i=0; i<n; i++, out++)
			*out = 0.0;

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

				realIR = x->x_irFreqDomData[startIdx+i][0] * x->x_eqArray[i];
				imagIR = x->x_irFreqDomData[startIdx+i][1] * x->x_eqArray[i];
				
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
		for(i=0; i<(numParts*(window+1))-(window+1); i++)
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
		for(i=0; i<((x->x_numParts+1)*x->x_windowDouble)-windowDouble; i++)
			x->x_nonOverlappedOutput[i] = x->x_nonOverlappedOutput[windowDouble+i];
		
		// init the newly available chunk at the end
		for(; i<(x->x_numParts+1)*x->x_windowDouble; i++)
			x->x_nonOverlappedOutput[i] = 0.0;
 		}
 
	};
	
	// output
	for(i=0; i<n; i++, out++)
	{
		if(x->x_numParts>0)
			*out = x->x_finalOutput[(x->x_dspTick*n)+i];
		else
			*out = 0.0;
	};
	
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

	if(sp[0]->s_n != x->x_n)
	{
		x->x_suspendDSP = 1;
	    pd_error(x, "%s: block size must be 64. DSP suspended.", x->x_objSymbol->s_name);
	}
	else
	{
		// best to flush all buffers and reset buffering process before resuming
		convolve_tilde_flush(x);

	    post("%s: DSP resumed.", x->x_objSymbol->s_name);

		x->x_suspendDSP = 0;
	}
	
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
    t_freebytes(x->x_signalBuf, x->x_window*sizeof(t_sample));
    t_freebytes(x->x_signalBufPadded, x->x_windowDouble*sizeof(t_sample));
    t_freebytes(x->x_invOutFftwOut, x->x_windowDouble*sizeof(t_sample));
    t_freebytes(x->x_eqArray, (x->x_window+1)*sizeof(t_float));
    t_freebytes(x->x_finalOutput, x->x_window*sizeof(t_sample));

	// free FFTW stuff
	fftwf_free(x->x_irFreqDomData);
	fftwf_free(x->x_sigBufPadFftwOut);
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