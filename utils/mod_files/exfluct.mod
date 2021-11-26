TITLE Fluctuating Na & K conductance

COMMENT
-----------------------------------------------------------------------------
	This file models fluctuating Na & K conductance (nominal AMPA synaptic input).
	To use it, set the g_e0 and std_e variables. 

	To drive the synaptic conductance using a waveform
		http://www.neuron.yale.edu/phpbb/viewtopic.php?f=16&t=2475

		Two tasks are to be performed:
		1. read time stream data from a file into a pair of Vectors--one for the series of conductance values, the other for the corresponding times
		2. arrange for the conductance values to be played into the desired target variable at the appropriate times

		Task 1 (reading data from a file) is straightforward to do with the File and Vector classes. Assuming the data file format is
		t0 g0
		t1 g1
		etc.
		where each tg pair specifies the conductance g at a particular time t, you could use the example provided by this item in the "Hot tips" area of the Forum--
		How to read numerical values from a file
		I'd call the vectors gvec and tvec.

		Then, if you have a point process that has some variable g that you want to drive with the waveform data (assuming that the point process is already attached to a section and is referenced by an objref called pp, and the )--
		gvec.play(&pp.g, tvec, 1)
		Notice the third argument is 1. This is the "continuous" argument; you should read about it in the Programmer's Reference discussion of the Vector class's play method.
		http://www.neuron.yale.edu/neuron/static/docs/help/neuron/general/classes/vector/vect.html#play

	Authors: Christopher Brian Currin and Andrew Trevelyan 

-----------------------------------------------------------------------------
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	THREADSAFE
	POINT_PROCESS exfluct
	RANGE g_e, g_e0, g_fluct
	RANGE std_e, tau_e, D_e
	RANGE Tdur
	RANGE new_seed
    USEION na READ nai,nao,ena WRITE ina
	USEION k READ ki,ko,ek WRITE ik
}

UNITS {
	(nA) 	= (nanoamp) 
	(mV) 	= (millivolt)
	(umho) 	= (micromho)
	(mA) 	= (milliamp)
	(molar)	= (1/liter)
	(mM) 	= (millimolar)
}

PARAMETER {
	dt		(ms)
	v		(mV)
	ko		(mM)
	ki		(mM)
	nai		(mM)
	nao		(mM)
	g_e0 = 0.0121 	(umho)	: average excitatory conductance
	std_e = 0.0030 	(umho)	: standard dev of excitatory conductance
	tau_e	= 2.728	(ms)	: time constant of excitatory conductance
	Tdur 	= 1.0	(ms)	: transmitter duration
}

ASSIGNED {
	ik	(nA)
	ek	(mV)
	ena	(mV)
	ina	(nA)	
	g_e	(umho)		: total excitatory conductance
	D_e	(umho umho /ms) : excitatory diffusion coefficient
	exp_e
	amp_e	(umho)
	}

STATE {
	g_fluct	(umho)		: fluctuating excitatory conductance
}

INITIAL {
	g_fluct = 0
	if(tau_e != 0) {
		D_e = 2 * std_e * std_e / tau_e
		exp_e = exp(-dt/tau_e)
		amp_e = std_e * sqrt( (1-exp(-2*dt/tau_e)) )
	}
}

BREAKPOINT {
	SOLVE updateFluct METHOD cnexp
	if(tau_e==0) {
	   g_e = std_e * normrand(0,1)
	}
	
	g_e = g_e0 + g_fluct
	if(g_e < 0) { g_e = 0 }
	
	ina = g_e * (v - ena) / 2
	ik = g_e * (v - ek) / 2
}

DERIVATIVE updateFluct{
	g_fluct' = -g_fluct/tau_e + amp_e*normrand(0,1)/dt
}

PROCEDURE new_seed(seed) {		: procedure to set the seed
	set_seed(seed)
	VERBATIM
	  printf("Setting random generator with seed = %g\n", _lseed);
	ENDVERBATIM
}

NET_RECEIVE(weight) {
	: do nothing
}
