TITLE Fluctuating inhibitory conductance

COMMENT
-----------------------------------------------------------------------------
	This file models fluctuating Cl inhibitory conductance (nominal GABAA synaptic input).
	To use it, set the e_fluct, g_i0 and std_i variables.

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
	POINT_PROCESS inhfluct_simple
	RANGE g_i, g_i0, g_fluct
	RANGE std_i, tau_i, D_i
	RANGE e_fluct, i
	RANGE Tdur
	RANGE new_seed
	NONSPECIFIC_CURRENT i
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
	dt					(ms)
	v					(mV)	: membrane voltage
	e_fluct	= -65 		(mV) 	: reversal potential
	g_i0 	= 0.0573 	(umho)	: average inhibitory conductance
	std_i 	= 0.0066 	(umho)	: standard dev of inhibitory conductance
	tau_i 	= 10.49		(ms)	: time constant of inhibitory conductance
	Tdur 	= 1.0		(ms)	: transmitter duration
}

ASSIGNED {
	g_i		(umho)			: total inhibitory conductance
	D_i		(umho umho /ms) : inhibitory diffusion coefficient
	exp_i
	amp_i	(umho)
	i       (nA)
}

STATE{ 
	g_fluct	(umho)		: fluctuating inhibitory conductance 
}

INITIAL {
	g_fluct = 0
	
	if(tau_i != 0) {
		D_i = 2 * std_i * std_i / tau_i
		exp_i = exp(-dt/tau_i)
		amp_i = std_i * sqrt( (1-exp(-2*dt/tau_i)) )
	}
}

BREAKPOINT {
	SOLVE updateFluct METHOD cnexp

	if(tau_i==0) {
	   g_i = std_i * normrand(0,1)
	}

	g_i = g_i0 + g_fluct
	if(g_i < 0) { g_i = 0 }
	i = g_i * (v - e_fluct)
}

DERIVATIVE updateFluct{
	g_fluct' = -g_fluct/tau_i + amp_i*normrand(0,1)/dt
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
