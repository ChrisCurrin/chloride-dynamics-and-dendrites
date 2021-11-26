TITLE Simplified ionic model of GABA-A synapse, with saturation

COMMENT
-----------------------------------------------------------------------------
	This file models a GABA-A synapse using chloride and bicarbonate flux
  	to determine overall current.

	The receptor mechanism is based on the simple conceptual model of
	transmitter-receptor interaction:
				α
		C + T <===> O
				β
	where transmitter T binds to a closed receptor channel C to produce an
	open channel O.

	The kinetic parameters are from Destexhe et al. (1998).
	The NEURON schema follows that from The Neuron Book: Chapter 10,
	Example 10.6: saturating synapses.
	Credit should also be given to Blake Richards and Joseph Raimondo for
	initial work on this model.

	For further information:
	Destexhe et al. (1998) has 3 'versions' of how to model GABAA (and others)
	A 5-gated model has been made by them in NEURON but uses the old POINTER
	network mechanism instead if the updated NET_RECEIVE mechanism.
	Destexhe et al. (1994) has more schemes, parameters and derivations for
	GABA-A and others.

	Authors: Christopher Brian Currin

	References:
	 -  Bormann, J., Hamill, O. P., & Sakmann, B. (1987). Mechanism of anion
		permeation through channels gated by glycine and gamma-aminobutyric acid
		in mouse cultured spinal neurones. The Journal of Physiology, 385, 243–286.
	 - 	Staley KJ and Proctor, WR. (1999). Journal of Physiology, 519: 693-712.
	 - 	Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. (1994).  An efficient method
	 	for computing synaptic conductances based on a kinetic model of receptor
	 	binding, Neural Computation 6: 10-14.
	 -  Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  (1998). Kinetic models of
  		synaptic transmission.  In: Methods in Neuronal Modeling (2nd edition;
  		edited by Koch, C. and Segev, I.), MIT press, Cambridge, pp. 1-25.
  	 -  Carnevale, N and Hines, M. (2001). The Neuron Book, Cambridge University
  	 	Press: Cambridge
-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
	THREADSAFE
	POINT_PROCESS GABAa
	RANGE g, gmax
	RANGE pcl, phco3
	RANGE Tdur, Alpha
    RANGE Beta, Rinf, Rtau
	USEION cl READ ecl WRITE icl VALENCE -1
	USEION hco3 READ ehco3 WRITE ihco3 VALENCE -1
	USEION gaba WRITE egaba VALENCE -1
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(M)  = (1/liter)
	(mM) = (milliM)
}

PARAMETER {
	pcl		= 0.8				: permeability fraction of Cl
	phco3	= 0.2				: permeability fraction of HCO3
	Tdur 	= 1.0		(ms)	: transmitter duration (rising phase)
	Alpha 	= 5.0		(/ms)	: forward (binding) rate
	Beta 	= 0.18 	  	(/ms)	: backward (unbinding) rate
	gmax 	= 0.00005	(uS)	: max conductance for 1 channel (use
								: 'weight' of 1000 in netcon to increase
								: to synapse levels of 50 nS)
}

ASSIGNED {
	v       (mV)    : postsynaptic voltage
	g       (uS)    : total conductance
	icl		(nA)	: chloride current
	ecl		(mV)	: chloride reversal potential
	ihco3	(nA)	: bicarbonate current
	ehco3	(mV)	: bicarbonate reversal potential
	egaba	(mV)	: overall reversal potential
	celsius	(degC)	: temperature
	Rtau 	(ms)	: time constant of channel binding
	Rinf 			: fraction of open channels if transmitter is present forever
	synon			: sum of weights of all synapses in "onset" state
}

STATE { Ron Roff } 	: initialized to 0 by default
					: Ron and Roff are the total conductances of all synapses
					: that are in the "onset" (transmitter pulse ON)
					: and "offset" (transmitter pulse OFF) states, respectively

INITIAL {
	synon = 0
	Rtau  = 1 / (Alpha + Beta)
	Rinf  = Alpha / (Alpha + Beta)
}

BREAKPOINT {
	SOLVE release METHOD cnexp
	g 	 = gmax * (Ron + Roff)
	icl  = g * pcl * (v - ecl)
  	ihco3 = g * phco3 * (v - ehco3)
  	egaba = pcl*ecl + phco3*ehco3
}

DERIVATIVE release {
	Ron' = (synon*Rinf - Ron)/Rtau
	Roff' = -Beta*Roff
}

NET_RECEIVE(weight, on, r0, t0 (ms), rt, ri) {
	: flag is an implicit argument of NET_RECEIVE, normally 0
	if (flag == 0) {
		: a spike arrived, start onset state if not already on
		if (!on) {
			: this synapse joins the set of synapses in onset state
			synon = synon + weight
			r0 = r0*exp(-Beta*(t - t0)) : r0 at start of onset state
			Ron = Ron + r0
			Roff = Roff - r0
			t0 = t
			on = 1
			: come again in Tdur with flag = 1
			net_send(Tdur, 1)
		} else {
			: already in onset state, so move offset time
			net_move(t + Tdur)
		}
	}
	if (flag == 1) {
		: "turn off transmitter"
		: i.e. this synapse enters the offset state
		synon = synon - weight : r0 at start of offset state
		rt = 1 / (Alpha + Beta)		:can't use Rtau and Rinf for some reason?
		ri = Alpha / (Alpha + Beta)
		r0 = weight*ri + (r0 - weight*ri)*exp(-(t-t0)/rt)
		Ron = Ron - r0 
		Roff = Roff + r0 
		t0 = t 
		on = 0
	} 
}

COMMENT
In its most general form, the NET_RECEIVE statement is
NET_RECEIVE (a0, a1, a2 . . . ) {
a0 is generally used to refer to the synaptic strength ("weight"), and for the 
simplest synaptic mechanisms this is the only item that appears inside the parentheses. 
If additional ai are present, then each NetCon that targets such a synapse actually has
 "weight vector" with storage for as many numerical values as there are items in the 
 parentheses of the NET_RECEIVE statement. This storage is typically used by synaptic 
 mechanisms that have use-dependent plasticity. It allows each NetCon that targets such 
 mechanism to store "state" information that reflects its past activation history. This 
 state information is calculated inside the NET_RECEIVE block. Each time a new event 
 arrives, the NET_RECEIVE block gets not only the numerical value of a0 but also the 
 numerical values of all the other ai from the weight vector of the NetCon that 
 delivered the event.
ENDCOMMENT
