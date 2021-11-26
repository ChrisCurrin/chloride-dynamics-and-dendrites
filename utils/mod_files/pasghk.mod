TITLE GHK passive conductances

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX pasghk
	
	USEION cl READ cli,clo,ecl WRITE icl VALENCE -1
        RANGE gclpbar, icl

	USEION na READ nai,nao,ena WRITE ina
        RANGE gnapbar, ina

	USEION k READ ki,ko,ek WRITE ik
        RANGE gkpbar, ik
}

UNITS {
	(mA) 	= (milliamp)
	(mV) 	= (millivolt)
	(molar)	= (1/liter)
	(mM) 	= (millimolar)
}

PARAMETER {
	v		(mV)

        cli (mM)
        clo (mM)
	gclpbar	= 0.0000225	(mho/cm2) :00024

        nai (mM)
        nao (mM)
	gnapbar	= 0.0000025	(mho/cm2) :0.000025

        ki (mM)
        ko (mM)
	gkpbar	= 0.000025	(mho/cm2) :0.00024
}


ASSIGNED {
	icl		(mA/cm2)
	ecl		(mV)

	ina		(mA/cm2)
	ena		(mV)

	ik		(mA/cm2)
	ek		(mV)
}

BREAKPOINT { 
	icl = gclpbar * (v-ecl)
	ina = gnapbar * (v-ena)
	ik = gkpbar * (v-ek)
}
