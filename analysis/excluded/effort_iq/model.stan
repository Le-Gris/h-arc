// RT by difficulty, and accuracy by RT and difficulty 

data {
    int NROWS; 
    int NSUBJ;
    int NITEMS;
    
    int<lower=0,upper=1> correct[NROWS]; // binary correct/incorrect data
    real RTz[NROWS];
	
    int subject[NROWS];
    int item[NROWS];
}

transformed data {
}

parameters {
    
  real difficulty[NITEMS]; // item difficulty 
	
	real gamma0[NSUBJ];
    real gamma0m;
    real<lower=0> gamma0s;

	real gamma1[NSUBJ];
    real gamma1m; // Hmm should we constrain here to be positive?
    real<lower=0> gamma1s;

	real<upper=0> gamma2[NSUBJ];
    real gamma2m;
    real<lower=0> gamma2s;
    
    real beta0[NSUBJ];
    real beta0m;
    real<lower=0> beta0s;
    
	real beta1[NSUBJ];
    real beta1m; // Hmm should we constrain here to be positive?
    real<lower=0> beta1s;   
	
    real<lower=0> rt_scale;

	real rtIcptEffect;
	real rtSlopeEffect;
	real rtInteraction;
	
}


    
transformed parameters {
}
    
model { 
	real SD = 3;
    real ER = 0.1; // rate on the exponential
	
	gamma0m ~ normal(0,SD);
    gamma0 ~ normal(0,1);
    gamma0s ~ exponential(ER);
	
	gamma1m ~ normal(0,SD);
    gamma1 ~ normal(0,1);
    gamma1s ~ exponential(ER);
	
	gamma2m ~ normal(0,SD);
    gamma2 ~ normal(0,1);
    gamma2s ~ exponential(ER);

    beta0m ~ normal(0,SD);
    beta0 ~ normal(0,1);
    beta0s ~ exponential(ER);
	
	beta1m ~ normal(0,SD);
    beta1 ~ normal(0,1);
    beta1s ~ exponential(ER);
	
    rt_scale ~ exponential(0.1);
    difficulty ~ normal(0,1);
	
    rtIcptEffect  ~ normal(0,SD);
	rtSlopeEffect ~ normal(0,SD);
	rtInteraction ~ normal(0,SD);

    for(r in 1:NROWS) {        
        int s = subject[r];
        int i = item[r];
		
		real sgamma0 = (gamma0m+gamma0[s]*gamma0s);
		real sgamma1 = (gamma1m+gamma1[s]*gamma1s);
		real sgamma2 = (gamma2m+gamma2[s]*gamma2s);
		
		real p = inv_logit( sgamma0 + 
							rtIcptEffect*beta0[s] +
							rtSlopeEffect*beta1[s] + 
							rtInteraction*beta0[s]*beta1[s] + 
							sgamma1*RTz[r] +
							sgamma2*difficulty[i] 
							);
		correct[r] ~ bernoulli(p+(1-p)/8);
		
        RTz[r] ~ normal( (beta0m+beta0[s]*beta0s) + 
                         (beta1m+beta1[s]*beta1s)*difficulty[i],
                         rt_scale);
        
    }
}

generated quantities {
	real sgamma0[NSUBJ];
	real sgamma1[NSUBJ];
	real sgamma2[NSUBJ];
	
    real sbeta0[NSUBJ];
    real sbeta1[NSUBJ];
    
	real threshold50[NSUBJ];
	real timehardest50[NSUBJ]; // how long would it take you to get 50% on hardest? -- NOTE not taking into account beta
	
	for(s in 1:NSUBJ) {
		sgamma0[s] = gamma0m+gamma0[s]*gamma0s;
		sgamma1[s] = gamma1m+gamma1[s]*gamma1s;
		sgamma2[s] = gamma2m+gamma2[s]*gamma2s;
		
        sbeta0[s] = beta0m+beta0[s]*beta0s;
        sbeta1[s] = beta1m+beta1[s]*beta1s;        
		
		threshold50[s] = -sgamma0[s]/sgamma2[s]; // hardest you can solve at the avg time
		timehardest50[s] = -(sgamma0[s] + sgamma2[s] * max(difficulty)) / sgamma1[s];
	}
}
