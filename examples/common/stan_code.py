logistic_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0,upper=1> y[n]; // outputs
  matrix[n,d] x; // inputs
}
parameters {
  real theta0; // intercept
  vector[d] theta; // auxiliary parameter
}
transformed parameters {
  vector[n] f;
  f = theta0 + x*theta;
}
model {
  theta0 ~ normal(0, 1);
  theta ~ normal(0, 1);
  y ~ bernoulli_logit(f);
}
"""
poisson_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0> y[n]; // outputs
  matrix[n,d] x; // inputs
}
parameters {
  real theta0; // intercept
  vector[d] theta; // auxiliary parameter
}
transformed parameters {
  vector[n] f;
  f = -log_inv_logit(-(theta0 + x*theta));
}
model {
  theta0 ~ normal(0, 1);
  theta ~ normal(0, 1);
  y ~ poisson(f);
}
"""

