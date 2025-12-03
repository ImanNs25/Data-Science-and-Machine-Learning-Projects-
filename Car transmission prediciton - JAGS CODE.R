data(mtcars)
head(mtcars)
library(rjags)
library(coda)
library(ggplot2)
library(dplyr)


# Outcome (binary): manual transmission
y <- mtcars$am   # 0 = automatic, 1 = manual

# Choose predictor columns (you can change this list)
predictors <- c("mpg", "hp", "wt", "qsec", "disp", "drat", 
                "cyl", "gear", "carb", "vs")

# Extract and standardize predictors
X_raw <- as.matrix(mtcars[, predictors])
X <- scale(X_raw)

# Dimensions
N <- nrow(X)
p <- ncol(X)

N; p



# ------------------------------------------------------------
# 3. Hyperparameters for SSVS
# ------------------------------------------------------------

# Spike-and-slab standard deviations
sigma0 <- 0.2   # small variance = shrink aggressively
sigma1 <- 2.0   # large variance = allow real effect

# Convert to precisions for JAGS
tau0 <- 1 / sigma0^2    # spike precision
tau1 <- 1 / sigma1^2    # slab precision

tau0; tau1






# ------------------------------------------------------------
# 4. JAGS model string with SSVS (safe version)
# ------------------------------------------------------------
model_string <- "
model {

  # ------------------------------------
  # Likelihood: logistic regression
  # ------------------------------------
  for (i in 1:N) {
    y[i] ~ dbern(p_bern[i])

    # Linear predictor: must NOT redefine eta[i]
    eta[i] <- alpha + inprod(beta[], X[i,])
    logit(p_bern[i]) <- eta[i]
  }

  # ------------------------------------
  # Prior on intercept
  # ------------------------------------
  alpha ~ dnorm(0, 0.01)

  # ------------------------------------
  # SSVS priors for coefficients
  # ------------------------------------
  for (j in 1:p) {
    gamma[j] ~ dbern(pi)

    tau_beta[j] <- (1 - gamma[j]) * tau0 + gamma[j] * tau1

    beta[j] ~ dnorm(0, tau_beta[j])
  }

  # Inclusion probability
  pi ~ dbeta(1,1)
}
"

writeLines(model_string, "logit_ssvs_mtcars.jags")





data_jags <- list(
  N = N,      # number of observations
  p = p,      # number of predictors
  y = y,      # outcome vector
  X = X,      # standardized design matrix
  tau0 = tau0, # spike precision
  tau1 = tau1  # slab precision
)



inits <- function() {
  list(
    alpha = 0,
    beta  = rep(0, p),
    gamma = rbinom(p, size = 1, prob = 0.5),   # random inclusion start
    pi    = 0.5
  )
}


params <- c("alpha", "beta", "gamma", "pi")




jags_mod <- jags.model(
  file = "logit_ssvs_mtcars.jags",
  data = data_jags,
  inits = inits,
  n.chains = 3,
  n.adapt = 1000
)

update(jags_mod, 5000)   # burn-in

samples <- coda.samples(
  model = jags_mod,
  variable.names = params,
  n.iter = 5000
)


# Convert coda object to matrix
samps_mat <- as.matrix(samples)

# Find all gamma[j] columns
gamma_cols <- grep("^gamma\\[", colnames(samps_mat))

# Extract gamma samples
gamma_samples <- samps_mat[, gamma_cols, drop = FALSE]

# Posterior inclusion probability for each predictor
post_inclusion <- colMeans(gamma_samples)

# Turn into a table with predictor names
ssvs_results <- data.frame(
  Predictor = predictors,
  PosteriorInclusionProb = round(post_inclusion, 3)
)

# Print sorted from highest to lowest probability
ssvs_results[order(-ssvs_results$PosteriorInclusionProb), ]


# ---- Plot posterior inclusion probabilities ----


plot_data <- data.frame(
  Predictor = predictors,
  InclusionProb = post_inclusion
)

ggplot(plot_data, aes(x = reorder(Predictor, InclusionProb), 
                      y = InclusionProb)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  ylim(0, 1) +
  labs(title = "Posterior Inclusion Probabilities (SSVS)",
       x = "Predictor",
       y = "P(gamma_j = 1 | data)") +
  theme_minimal(base_size = 12)



# ---- Posterior summaries for beta_j ----

beta_cols <- grep("^beta\\[", colnames(samps_mat))
beta_samples <- samps_mat[, beta_cols, drop = FALSE]

beta_summary <- t(apply(beta_samples, 2, function(x) {
  c(mean = mean(x),
    sd   = sd(x),
    lower95 = quantile(x, 0.025),
    upper95 = quantile(x, 0.975))
}))


beta_summary <- as.data.frame(beta_summary)
beta_summary$Predictor <- predictors

# Rename the CI columns properly
colnames(beta_summary)[colnames(beta_summary) == "lower95.2.5%"]  <- "lower95"
colnames(beta_summary)[colnames(beta_summary) == "upper95.97.5%"] <- "upper95"

# Reorder the columns
beta_summary <- beta_summary[, c("Predictor", "mean", "sd", "lower95", "upper95")]

print(beta_summary)


beta_plot_df <- beta_summary %>%
  mutate(IntervalWidth = upper95 - lower95) %>%
  arrange(desc(mean)) %>%
  mutate(Predictor = factor(Predictor, levels = Predictor))

ggplot(beta_plot_df, aes(x = Predictor, y = mean)) +
  geom_point(size = 3, aes(color = IntervalWidth)) +
  geom_errorbar(aes(ymin = lower95, ymax = upper95, color = IntervalWidth),
                width = 0.2, linewidth = 1) +
  scale_color_gradient(low = "darkgreen", high = "red",
                       name = "Uncertainty\n(CI Width)") +
  coord_flip() +
  labs(
    title = "Posterior Means and 95% Credible Intervals for β Coefficients",
    x = "Predictor",
    y = "Posterior Mean"
  ) +
  theme_minimal(base_size = 14)




# ----- Step 3: Posterior summaries for alpha and pi -----

# Extract samples as matrix
samps_mat <- as.matrix(samples)

# Extract alpha and pi columns
alpha_samples <- samps_mat[, "alpha"]
pi_samples    <- samps_mat[, "pi"]

# Summaries
alpha_summary <- c(
  mean   = mean(alpha_samples),
  sd     = sd(alpha_samples),
  lower  = quantile(alpha_samples, 0.025),
  upper  = quantile(alpha_samples, 0.975)
)

pi_summary <- c(
  mean   = mean(pi_samples),
  sd     = sd(pi_samples),
  lower  = quantile(pi_samples, 0.025),
  upper  = quantile(pi_samples, 0.975)
)

alpha_summary
pi_summary



# Extract posterior samples
samps_mat <- as.matrix(samples)

alpha_samps <- samps_mat[, "alpha"]
beta_samps  <- samps_mat[, grep("^beta\\[", colnames(samps_mat)), drop=FALSE]

# Compute predicted probabilities for each iteration
pred_probs <- plogis(alpha_samps + beta_samps %*% t(X))

# pred_probs is MCMC_iter x N
dim(pred_probs)


# Generate posterior predictive binary outcomes
set.seed(123)
y_pred <- matrix(rbinom(length(pred_probs), size=1, prob=pred_probs),
                 nrow = nrow(pred_probs),
                 ncol = ncol(pred_probs))


# Posterior predictive mean (probability of manual)
y_pred_mean <- colMeans(y_pred)

# Compare to observed
pp_check_table <- data.frame(
  Car = rownames(mtcars),
  Observed = y,
  PredictedProb = y_pred_mean
)

head(pp_check_table)



plot(
  y_pred_mean, y,
  pch = 19,
  xlab = "Posterior Predicted Probability (manual)",
  ylab = "Observed am",
  main = "Posterior Predictive Check: Manual Transmission"
)
abline(h=c(0,1), col=c("red","blue"), lty=2)


hist(colMeans(y_pred),
     breaks = 10,
     main="Posterior Predictive Distribution of Mean(am)",
     xlab="Predicted proportion of manual cars")

abline(v = mean(y), col="red", lwd=3)





# start with your beta_summary table


