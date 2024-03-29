library(ggplot2)
library(dplyr)


g <- function(x)
  -sin(6 * pi * x)^2 + 6 * x^2 - 5 * x^4 + 3 / 2

periodic_kernel <- function(x, x_, v, p, l) {
  dists <- outer(x, x_, `-`)
  s <- sin(pi * dists / p) / l
  v * exp(-2 * s^2)
}

linear_kernel <- function(x, x_, a, b)
  b + a * outer(x, x_)


l <- 30
n <- 20
xs <- (0:(l-1)) / (l-1)
ys <- g(xs) + rnorm(l, 0, sqrt(0.01))

train_indices <- sample(l, n, replace = FALSE)
train_mask <- (1:l) %in% train_indices

xs_train <- xs[train_mask]
ys_train <- ys[train_mask]

xs_test <- xs[!train_mask]
ys_test <- ys[!train_mask]


log_likelihood <- function(x, y, kernel, noise) {
  n <- length(y)
  
  -(t(y) %*% solve(noise * diag(n) + kernel(x, x)) %*% y + prod(unlist(determinant(noise * diag(n) + kernel(x, x), logarithm = TRUE))) + n * log(2 * pi)) / 2
}

# posterior predictive
pp <- function(x_test, x, y, kernel, noise) {
  n <- length(y)
  mu <- t(kernel(x, x_test)) %*% solve(kernel(x, x) + noise * diag(n)) %*% y
  Sigma <- kernel(x_test, x_test) - t(kernel(x, x_test)) %*% solve(kernel(x, x) + noise * diag(n)) %*% kernel(x, x_test)
  
  list(mu, Sigma)
}

xs <- seq(0, 1, length.out = 100)
mu_sigma <- pp(
  xs,
  xs_train,
  ys_train,
  \(x, x_) linear_kernel(x, x_, 2.45, 0.58) + periodic_kernel(x, x_, 4, 0.167, 5),
  0.01
)

mu <- mu_sigma[[1]]
sigma <- mu_sigma[[2]]
ggplot() +
  geom_line(aes(x = xs, y = mu), colour = 'red') +
  geom_line(aes(x = xs, y = g(xs)), alpha = 0.5) +
  geom_point(aes(x = xs_train, y = ys_train))

grid <- expand.grid(
  period = 10^seq(0, 2, length.out = 100),
  noise = 10^seq(-4, 1, length.out = 100)
)

lls <- grid |>
  rowwise() |>
  mutate(
    ll = ({
      # kernel <- \(x, x_) linear_kernel(x, x_, 2.45, 0.58) + periodic_kernel(x, x_, 4, period, 1)
      # kernel <- \(x, x_) periodic_kernel(x, x_, 2.729, period, 0.326)
      kernel <- \(x, x_) periodic_kernel(x, x_, 5, period, 0.2) + linear_kernel(x, x_, 1.3, 0.5)
      # kernel <- \(x, x_) periodic_kernel(x, x_, 2.37, period, 0.2) + linear_kernel(x, x_, 3, 3)
      # kernel <- \(x, x_) periodic_kernel(x, x_, 6.5855, period, 4.1787) + linear_kernel(x, x_, -0.3467, 0.5012)
      # log_likelihood(xs_train, ys_train, kernel, noise)[1, 1] + log(dlnorm(period, 0.1, 100)) + log(dlnorm(noise, 0.01, 100))
      # log_likelihood(xs_train, ys_train, kernel, noise)[1, 1] + log(dlnorm(period, 0.1, 100)) + log(dlnorm(noise, -3.8, 0.5)) #only samples small noise hill
      log_likelihood(xs_train, ys_train, kernel, noise)[1, 1] + log(dlnorm(period, 0.1, 100)) + log(dlnorm(noise, -3.0, 1.5))
    })
  )

lls |> ggplot() +
  aes(x = period, y = noise, z = ll, colour = after_stat(level)) +
  coord_cartesian(expand = FALSE) +
  geom_contour(bins = 5000, show.legend = TRUE) +
  scale_x_log10() +
  scale_y_log10() +
  labs(colour='log-likelihood')

ggsave("contour_with_prior.png", width = 20, height = 8, units = "cm", dpi = 600)

# MAP estimate (according to grid search)
map_estimate <- lls[which.max(lls$ll), ]

# plot prior
plot(\(x) log(dlnorm(x, 0.1, 100)), 0, 10)
plot(\(x) log(dlnorm(x, 0.01, 10)), 0, 5)

# plot posterior predictive
xs <- seq(0, 1, length.out = 100)
mu_sigma <- pp(
  xs,
  xs_train,
  ys_train,
  # \(x, x_) periodic_kernel(x, x_, 2.729, 16.3, 0.326),
  # \(x, x_) periodic_kernel(x, x_, 2.729, 1.38, 0.326) + linear_kernel(x, x_, 1.3, 2),
  \(x, x_) periodic_kernel(x, x_, 5, map_estimate$period, 0.2), #+ linear_kernel(x, x_, 1.3, 0.5),
  # \(x, x_) periodic_kernel(x, x_, 6.5855, 1.5399, 4.1787) + linear_kernel(x, x_, -0.3467, 0.5012),
  sqrt(map_estimate$noise)
  # sqrt(0.142)
)

mu <- mu_sigma[[1]]
sigma <- mu_sigma[[2]]
ggplot() +
  geom_line(aes(x = xs, y = mu), colour = 'red') +
  geom_ribbon(aes(x = xs, ymin = mu - 1.96 * sqrt(diag(sigma)), ymax = mu + 1.96 * sqrt(diag(sigma))), alpha = 0.3) +
  geom_line(aes(x = xs, y = g(xs)), alpha = 0.5) +
  geom_point(aes(x = xs_train, y = ys_train))

ggsave('prediction_with_prior.png', width = 20, height = 8, units = "cm", dpi = 600)

mu_sigma <- pp(
  xs_test,
  xs_train,
  ys_train,
  # \(x, x_) periodic_kernel(x, x_, 2.729, 16.3, 0.326),
  # \(x, x_) periodic_kernel(x, x_, 2.729, 1.38, 0.326) + linear_kernel(x, x_, 1.3, 2),
  \(x, x_) periodic_kernel(x, x_, 5, map_estimate$period, 0.2), #+ linear_kernel(x, x_, 1.3, 0.5),
  # \(x, x_) periodic_kernel(x, x_, 6.5855, 1.5399, 4.1787) + linear_kernel(x, x_, -0.3467, 0.5012),
  sqrt(map_estimate$noise)
  # sqrt(0.142)
)

mu <- mu_sigma[[1]]
sigma <- mu_sigma[[2]]

dnorm(ys_test, mu, sqrt(diag(sigma))) |> log() |> sum()
