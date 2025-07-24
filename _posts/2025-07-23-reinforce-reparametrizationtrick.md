---
title: score-function estimator vs reparametrization trick
date: 2025-07-23 17:00:00 -0700
categories: [Math]
tags: [optimzation, RL, MC]
math: true
---
In many machine learning problems, our objective is to optimize an expectation. A classic example is Reinforcement Learning (RL), where the goal is to find a policy that maximizes the expected future reward.

Mathematically, we're trying to optimize an objective $J(\theta)$ defined as:
$$
J(\theta) = \mathbb{E}_{x \sim p_{\theta}(x)}[f(x)]
$$
Our goal is to find the parameters $\theta$ of the distribution $p_{\theta}(x)$ that maximize or minimize this expected value.

As machine learning practitioners, we'd love to use powerful methods like gradient ascent or descent for this optimization. This requires computing the gradient of our objective with respect to $\theta$.

Let's write out the gradient:
$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{x \sim p_{\theta}(x)}[f(x)] = \nabla_{\theta} \int p_{\theta}(x) f(x) dx
$$
The challenge is that the parameter $\theta$ we're differentiating with respect to is part of the probability distribution from which we sample. If we push the gradient inside the integral, we get:
$$
\int (\nabla_{\theta} p_{\theta}(x)) f(x) dx
$$
This is where we hit a wall. This expression is no longer an expectation with respect to the original distribution $p_{\theta}(x)$, so we can't directly use standard Monte Carlo estimation, which looks like this:
$$
\mathbb{E}_{x \sim p_{\theta}(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_{i}), \quad \text{where } x_i \sim p_\theta(x)
$$
We need a way to rewrite the gradient into a form that \textit{is} an expectation, allowing us to approximate it easily with samples. This is where clever techniques like the **score-function estimator (REINFORCE)** and the **reparameterization trick** come to the rescue.
