

function pred = gpr_predict(post_params, xnew)

covfunc = {'covSum', {'covSEiso', 'covNoise'}};
loghyper = [1.8895   -0.6947   -1.3123]';

if isempty(post_params)
  t = 1;
else
  t = post_params(1, 1) + 1;
end

[mu, sigma] = gpr1step2(loghyper, covfunc, post_params(:, 1), post_params(:, 2), t);

pred = normpdf(xnew, mu, sqrt(sigma));
