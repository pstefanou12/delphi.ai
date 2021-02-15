


class TruncatedMSE(ch.autograd.Function):
    """
    Computes the gradient of the negative population log likelihood for censored regression
    with known noise variance.
    """

    @staticmethod
    def forward(ctx, pred, targ):
        ctx.save_for_backward(pred, targ)
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ = ctx.saved_tensors
        # make args.num_samples copies of pred, N x B x 1
        stacked = pred[None, ...].repeat(config.args.num_samples, 1, 1)
        # add random noise to each copy
        noised = stacked + ch.randn_like(stacked)
        # filter out copies where pred is in bounds
        filtered = ch.stack([config.args.phi(batch).unsqueeze(1) for batch in noised]).float()
        # average across truncated indices
        out = (filtered * noised).sum(dim=0) / (filtered.sum(dim=0) + config.args.eps)
        return (out - targ) / pred.size(0), targ / pred.size(0)
        

class TruncatedUnknownVarianceMSE(ch.autograd.Function):
    """
    Computes the gradient of negative population log likelihood for truncated linear regression
    with unknown noise variance.
    """
    @staticmethod
    def forward(ctx, pred, targ, lambda_):
        ctx.save_for_backward(pred, targ, lambda_)
        return 0.5 * (pred.float() - targ.float()).pow(2).mean(0)

    @staticmethod
    def backward(ctx, grad_output):
        pred, targ, lambda_ = ctx.saved_tensors
        # calculate std deviation of noise distribution estimate
        sigma, z = ch.sqrt(lambda_.inverse()), Tensor([]).to(config.args.device)

        for i in range(pred.size(0)):
            # add random noise to logits
            noised = pred[i] + sigma*ch.randn(ch.Size([config.args.num_samples, 1])).to(config.args.device)
            # filter out copies within truncation set
            filtered = config.args.phi(noised).bool()
            z = ch.cat([z, noised[filtered.nonzero(as_tuple=False)][0]]) if ch.any(filtered) else ch.cat([z, pred[i].unsqueeze(0)])
        """
        multiply the v gradient by lambda, because autograd computes
        v_grad*x*variance, thus need v_grad*(1/variance) to cancel variance
        factor
        """
        out = (z - targ)
        return lambda_*out / (out.nonzero(as_tuple=False).size(0) + 1e-5), targ / (out.nonzero(as_tuple=False).size(0) + 1e-5),\
                (0.5 * targ.pow(2) - 0.5 * z.pow(2)) / (out.nonzero(as_tuple=False).size(0) + 1e-5)
