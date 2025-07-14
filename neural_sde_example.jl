using Plots, Statistics, ComponentArrays, Optimization, OptimizationOptimisers, DiffEqFlux,
      StochasticDiffEq, SciMLBase.EnsembleAnalysis, Random

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)


function trueSDEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

function true_noise_func(du, u, p, t)
    mp = Float32[0.2, 0.2]
    du .= mp .* u
end

prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)


# Take a typical sample from the mean
ensemble_prob = EnsembleProblem(prob_truesde; safetycopy = false)
ensemble_sol = solve(ensemble_prob, SOSRI(); trajectories = 10000)
ensemble_sum = EnsembleSummary(ensemble_sol)

sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))

drift_dudt = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
diffusion_dudt = Dense(2, 2)

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI();
    saveat = tsteps, reltol = 1e-1, abstol = 1e-1)
ps, st = Lux.setup(Xoshiro(0), neuralsde)
ps = ComponentArray(ps)

using Lux.Experimental: StatefulLuxLayer
import Lux.Experimental

# Get the prediction using the correct initial condition
prediction0 = neuralsde(u0, ps, st)[1]

drift_model = Lux.Experimental.StatefulLuxLayer(drift_dudt, ps.drift, st.drift)
diffusion_model = StatefulLuxLayer(diffusion_dudt, ps.diffusion, st.diffusion)

drift_(u, p, t) = drift_model(u, p.drift)
diffusion_(u, p, t) = diffusion_model(u, p.diffusion)

prob_neuralsde = SDEProblem(drift_, diffusion_, u0, (0.0f0, 1.2f0), ps)

ensemble_nprob = EnsembleProblem(prob_neuralsde; safetycopy = false)
ensemble_nsol = solve(ensemble_nprob, SOSRI(); trajectories = 100, saveat = tsteps)
ensemble_nsum = EnsembleSummary(ensemble_nsol)

plt1 = plot(ensemble_nsum; title = "Neural SDE: Before Training")
scatter!(plt1, tsteps, sde_data'; lw = 3)

scatter(tsteps, sde_data[1, :]; label = "data")
scatter!(tsteps, prediction0[1, :]; label = "prediction")


neuralsde_model = StatefulLuxLayer(neuralsde, ps, st)

function predict_neuralsde(p, u = u0)
    return Array(neuralsde_model(u, p))
end

function loss_neuralsde(p; n = 100)
    u = repeat(reshape(u0, :, 1), 1, n)
    samples = predict_neuralsde(p, u)
    currmeans = mean(samples; dims = 2)
    currvars = var(samples; dims = 2, mean = currmeans)[:, 1, :]
    currmeans = currmeans[:, 1, :]
    loss = sum(abs2, sde_data - currmeans) + sum(abs2, sde_data_vars - currvars)
    global means = currmeans
    global vars = currvars
    return loss
end

list_plots = []
iter = 0
u = repeat(reshape(u0, :, 1), 1, 100)
samples = predict_neuralsde(ps, u)
means = mean(samples; dims = 2)
vars = var(samples; dims = 2, mean = means)[:, 1, :]
means = means[:, 1, :]

# Callback function to observe training
callback = function (state, loss; doplot = false)
    global list_plots, iter, means, vars

    if iter == 0
        list_plots = []
    end
    iter += 1

    # loss against current data
    display(loss)

    # plot current prediction against data
    plt = Plots.scatter(tsteps, sde_data[1, :]; yerror = sde_data_vars[1, :],
        ylim = (-4.0, 8.0), label = "data")
    Plots.scatter!(plt, tsteps, means[1, :]; ribbon = vars[1, :], label = "prediction")
    push!(list_plots, plt)

    if doplot
        display(plt)
    end
    return false
end

opt = OptimizationOptimisers.Adam(0.025)

# First round of training with n = 10
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralsde(x; n = 10), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)
result1 = Optimization.solve(optprob, opt; callback, maxiters = 100)


opt = OptimizationOptimisers.Adam(0.001)
optf2 = Optimization.OptimizationFunction((x, p) -> loss_neuralsde(x; n = 1000), adtype)
optprob2 = Optimization.OptimizationProblem(optf2, result1.u)
result2 = Optimization.solve(optprob2, opt; callback, maxiters = 1000)

n = 1000
u = repeat(reshape(u0, :, 1), 1, n)
samples = predict_neuralsde(result2.u)
currmeans = mean(samples; dims = 2)
currvars = var(samples; dims = 2, mean = currmeans)[:, 1, :]
currmeans = currmeans[:, 1, :]

plt2 = Plots.scatter(tsteps, sde_data'; yerror = sde_data_vars', label = "data",
    title = "Neural SDE: After Training", xlabel = "Time")
plot!(plt2, tsteps, means'; lw = 8, ribbon = vars', label = "prediction")

plt = plot(plt1, plt2; layout = (2, 1))