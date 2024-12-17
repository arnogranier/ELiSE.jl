using MIDI;
using Distributions;
using Images;
using Random;
using ProgressMeter;

@kwdef struct Config
    dt::Float64=0.1
    E_exc::Float64=1.
    E_inh::Float64=-1.
    g_l::Float64=0.1
    g_den::Float64=2.
    g_e::Float64=0.3
    g_i::Float64=0.9
    δ::Float64=25.
    η::Float64=1e-5
    λ::Float64=0.6
    a::Float64=21.
    b::Float64=0.18
end

rho(u::Float64, a::Float64, b::Float64)::Float64 = 1. / (1. +exp(a*(b-u)))

function from_midi(filename::String;precision::Int=4)::Matrix{Bool}
    #check ms_per_tick
    fur_elise = readMIDIFile(filename)
    tpq = fur_elise.tpq ÷ precision
    notes = Notes(tpq=fur_elise.tpq)
    for track_nb in 2:length(fur_elise.tracks)
        append!(notes, getnotes(fur_elise, track_nb))
    end
    Target = zeros(Bool,128,(maximum([n.position+n.duration for n in notes])) ÷ tpq)
    for note in notes
        pos = Int(note.position)
        dur = Int(note.duration)
        Target[128-note.pitch, 1 + pos ÷ tpq : (pos+dur) ÷ tpq] .= true
    end
    return Target[findfirst(i->any(Target[i,:]),1:128):findlast(i->any(Target[i,:]),1:128),
                           findfirst(i->any(Target[:,i]),1:size(Target,2)):findlast(i->any(Target[:,i]),1:size(Target,2))]
end

function create_queues!(ds::Vector{Float64}, v::Float64, dt::Float64)::Vector{Vector{Float64}}
    qs = [Vector{Float64}() for _ in 1:length(ds)]
    initialize!((q,d)) = (x->push!(q,x)).(repeat([v],Int(d ÷ dt)))
    initialize!.(zip(qs, ds))
    return qs
end;

function memory!(m_den::Vector{Vector{Float64}},m_som_exc::Vector{Vector{Float64}},m_som_inh::Vector{Vector{Float64}},r::Vector{Float64})
    for (q_den, q_exc, q_inh, x) in zip(m_den, m_som_exc, m_som_inh, r)
        push!(q_den, x)
        push!(q_exc, x)
        push!(q_inh, x)
    end
end;

function step!(u::Vector{Float64},u_tgt::Vector{Float64},v::Vector{Float64},rbar::Vector{Float64},W::Matrix{Float64},B::Matrix{Float64},cfg::Config,
               m_den::Vector{Vector{Float64}},m_som_exc::Vector{Vector{Float64}},m_som_inh::Vector{Vector{Float64}};teacher::Bool=true)
    N = length(u)
    N_out = length(u_tgt)
    past_den_r = popfirst!.(m_den)
    I_den = W * past_den_r
    g_exc = cfg.g_e * B * popfirst!.(m_som_exc)
    g_inh = cfg.g_i * B * popfirst!.(m_som_inh)
    effective_alpha_t::Float64 = teacher ? (cfg.λ/(1. -cfg.λ)) * cfg.g_den : 0.
    for i in 1:N_out
        @inbounds u[i] += cfg.dt* (-(cfg.g_l+cfg.g_den)*u[i] + cfg.g_den * v[i] + effective_alpha_t *(u_tgt[i] - u[i]) )
    end
    for i in N_out+1:N
        @inbounds u[i] += cfg.dt* (-(cfg.g_l+cfg.g_den)*u[i] + cfg.g_den * v[i] + g_exc[i-N_out] * (cfg.E_exc - u[i]) + g_inh[i-N_out] * (cfg.E_inh - u[i]))
    end
    for i in 1:N
        @inbounds v[i] += cfg.dt*(-cfg.g_l*v[i] + I_den[i])
        @inbounds rbar[i] += cfg.dt*(-cfg.g_l*rbar[i] + past_den_r[i])
    end
    r = rho.(u, cfg.a, cfg.b)
    vstar = rho.((cfg.g_den/(cfg.g_l+cfg.g_den))*v, cfg.a, cfg.b)
    memory!(m_den, m_som_exc, m_som_inh, r)
    if teacher
        for i in 1:N, j in 1:N
            @inbounds W[i,j] += cfg.dt * cfg.η *(r[i]-vstar[i])*rbar[j]
        end
    end
    return nothing;
end;

function initialize(d_d::Vector{Float64}, d_s::Vector{Float64}, cfg::Config)
    N = length(d_d)
    v::Vector{Float64} = zeros(N)
    u::Vector{Float64} = zeros(N)
    rbar::Vector{Float64} = rho.(u, cfg.a, cfg.b);

    m_den::Vector{Vector{Float64}} = create_queues!(d_d, rho(0., cfg.a, cfg.b), cfg.dt)
    m_som_exc::Vector{Vector{Float64}} = create_queues!(d_s, rho(0., cfg.a, cfg.b), cfg.dt)
    m_som_inh::Vector{Vector{Float64}} = create_queues!(d_s .+ cfg.δ, rho(0., cfg.a, cfg.b), cfg.dt);
    return u, v, rbar, m_den, m_som_exc, m_som_inh
end

function train!(W::Matrix{Float64}, B::Matrix{Float64}, d_d::Vector{Float64}, d_s::Vector{Float64}, cfg::Config, target::Function, T_m::Float64, T_M::Float64, n_trials::Int)::Nothing
    u, v, rbar, m_den, m_som_exc, m_som_inh = initialize(d_d, d_s, cfg)

    @showprogress for trial in 0:n_trials-1
        for t in T_m:cfg.dt:T_M
            step!(u, target(t), v, rbar, W, B, cfg, m_den, m_som_exc, m_som_inh, teacher=true)
        end
    end

    return nothing;
end

function test!(W::Matrix{Float64}, B::Matrix{Float64},  d_d::Vector{Float64}, d_s::Vector{Float64}, cfg::Config, target::Function, T_m::Float64, T_M::Float64, metric::Function)
    u, v, rbar, m_den, m_som_exc, m_som_inh = initialize(d_d, d_s, cfg)
    
    for t in T_m:cfg.dt:T_M
        step!(u, target(t), v, rbar, W, B, cfg, m_den, m_som_exc, m_som_inh, teacher=true)
    end

    r_prod = Vector{Vector{Float64}}()
    for t in T_m:cfg.dt:T_M-1
        step!(u, target(t), v, rbar, W, B, cfg, m_den, m_som_exc, m_som_inh, teacher=false)
        isinteger(t) && push!(r_prod, rho.(u[1:N_out], cfg.a, cfg.b))
    end
    
    return metric(reduce(hcat, r_prod), reduce(hcat, [target(t) for t in T_m:T_M-1]))
end

function run!(W::Matrix{Float64}, B::Matrix{Float64},  d_d::Vector{Float64}, d_s::Vector{Float64}, cfg::Config, target::Function, T_m::Float64, T_M::Float64, n::Int, target_nudging::Function)
    u, v, rbar, m_den, m_som_exc, m_som_inh = initialize(d_d, d_s, cfg)
    
    r_prod = Vector{Vector{Float64}}()
    u_prod = Vector{Vector{Float64}}()
    v_prod = Vector{Vector{Float64}}()

    for trial in 0:n-1
        nudge = target_nudging(trial)
        for t in T_m:cfg.dt:T_M
            step!(u, target(t), v, rbar, W, B, cfg, m_den, m_som_exc, m_som_inh, teacher=nudge)
            if isinteger(t)
                push!(r_prod, rho.(u, cfg.a, cfg.b))
                push!(u_prod, u)
                push!(v_prod, v)
            end
        end
    end

    return reduce(hcat,r_prod), reduce(hcat,u_prod), reduce(hcat,v_prod)
end

plot(x::Matrix{Float64};ratio=(1,1)) = imresize(Gray.(1. .- x), ratio=ratio)
plot(x::Matrix{Float64}, θ::Float64;ratio=(1,1)) = imresize(Gray.(1. .- (x .> θ)), ratio=ratio)
function plot(x::Matrix{Float64}, T_m::Float64, T_M::Float64, n::Int;ratio=(1,1))
    p = imresize(Gray.(1. .- x), ratio=ratio)
    p[:,[i*round(Int,1+(T_M-T_m)) for i in 1:n-1]] .= colorant"violet"
    return imresize(p, ratio=ratio)
end
function plot(x::Matrix{Float64}, T_m::Float64, T_M::Float64, n::Int, θ::Float64;ratio=(1,1))
    p = imresize(Gray.(1. .- (x .> θ)), ratio=ratio)
    p[:,[i*round(Int,1+(T_M-T_m)) for i in 1:n-1]] .= colorant"violet"
    return imresize(p, ratio=ratio)
end
function plot(x::Matrix{Float64}, target::Function, T_m::Float64, T_M::Float64, n::Int, θ::Float64;ratio=(1,1))
    p = plot(x, reduce(hcat,repeat([target(t) for t in T_m:T_M],n)), θ)
    p[:,[i*round(Int,1+(T_M-T_m)) for i in 1:n-1]] .= colorant"violet"
    return imresize(p, ratio=ratio)
end
function plot(x::Matrix{Float64}, y::Matrix{Float64}, θ::Float64)
    colors = [colorant"white", colorant"red", colorant"cyan", colorant"black"]
    return colors[1 .+ 2*(x.>θ) + (y.>θ)]
end