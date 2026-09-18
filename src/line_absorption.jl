using SpecialFunctions: gamma
using ProgressMeter: @showprogress
using Base.Iterators: partition

"""
    line_absorption!(α, linelist, λs, temp, nₑ, n_densities, partition_fns, ξ
                   ; α_cntm=nothing, cutoff_threshold=1e-3, window_size=20.0*1e-8)

Calculate the opacity coefficient, `α`, in units of cm^-1 from all lines in `linelist`, at
wavelengths `λs`.

# Arguments

  - `α`: absorption coefficient matrix to be filled in-place

  - `linelist`: vector of [`Korg.Line`](@ref)s.
  - `λs`: a [`Korg.Wavelengths`](@ref) object.
  - `temp` the temperature in K (as a vector, for multiple layers, if you like)
  - `n_densities`, a Dict mapping species to absolute number density in cm^-3 (as a vector, if temp is
    a vector).
  - `partition_fns`, a Dict containing the partition function of each species
  - `ξ` is the microturbulent velocity in cm/s (n.b. NOT km/s)
  - `α_cntm` is as a callable returning the continuum opacity as a function of wavelength. The window
    within which a line is calculated will extend to the wavelength at which the Lorentz wings or
    Doppler core of the line are at `cutoff_threshold * α_cntm[line.wl]`, whichever is greater.

# Keyword Arguments

  - `cuttoff_threshold` (default: 3e-4): see `α_cntm`
  - `tasks_per_thread` (default: 1): the number of tasks to run per Julia thread. This function
    is multithreaded over the lines in `linelist`.
  - `nlte_tags`, `b_lower`, `b_upper`, `dev`: NLTE departure coefficients, all `nothing` (the
    default) for a pure-LTE calculation.  `nlte_tags` is a vector parallel to `linelist` giving
    each line's transition index (0 for an LTE line, see [`Korg.nlte_tags`](@ref)); `b_lower` and
    `b_upper` are (transitions × layers) matrices of b_l and b_u; `dev` is a matrix shaped like `α`
    which is filled in-place with `Σᵢ κᵢ(rᵢ − 1)`, the deviation of the line emissivity from LTE.
    Passing `dev` is what turns NLTE on, and all four must be given together.  See `nlte.jl`.
"""
function line_absorption!(α, linelist, λs::Wavelengths, temps, nₑ, n_densities, partition_fns, ξ,
                          α_cntm; cutoff_threshold=3e-4, tasks_per_thread=1,
                          nlte_tags=nothing, b_lower=nothing, b_upper=nothing, dev=nothing)
    if length(linelist) == 0
        return zeros(length(λs))
    end

    # NLTE is on only when a deviation accumulator was handed in.  When it is off, the line loop
    # pays one integer compare per line and nothing else; see `nlte.jl` for the physics.
    do_nlte = !isnothing(dev)
    if do_nlte && (isnothing(nlte_tags) || isnothing(b_lower) || isnothing(b_upper))
        throw(ArgumentError("`dev` was passed without `nlte_tags`/`b_lower`/`b_upper`"))
    end

    β = @. 1 / (kboltz_eV * temps)

    # precompute number density / partition function for each species in the linelist
    n_div_U = map(unique([l.species for l in linelist])) do spec
        spec => @. (n_densities[spec] / partition_fns[spec](log(temps)))
    end |> Dict
    if species"H I" in keys(n_div_U)
        @error "Atomic hydrogen should not be in the linelist. Korg has built-in hydrogen lines."
    end

    # precompute the effective perturber number density for vdW broadening.
    # Coefficients from polarizability ratio and thermal velocity of He and H₂ relative to H.
    # (a/a_H)^0.4 + (m/m_H)^-0.3
    # species beyond these three are not important.
    # these are in a.u. (not Å), from https://doi.org/10.1080/00268976.2018.1535143
    a_H = polarizability_H_au
    a_He = polarizability_He_au
    a_H2 = polarizability_H2_au   # ± 0.049 a.u. (in text, not Table 1)
    c_He = (a_He / a_H)^0.4 * (atomic_masses[2] / atomic_masses[1])^-0.3
    c_H2 = (a_H2 / a_H)^0.4 * 2^-0.3
    n_eff_vdW = @. n_densities[species"H_I"] + c_He * n_densities[species"He_I"] +
                   c_H2 * n_densities[species"H2"]

    # Molecular lines of species with ExoMol/Gharib-Nezhad+21 broadening data don't use n_eff_vdW at
    # all: their perturbers are summed individually, with each species' own temperature exponents.
    # Γ_vdW is line-independent for these, so it's computed once per species here rather than once
    # per line below.  See molecular_broadening.jl.
    molecular_Γ_vdW = molecular_vdW_Γs(linelist, temps, n_densities)

    n_chunks = tasks_per_thread * Threads.nthreads()
    chunk_size = max(1, length(linelist) ÷ n_chunks + (length(linelist) % n_chunks > 0))
    # chunk over INDICES, not lines, so that each line can find its own NLTE tag
    linelist_chunks = partition(eachindex(linelist), chunk_size)
    tasks = map(linelist_chunks) do index_chunk
        # Each chunk of your data gets its own spawned task that does its own local, sequential work
        # and then returns the result
        Threads.@spawn begin
            α_task = zeros(eltype(α), size(α))
            # the companion to α_task holding Σᵢ κᵢ(rᵢ − 1), rᵢ = Sᵢ/B_ν.  Only lines with
            # departure coefficients ever write to it, so it stays identically zero away from them.
            # Allocated only when NLTE is on: it is the same size as α_task, and there is no reason
            # to pay that in production.
            dev_task = do_nlte ? zeros(eltype(dev), size(dev)) : nothing

            # preallocate some arrays for the core loop.
            # Each element of the arrays corresponds to an atmospheric layer, same at the "temps" array and
            # the values in "number_densities"
            Γ = Vector{eltype(α)}(undef, size(temps))
            γ = Vector{eltype(α)}(undef, size(temps))
            σ = Vector{eltype(α)}(undef, size(temps))
            amplitude = Vector{eltype(α)}(undef, size(temps))
            levels_factor = Vector{eltype(α)}(undef, size(temps))
            fkappa = do_nlte ? Vector{eltype(α)}(undef, size(temps)) : nothing
            fdev = do_nlte ? Vector{eltype(α)}(undef, size(temps)) : nothing
            ρ_crit = Vector{eltype(α)}(undef, size(temps))
            inverse_densities = Vector{eltype(α)}(undef, size(temps))

            for line_index in index_chunk
                line = linelist[line_index]
                m = get_mass(line.species)

                # doppler-broadening width, σ (NOT √[2]σ)
                σ .= doppler_width.(line.wl, temps, m, ξ)

                # sum up the damping parameters.  These are FWHM (γ is usually the Lorentz HWHM) values in
                # angular, not cyclical frequency (ω, not ν).
                Γ .= line.gamma_rad
                if haskey(molecular_Γ_vdW, line.species)
                    # a molecule with per-species ExoMol data: Korg's own treatment overrides
                    # whatever vdW width the linelist supplied.  See molecular_broadening.jl.
                    Γ .+= molecular_Γ_vdW[line.species]
                else
                    # atoms, and molecules absent from the table, use the linelist's γ_vdW
                    # (or ABO σ, α) against the effective perturber density.
                    Γ .+= n_eff_vdW .* scaled_vdW.(Ref(line.vdW), m, temps)   # now also for molecules
                end
                if !ismolecule(line.species)
                    @. Γ += nₑ * scaled_stark.(line.gamma_stark, temps)
                end
                # calculate the lorentz broadening parameter in wavelength. Doing this involves an
                # implicit aproximation that λ(ν) is linear over the line window.
                # the factor of λ²/c is |dλ/dν|, the factor of 1/2π is for angular vs cyclical freqency,
                # and the last factor of 1/2 is for FWHM vs HWHM
                @. γ = Γ * line.wl^2 / (c_cgs * 4π)

                E_upper = line.E_lower + c_cgs * hplanck_eV / line.wl

                # Departure coefficients, if this line has them.  The LTE `levels_factor` is
                # exp(−βE_l) − exp(−βE_u); the NLTE opacity is the same expression with each term
                # weighted by its level's b, which is κ_LTE·b_l[1 − (b_u/b_l)e^−x]/[1 − e^−x]
                # exactly.  Applied BEFORE the cutoff test below, so a line weakened out of
                # relevance by NLTE is dropped on its true strength.
                k_nlte = do_nlte ? nlte_tags[line_index] : 0
                if k_nlte > 0
                    nlte_line_factors!(fkappa, fdev, view(b_lower, k_nlte, :),
                                       view(b_upper, k_nlte, :), β, line.E_lower, E_upper)
                    @. levels_factor = (exp(-β * line.E_lower) - exp(-β * E_upper)) * fkappa
                else
                    @. levels_factor = exp(-β * line.E_lower) - exp(-β * E_upper)
                end

                #total wl-integrated absorption coefficient
                @. amplitude = 10.0^line.log_gf * sigma_line(line.wl) * levels_factor *
                               n_div_U[line.species]

                ρ_crit .= (line.wl .|> α_cntm) .* cutoff_threshold ./ amplitude
                inverse_densities .= inverse_gaussian_density.(ρ_crit, σ)
                doppler_line_window = maximum(inverse_densities)
                inverse_densities .= inverse_lorentz_density.(ρ_crit, γ)
                lorentz_line_window = maximum(inverse_densities)
                window_size = sqrt(lorentz_line_window^2 + doppler_line_window^2)
                lb = searchsortedfirst(λs, line.wl - window_size)
                ub = searchsortedlast(λs, line.wl + window_size)
                # not necessary, but is faster as of 8f979cc2c28f45cd7230d9ee31fbfb5a5164eb1d
                if lb > ub
                    continue
                end

                α_task[:, lb:ub] .+= line_profile.(line.wl, σ, γ, amplitude, view(λs, lb:ub)')
                if k_nlte > 0
                    # `fdev` is per unit of the already-scaled opacity, so the deviation is the
                    # same profile with the amplitude scaled — one extra broadcast, for a handful
                    # of lines out of millions.
                    dev_task[:, lb:ub] .+= line_profile.(line.wl, σ, γ, amplitude .* fdev,
                                                         view(λs, lb:ub)')
                end
            end
            return α_task, dev_task
        end
    end

    results = fetch.(tasks)
    α .+= sum(first.(results))
    if do_nlte
        dev .+= sum(last.(results))
    end
    return nothing
end

"""
    inverse_gaussian_density(ρ, σ)

Calculate the inverse of a (0-centered) Gaussian PDF with standard deviation `σ`, i.e. the value of
`x` for which `ρ = exp(-0.5 x^2/σ^2}) / √[2π]`, which is given by `σ √[-2 log (√[2π]σρ)]`.  Returns
0 when ρ is larger than any value taken on by the PDF.

See also: [`inverse_lorentz_density`](@ref).
"""
inverse_gaussian_density(ρ, σ) =
    if ρ > 1 / (sqrt(2π) * σ)
        0.0
    else
        σ * sqrt(-2log(sqrt(2π) * σ * ρ))
    end

"""
    inverse_lorentz_density(ρ, γ)

Calculate the inverse of a (0-centered) Lorentz PDF with width `γ`, i.e. the value of `x` for which
`ρ = 1 / (π γ (1 + x^2/γ^2))`, which is given by `√[γ/(πρ) - γ^2]`. Returns 0 when ρ is larger than
any value taken on by the PDF.

See also: [`inverse_gaussian_density`](@ref).
"""
inverse_lorentz_density(ρ, γ) =
    if ρ > 1 / (π * γ)
        0.0
    else
        sqrt(γ / (π * ρ) - γ^2)
    end

"""
    exponential_integral_1(x)

Compute the first exponential integral, E1(x).  This is a rough approximation lifted from Kurucz's
VCSE1F. Used in `brackett_line_profile`.
"""
function exponential_integral_1(x)
    if x < 0
        0.0
    elseif x <= 0.01
        -log(x) - 0.577215 + x
    elseif x <= 1.0
        -log(x) - 0.57721566 +
        x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 + x * (-0.00976004 + x * 0.00107857))))
    elseif x <= 30.0
        (x * (x + 2.334733) + 0.25062) / (x * (x + 3.330657) + 1.681534) / x * exp(-x)
    else
        0.0
    end
end

"""
    doppler_width(λ₀ T, m, ξ)

The standard deviation of of the doppler-broadening profile.  In standard spectroscopy texts, the
Doppler width often refers to σ√2, but this is σ
"""
doppler_width(λ₀, T, m, ξ) = λ₀ * sqrt(kboltz_cgs * T / m + (ξ^2) / 2) / c_cgs

"""
the stark broadening gamma scaled acording to its temperature dependence
"""
scaled_stark(γstark, T; T₀=10_000) = γstark * (T / T₀)^(1 / 6)

"""
    scaled_vdW(vdW, m, T)

The vdW broadening gamma scaled acording to its temperature dependence, using either simple scaling
or ABO. See Anstee & O'Mara (1995) or
[Paul Barklem's notes](https://github.com/barklem/public-data/tree/master/broadening-howto) for the
definition of the ABO γ.

`vdW` should be either `γ_vdW` evaluated at 10,000 K, or tuple containing the ABO params `(σ, α)`.
The species mass, `m`, is ignored in the former case.
"""
function scaled_vdW(vdW::Tuple{F,F}, m, T) where F<:Real
    if vdW[2] == -1
        return vdW[1] * (T / 10_000)^0.3
    else
        v₀ = 1e6 #σ is given at 10_000 m/s = 10^6 cm/s
        σ = vdW[1]
        α = vdW[2]

        invμ = 1 / (1.008 * amu_cgs) + 1 / m #inverse reduced mass
        vbar = sqrt(8 * kboltz_cgs * T / π * invμ) #relative velocity
        #n.b. "gamma" is the gamma function, not a broadening parameter
        2 * (4 / π)^(α / 2) * gamma((4 - α) / 2) * v₀ * σ * (vbar / v₀)^(1 - α)
    end
end

"""
    sigma_line(wl)

The cross-section (divided by gf) at wavelength `wl` in Ångstroms of a transition for which the
product of the degeneracy and oscillator strength is `10^log_gf`.
"""
function sigma_line(λ::Real)
    #work in cgs
    e = electron_charge_cgs
    mₑ = electron_mass_cgs
    c = c_cgs

    #the factor of |dλ/dν| = λ²/c is because we are working in wavelength rather than frequency
    (π * e^2 / mₑ / c) * (λ^2 / c)
end

"""
    line_profile(λ₀, σ, γ, amplitude, λ)

A voigt profile centered on λ₀ with Doppler width σ (NOT √[2] σ, as the "Doppler width" is often
defined) and Lorentz HWHM γ evaluated at `λ` (cm).  Returns values in units of cm^-1.
"""
function line_profile(λ₀::Real, σ::Real, γ::Real, amplitude::Real, λ::Real)
    inv_σsqrt2 = 1 / (σ * sqrt(2))
    scaling = inv_σsqrt2 / sqrt(π) * amplitude
    voigt_hjerting(γ * inv_σsqrt2, abs(λ - λ₀) * inv_σsqrt2) * scaling
end

@inline function harris_series(v) # assume v < 5
    v2 = v * v
    H₀ = exp(-(v2))
    H₁ = if (v < 1.3)
        -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
    elseif v < 2.4
        -4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v
    else #v < 5
        ((0.554153432 +
          (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
         (v2 - 3 / 2))
    end
    H₂ = (1 - 2v2) * H₀
    H₀, H₁, H₂
end

"""
    voigt_hjerting(α, v)

The [Hjerting function](https://en.wikipedia.org/wiki/Voigt_profile#Voigt_functions), ``H``,
somtimes called the Voigt-Hjerting function. ``H`` is defined as
`H(α, v) = ∫^∞_∞ exp(-y^2) / ((u-y)^2 + α^2) dy`
(see e.g. the unnumbered equation after Gray equation 11.47).  It is equal to the ratio of the
absorption coefficient to the value of the absorption coefficient obtained at the line center with
only Doppler broadening.

If `x = λ-λ₀`, `Δλ_D = σ√2` is the Doppler width, and `Δλ_L = 4πγ` is the Lorentz width,

```
voigt(x|Δλ_D, Δλ_L) = H(Δλ_L/(4πΔλ_D), x/Δλ_D) / (Δλ_D√π)
                    = H(γ/(σ√2), x/(σ√2)) / (σ√(2π))
```

Approximation from [Hunger 1965](https://ui.adsabs.harvard.edu/abs/1956ZA.....39...36H/abstract).
"""
function voigt_hjerting(α, v)
    v2 = v * v
    if α <= 0.2 && (v >= 5)
        invv2 = (1 / v2)
        (α / sqrt(π) * invv2) * (1 + 1.5invv2 + 3.75 * invv2^2)
    elseif α <= 0.2 #v < 5
        H₀, H₁, H₂ = harris_series(v)
        H₀ + (H₁ + H₂ * α) * α
    elseif (α <= 1.4) && (α + v < 3.2)
        #modified harris series: M_i is H'_i in the source text
        H₀, H₁, H₂ = harris_series(v)
        M₀ = H₀
        M₁ = H₁ + 2 / sqrt(π) * M₀
        M₂ = H₂ - M₀ + 2 / sqrt(π) * M₁
        M₃ = 2 / (3sqrt(π)) * (1 - H₂) - (2 / 3) * v2 * M₁ + (2 / sqrt(π)) * M₂
        M₄ = 2 / 3 * v2 * v2 * M₀ - 2 / (3sqrt(π)) * M₁ + 2 / sqrt(π) * M₃
        ψ = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * α) * α) * α
        ψ * (M₀ + (M₁ + (M₂ + (M₃ + M₄ * α) * α) * α) * α)
    else #α > 1.4 or (α > 0.2 and α + v > 3.2)
        r2 = (v2) / (α * α)
        α_invu = 1 / sqrt(2) / ((r2 + 1) * α)
        α2_invu2 = α_invu * α_invu
        sqrt(2 / π) * α_invu * (1 + (3 * r2 - 1 + ((r2 - 2) * 15 * r2 + 2) * α2_invu2) * α2_invu2)
    end
end
