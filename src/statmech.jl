using ForwardDiff

"""
    saha_ion_weights(T, nₑ, atom, ionization_energies, partition_functions)

Returns `(wII, wIII)`, where `wII` is the ratio of singly ionized to neutral atoms of a given

element, and `wIII` is the ratio of doubly ionized to neutral atoms.

arguments:

  - temperature `T` [K]
  - electron number density `nₑ` [cm^-3]
  - atom, the atomic number of the element
  - `ionization_energies` is a collection indexed by integers (e.g. a `Vector`) mapping elements'
    atomic numbers to their first three ionization energies
  - `partition_funcs` is a `Dict` mapping species to their partition functions
"""
function saha_ion_weights(T, nₑ, atom, ionization_energies, partition_funcs::Dict)
    χI, χII, χIII = ionization_energies[atom]
    atom = Formula(atom)
    UI = partition_funcs[Species(atom, 0)](log(T))
    UII = partition_funcs[Species(atom, 1)](log(T))

    k = kboltz_eV
    transU = translational_U(electron_mass_cgs, T)

    wII = 2.0 / nₑ * (UII / UI) * transU * exp(-χI / (k * T))
    wIII = if atom == Formula(1) # hydrogen
        0.0
    else
        UIII = partition_funcs[Species(atom, 2)](log(T))
        wII * 2.0 / nₑ * (UIII / UII) * transU * exp(-χII / (k * T))
    end
    wII, wIII
end

"""
    translational_U(m, T)

The (possibly inverse) contribution to the partition function from the free movement of a particle.
Used in the Saha equation.

arguments

  - `m` is the particle mass
  - `T` is the temperature in K
"""
function translational_U(m, T)
    k = kboltz_cgs
    h = hplanck_cgs
    (2π * m * k * T / h^2)^1.5
end

"""
    get_log_nK(mol, T, log_equilibrium_constants)

Given a molecule, `mol`, a temperature, `T`, and a dictionary of log equilibrium constants in partial
pressure form, return the base-10 log equilibrium constant in number density form, i.e. `log10(nK)`
where `nK = n(A)n(B)/n(AB)`.
"""
function get_log_nK(mol, T, log_equilibrium_constants)
    log_equilibrium_constants[mol](log(T)) - (n_atoms(mol) - 1) * log10(kboltz_cgs * T)
end

"""
    Hminus_nK(T)

Returns `nK` for the H⁻ formation reaction H I + e⁻ → H⁻, i.e. `n(H⁻) = nK(T) * n(H I) * nₑ`.

The partition function of H⁻ is 1 (singlet ground state only), and the statistical weight of the
free electron is 2. Note that unlike the molecular equilibrium constants, this is in terms
of number densities, not partial pressures.
"""
function Hminus_nK(T)
    χ_ea = 0.754204 # [eV] electron affinity used by McLaughlin+ 2017 H⁻ ff cross sections
    # inverse of translational_U for the electron, times U(H⁻)/U(H I) = 1/2, times exp(χ_ea/kT)
    exp(χ_ea / (kboltz_eV * T)) / (4 * translational_U(electron_mass_cgs, T))
end

struct ChemicalEquilibriumError <: Exception
    msg::String
end
function Base.showerror(io::IO, e::ChemicalEquilibriumError)
    print(io, "Chemical equilibrium failed: ", e.msg)
end

"""
    chemical_equilibrium(T, nₜ, nₑ, absolute_abundances, ionization_energies,
                         partition_fns, log_equilibrium_constants; x0=nothing)

Iteratively solve for the number density of each species. Returns a pair containing the electron
number density and `Dict` mapping species to number densities.

arguments:

  - the temperature, `T`, in K
  - the total number density `nₜ`
  - the electron number density `nₑ`
  - A Dict of `absolute_abundances`, N_X/N_total
  - a Dict of ionization energies, `ionization_energies`.  The keys of act as a list of all atoms.
  - a Dict of partition functions, `partition_fns`
  - a Dict of log molecular equilibrium constants, `log_equilibrium_constants`, in partial pressure form.
    The keys of `equilibrium_constants` act as a list of all molecules.

keyword arguments:

  - `x0` (default: `nothing`) is an initial guess for the solution (in a format internal to

    `chemical_equilibrium`). If not supplied, a good guess is computed by neglecting molecules.

  - `electron_number_density_warn_threshold` (default: `0.1`) is the fractional difference between
    the calculated electron number density and the model atmosphere electron number density at which
    a warning is issued.
  - `electron_number_density_warn_min_value` (default: `1e-4`) is the minimum value of the electron
    number density at which a warning is issued.  This is to avoid warnings when the electron number

    density is very small.

The system of equations is specified with the number densities of the neutral atoms as free
parameters.  Each equation specifies the conservation of a particular species, e.g. (simplified)

    n(O) = n(CO) + n(OH) + n(O I) + n(O II) + n(O III).

for oxygen.  There is also one charge balance equation:

    0 = -n(e) - n(H-) + n(H II) + 2n(H III) + ...

In these equations:

  - `n(O)` is the number density of oxygen atoms in any form.

  - `n(O I)` is a free parameter.  The numerical solver is varying this to satisfy the system of
    equations.
  - `n(O II)`, and `n(O III)` come from the Saha (ionization) equation given `n(O I)`
  - `n(CO)` and `n(OH)` come from the molecular equilibrium constants K, which are precomputed
    over a range of temperatures.
  - `n(e)` is a free parameter. (Thus the
    `electron_number_density_warn_threshold` kwarg, which warns you when the
    `n(e)` that the solver arrives at differs from that of the model atmosphere.)
  - Like for oxygen, `n(H I)` is a free parameter, and `n(H-)`, `n(H II)`, and `n(H III)` are
    computed from the free parameters.

Equilibrium constants are defined in terms of partial pressures, so e.g.

    K(OH)  ==  (p(O) p(H)) / p(OH)  ==  (n(O) n(H)) / n(OH)) kT
"""
function chemical_equilibrium(temp, nₜ, model_atm_nₑ, absolute_abundances, ionization_energies,
                              partition_fns, log_equilibrium_constants;
                              electron_number_density_warn_threshold=0.1,
                              electron_number_density_warn_min_value=1e-4)
    #compute good first guess by neglecting molecules
    neutral_fraction_guess = map(1:MAX_ATOMIC_NUMBER) do Z
        wII, wIII = saha_ion_weights(temp, model_atm_nₑ, Z, ionization_energies, partition_fns)
        1 / (1 + wII + wIII)
    end

    #actually run the core solver, get back a "zero" for that system of equations
    solver_zero = _solve_chemical_equilibrium(temp, nₜ, absolute_abundances, neutral_fraction_guess,
                                              model_atm_nₑ, ionization_energies, partition_fns,
                                              log_equilibrium_constants)

    # now put the solution from the solver in directly usable form
    nₑ = 10.0^solver_zero[end]

    if ((nₑ / nₜ > electron_number_density_warn_min_value) &&
        (abs((nₑ - model_atm_nₑ) / model_atm_nₑ) > electron_number_density_warn_threshold))
        @warn "Electron number density differs from model atmosphere by a factor greater than $electron_number_density_warn_threshold. (calculated nₑ = $nₑ, model atmosphere nₑ = $model_atm_nₑ)"
    end

    # neutral atomic species are what's solved for
    n_neutral = 10.0 .^ solver_zero[1:(end-1)]
    number_densities = Dict(Species.(Formula.(1:MAX_ATOMIC_NUMBER), 0) .=> n_neutral)

    #now the ionized atomic species
    for a in 1:MAX_ATOMIC_NUMBER
        wII, wIII = saha_ion_weights(temp, nₑ, a, ionization_energies, partition_fns)
        number_densities[Species(Formula(a), 1)] = wII * number_densities[Species(Formula(a), 0)]
        number_densities[Species(Formula(a), 2)] = wIII * number_densities[Species(Formula(a), 0)]
    end

    number_densities[species"H-"] = Hminus_nK(temp) * number_densities[species"H I"] * nₑ

    #now the molecules
    for mol in keys(log_equilibrium_constants)
        log_nK = get_log_nK(mol, temp, log_equilibrium_constants)
        element_log_ns = if mol.charge == 0
            (log10(number_densities[Species(Formula(el), 0)]) for el in get_atoms(mol.formula))
        else # singly ionized diatomic
            Z1, Z2 = get_atoms(mol.formula)
            # the first atom has the lower atomic number.  That is the charged component for out Ks.
            log10(number_densities[Species(Formula(Z1), 1)]) +
            log10(number_densities[Species(Formula(Z2), 0)])
        end
        number_densities[mol] = 10^(sum(element_log_ns) - log_nK)
    end

    nₑ, number_densities
end

# This function sets up the inital parameter guess, then runs the nonlinear solver with the 
# appropriate method
#
# We solve in log10 number density space via damped Newton with continuation on the molecular
# equilibrium and H-.  At ξ → 0 (logξ → -∞) molecules and H- are suppressed and the system 
# reduces to atomic Saha (easy to solve). We then anneal logξ → 0. This is required for cool/dense 
# regimes (T ≲ 3000 K, nₜ ≳ 1e15) where the atoms-only guess massively over-predicts molecule 
# densities. 
function _solve_chemical_equilibrium(temp, nₜ, absolute_abundances, neutral_fraction_guess,
                                     nₑ_guess, ionization_energies, partition_fns,
                                     log_equilibrium_constants;
                                     ftol=1e-8, minimum_annealing_Δlogξ=0.0625)
    n_neutral_guess = (nₜ - nₑ_guess) .* absolute_abundances .* neutral_fraction_guess
    x0 = [log10.(n_neutral_guess); log10(nₑ_guess)]

    # this wacky maneuver ensures that x0 has the appropriate dual number type for autodiff
    # if that is going on.  I'm sure there's a better way...
    x0 = x0 .* (absolute_abundances[1] / absolute_abundances[1])

    # Try to solve first with full molecules (logξ=0). For warm and/or low-density regimes this 
    # converges in a handful of iterations.  If it fails, we use the continuation.
    residuals_full! = setup_chemical_equilibrium_residuals(temp, nₜ, absolute_abundances,
                                                           ionization_energies, partition_fns,
                                                           log_equilibrium_constants, 0.0)
    x, converged,
    last_inf = clipped_newton(residuals_full!, copy(x0); tol=ftol, max_iter=50,
                              max_step=1.0)
    if !converged
        # Anneal logξ from -50 (molecules numerically off for Float64s) up to 0, bisecting the
        # interval whenever Newton fails so that we take only as many sub-steps as the regime
        # demands.  Each successful level warm-starts the next.
        function solve_at(logξ, xstart)
            r! = setup_chemical_equilibrium_residuals(temp, nₜ, absolute_abundances,
                                                      ionization_energies, partition_fns,
                                                      log_equilibrium_constants, logξ)
            clipped_newton(r!, copy(xstart); tol=ftol, max_iter=100, max_step=1.0)
        end

        # First find a starting value of ξ that works, but isn't too small.
        local x_anchor, anchor_converged, last_inf, logξ_anchor
        for λ_try in [-5.0, -20.0, -50.0]
            x_anchor, anchor_converged, last_inf = solve_at(λ_try, x0)
            if anchor_converged
                logξ_anchor = λ_try
                break
            end
        end
        anchor_converged ||
            throw(ChemicalEquilibriumError("unconverged at anchor (inf-norm=$last_inf)"))

        # Now solve system repeatedly, annealing ξ up to 1. Start with Δlogξ = 2, bisecting if that 
        # fails.
        x = x_anchor
        Δlogξ_init = 2.0
        Δlogξ = Δlogξ_init
        logξ = logξ_anchor
        while logξ < -1e-12
            step = min(Δlogξ, 0 - logξ)
            x_try, conv, last_inf = solve_at(logξ + step, x)
            if !conv # step failed
                # bisect down in step size
                step /= 2
                while !conv && step >= minimum_annealing_Δlogξ
                    x_try, conv, last_inf = solve_at(logξ + step, x)
                    conv || (step /= 2)
                end
                conv ||
                    throw(ChemicalEquilibriumError("unconverged at logξ=$(logξ+ 2step) (inf-norm=$last_inf)"))

                # shrink default step after a failure
                Δlogξ = step
            else # step succeeded
                Δlogξ = min(Δlogξ_init, 2step)  # double step (capped to init value) after success
            end
            x = x_try
            logξ += step
        end
    end

    if !all(isfinite, x)
        throw(ChemicalEquilibriumError("solution contains non-finite values"))
    end

    x
end

# handle the case where a derivative is being taken with respect to T and ntot, but not abundances
function _solve_chemical_equilibrium(temp::ForwardDiff.Dual{T,V1,P},
                                     nₜ::ForwardDiff.Dual{T,V2,P},
                                     absolute_abundances::Vector{F},  # not duals!
                                     neutral_fraction_guess::Vector{ForwardDiff.Dual{T,V3,P}},
                                     nₑ_guess, # this type doesn't matter if the solver converges
                                     # Require that the types of the following be the same as the
                                     # default, thus containing no duals. This is over-restrictive,
                                     # but I'm not sure how to enforce the more general condition
                                     # via the type system.
                                     ionization_energies::typeof(Korg.ionization_energies),
                                     partition_fns::typeof(Korg.default_partition_funcs),
                                     log_equilibrium_constants::typeof(Korg.default_log_equilibrium_constants)) where {
                                                                                                                       T,
                                                                                                                       V1,
                                                                                                                       V2,
                                                                                                                       V3,
                                                                                                                       P,
                                                                                                                       F<:AbstractFloat
                                                                                                                       }
    vtemp = ForwardDiff.value(temp)
    vnₜ = ForwardDiff.value(nₜ)
    vneutral_fraction_guess = ForwardDiff.value.(neutral_fraction_guess)
    vnₑ_guess = ForwardDiff.value(nₑ_guess)

    ptemp = ForwardDiff.partials(temp)
    pnₜ = ForwardDiff.partials(nₜ)
    partials = [ptemp pnₜ]'

    zero = _solve_chemical_equilibrium(vtemp, vnₜ, absolute_abundances, vneutral_fraction_guess,
                                       vnₑ_guess, ionization_energies, partition_fns,
                                       log_equilibrium_constants)

    residuals! = setup_chemical_equilibrium_residuals(vtemp, vnₜ, absolute_abundances,
                                                      ionization_energies,
                                                      partition_fns, log_equilibrium_constants, 0)

    tmp = similar(zero) # for storing results of residuals!. jacobian handles this nicely.
    drdx = ForwardDiff.jacobian((tmp, x) -> residuals!(tmp, x), tmp, zero)
    drdp = ForwardDiff.jacobian(tmp, [vtemp, vnₜ]) do tmp, p
        r! = setup_chemical_equilibrium_residuals(p[1], p[2], absolute_abundances,
                                                  ionization_energies, partition_fns,
                                                  log_equilibrium_constants, 0)
        r!(tmp, zero)
    end
    dxdp = -(drdx \ drdp)
    partial_zero = dxdp * partials

    dual_zero = map(zero, eachrow(partial_zero)) do v, p
        ForwardDiff.Dual{T}(v, p...)
    end

    dual_zero
end

# returns the "residuals!" function that is solved.
# This is the "heart" of the chemical/ionization solver.
function setup_chemical_equilibrium_residuals(T, nₜ, absolute_abundances, ionization_energies,
                                              partition_fns, log_equilibrium_constants, logξ)
    molecules = collect(keys(log_equilibrium_constants))

    # precalculate equilibrium coefficients. Here, K is in terms of number density, not partial
    # pressure, unlike those in equilibrium_constants.
    log_nKs = get_log_nK.(molecules, T, Ref(log_equilibrium_constants))

    # H⁻: n(H⁻) = nK_Hminus(T) * n(H I) * nₑ  (formation reaction H I + e⁻ → H⁻).
    # We scale this by mol_log_scale as well.
    log_nK_Hminus = log10(Hminus_nK(T))

    # precompute the ratio of singly and doubly ionized to neutral atoms with factors of nₑ^-1 and
    # nₑ^-2 divided out
    pairs = map(1:MAX_ATOMIC_NUMBER) do Z
        # plug in 1 for nₑ
        saha_ion_weights(T, 1, Z, ionization_energies, partition_fns)
    end
    log_wII_ne, log_wIII_ne2 = log10.(first.(pairs)), log10.(last.(pairs))

    let nₜ = nₜ, log_nKs = log_nKs, molecules = molecules,
        absolute_abundances = absolute_abundances, log_wII_ne = log_wII_ne,
        log_wIII_ne2 = log_wIII_ne2, log_nK_Hminus = log_nK_Hminus
        #`residuals!` puts the residuals the system of molecular equilibrium equations in `F`
        #`x` is a vector containing the number density of the neutral species of each element
        function residuals!(F, x)
            # free params: log10 nₑ and log10 n(X I) for each element
            log_nₑ = x[end]
            nₑ = 10.0^log_nₑ
            log_n_neutral = view(x, 1:MAX_ATOMIC_NUMBER)   # allocation-free

            # write zeros to F once and += it everywhere below
            fill!(F, zero(eltype(F)))

            # initialize n_nuclei to the no-molecules case, adjust as we loop over mols
            n_nuclei = nₜ - nₑ

            # loop over molecules first
            for (mol, log_nK) in zip(molecules, log_nKs)
                atoms = get_atoms(mol.formula)

                # log n_mol = Σ log n(constituent) − log K_n
                # constituent is neutral except for the first atom of a charged diatomic
                log_n_mol = -log_nK + logξ
                if mol.charge == 0 # neutral molecule, possibly polyatomic
                    for Z in atoms
                        log_n_mol += log_n_neutral[Z]
                    end
                else # singly-ionized diatomic, first atom charged
                    Z1, Z2 = atoms
                    # log n(X₁ II) = log n(X1 I) + log wII_ne[Z1] − log nₑ
                    log_n_X1_II = log_n_neutral[Z1] + log_wII_ne[Z1] - log_nₑ
                    log_n_mol += log_n_X1_II + log_n_neutral[Z2]
                end
                n_mol = 10.0^log_n_mol

                for Z in atoms
                    F[Z] += n_mol # RHS molecular contribution
                end

                # "extra" nuclei because this is a molecule
                n_nuclei += (n_atoms(mol) - 1) * n_mol

                # charge conservation
                F[end] += mol.charge * n_mol
            end

            # H⁻: contributes to H nuclei and (negatively) to charge balance.
            #   log n(H⁻) = log nK_Hminus + log n(H I) + log nₑ  + mol_log_scale
            log_n_Hminus = log_nK_Hminus + log_n_neutral[1] + log_nₑ + logξ
            n_Hminus = 10.0^log_n_Hminus
            F[1] += n_Hminus            # counts toward H nuclei conservation
            F[end] -= n_Hminus           # negative charge

            # now loop over atoms, and compute
            #   - atomic number densities  on RHS
            #   - each element LHS (nucleus density)
            #   - atomic ions charge balance RHS
            for Z in 1:MAX_ATOMIC_NUMBER
                log_n_I = log_n_neutral[Z]
                n_I = 10.0^log_n_I
                n_II = 10.0^(log_n_I + log_wII_ne[Z] - log_nₑ)
                n_III = 10.0^(log_n_I + log_wIII_ne2[Z] - 2 * log_nₑ)

                F[Z] += n_I + n_II + n_III - absolute_abundances[Z] * n_nuclei
                F[end] += n_II + 2 * n_III
            end

            # LHS of charge balance
            F[end] -= nₑ

            # scaling
            F[end] /= nₜ
            F[1:MAX_ATOMIC_NUMBER] ./= absolute_abundances .* nₜ
        end
    end
end

"""
    hummer_mihalas_w(T, n_eff, nH, nHe, ne; use_hubeny_generalization=false)

Calculate the correction, w, to the occupation fraction of a hydrogen energy level using the
occupation probability formalism from Hummer and Mihalas 1988, optionally with the generalization by
Hubeny+ 1994.  (Sometimes Daeppen+ 1987 is cited instead, but H&M seems to be where the theory
originated. Presumably it was delayed in publication.)

The expression for w is in equation 4.71 of H&M.  K, the QM correction used in defined in equation 4.24.
Note that H&M's "N"s are numbers (not number densities), and their "V" is volume.  These quantities
apear only in the form N/V, so we use the number densities instead.

This is based partially on Paul Barklem and Kjell Eriksson's
[WCALC fortran routine](https://github.com/barklem/hlinop/blob/master/hbop.f)
(part of HBOP.f), which is used by (at least) Turbospectrum and SME.  As in that routine, we do
consider hydrogen and helium as the relevant neutral species, and assume them to be in the ground
state.  All ions are assumed to have charge 1.  Unlike that routine, the generalization to the
formalism from Hubeny+ 1994 is turned off by default because I haven't closely checked it.  The
difference effects the charged_term only, and temperature is only used when
`use_hubeny_generalization` is set to `true`.
"""
function hummer_mihalas_w(T, n_eff, nH, nHe, ne; use_hubeny_generalization=false)
    # contribution to w from neutral species (neutral H and He, in this implementation)
    # this is sqrt<r^2> assuming l=0.  I'm unclear why this is the approximation barklem uses.
    r_level = sqrt(5 / 2 * n_eff^4 + 1 / 2 * n_eff^2) * bohr_radius_cgs
    # how do I reproduce this helium radius?
    neutral_term = nH * (r_level + sqrt(3) * bohr_radius_cgs)^3 +
                   nHe * (r_level + 1.02bohr_radius_cgs)^3

    # contributions to w from ions (these are assumed to be all singly ionized, so n_ion = n_e)
    # K is a  QM correction defined in H&M '88 equation 4.24
    K = if n_eff > 3
        # WCALC drops the final factor, which is nearly within 1% of unity for all n
        16 / 3 * (n_eff / (n_eff + 1))^2 * ((n_eff + 7 / 6) / (n_eff^2 + n_eff + 1 / 2))
    else
        1.0
    end
    χ = RydbergH_eV / n_eff^2 * eV_to_cgs # binding energy
    e = electron_charge_cgs
    charged_term = if use_hubeny_generalization
        # this is a straight line-by-line port from HBOP. Review and rewrite if used.
        if (ne > 10) && (T > 10)
            A = 0.09 * exp(0.16667 * log(ne)) / sqrt(T)
            X = exp(3.15 * log(1 + A))
            BETAC = 8.3e14 * exp(-0.66667 * log(ne)) * K / n_eff^4
            F = 0.1402 * X * BETAC^3 / (1 + 0.1285 * X * BETAC * sqrt(BETAC))
            log(F / (1 + F)) / (-4π / 3)
        else
            0
        end
    else
        16 * ((e^2) / (χ * sqrt(K)))^3 * ne
    end

    exp(-4π / 3 * (neutral_term + charged_term))
end

# hummer_mihalas_w is based partially on Paul Barklem and Nicolai Piskunov's HBOP routine. The
# familly resemblance is limited to the "use_hubeny_generalization" option, which is not on be
# defult, but we include license for HBOP here.
#
# Copyright (c) 2020, Paul Barklem and Nikolai Piskunov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
    Q_HOLTSMARK

The cumulative Holtsmark microfield distribution, ``Q(\\beta) = P(F < \\beta F_0)``, where ``F_0``
is the Holtsmark normal field strength, tabulated on 150 points uniform in ``\\log_{10}\\beta`` from
0.01 to 50.  Used by [`synthe_mhd_w`](@ref).

This table is copied verbatim from the ATLAS12/SYNTHE `Q_HOLTSMARK` array so that Korg's
`MHD_method=:synthe` occupation probabilities are numerically identical to SYNTHE's.
"""
const Q_HOLTSMARK = [
     1.414671303101461e-07, 1.679306576215498e-07, 1.993444994252022e-07, 2.366346434794731e-07,
     2.809002805410456e-07, 3.334461984807552e-07, 3.958212340900693e-07, 4.698639150426803e-07,
     5.577566360795190e-07, 6.620899645927724e-07, 7.859389687582573e-07, 9.329538149374408e-07,
     1.107467300593689e-06, 1.314622486716282e-06, 1.560524184270007e-06, 1.852418749731333e-06,
     2.198907475766682e-06, 2.610199848759794e-06, 3.098414113872032e-06, 3.677933974566057e-06,
     4.365831897218437e-06, 5.182371440131865e-06, 6.151603336163127e-06, 7.302072795786128e-06,
     8.667658741262937e-06, 1.028856952546623e-05, 1.221252424026525e-05, 1.449615410836965e-05,
     1.720666483126171e-05, 2.042380831343906e-05, 2.424222111026057e-05, 2.877419750061379e-05,
     3.415297755655442e-05, 4.053664530978627e-05, 4.811274949652059e-05, 5.710377986116004e-05,
     6.777365615449882e-05, 8.043541539930427e-05, 9.546031643883215e-05, 1.132886200658068e-04,
     1.344423491071419e-04, 1.595403868046069e-04, 1.893163349213092e-04, 2.246396266107989e-04,
     2.665404747620448e-04, 3.162393359894821e-04, 3.751816855286531e-04, 4.450790309992992e-04,
     5.279572453545271e-04, 6.262134733849464e-04, 7.426830638017916e-04, 8.807182017925553e-04,
     1.044280166083684e-03, 1.238047410111971e-03, 1.467541967671182e-03, 1.739277006102283e-03,
     2.060928688577077e-03, 2.441535851096500e-03, 2.891731333794821e-03, 3.424009106941961e-03,
     4.053031566870220e-03, 4.795981500493777e-03, 5.672963167595142e-03, 6.707456645870922e-03,
     7.926828918285955e-03, 9.362904019250405e-03, 1.105259272456321e-02, 1.303857956034104e-02,
     1.537006106786488e-02, 1.810352400480589e-02, 2.130354516765621e-02, 2.504358545069891e-02,
     2.940673929734126e-02, 3.448638660516894e-02, 4.038667732307838e-02, 4.722275959985061e-02,
     5.512064100258095e-02, 6.421655023415057e-02, 7.465564600958080e-02, 8.658990347532045e-02,
     1.001750012765402e-01, 1.155660400501043e-01, 1.329119530555128e-01, 1.523485300474136e-01,
     1.739900743830450e-01, 1.979198568471881e-01, 2.241797192777529e-01, 2.527594102861463e-01,
     2.835864859104609e-01, 3.165178467055500e-01, 3.513341604156768e-01, 3.877384740815520e-01,
     4.253601858765986e-01, 4.637651729352414e-01, 5.024722401913855e-01, 5.409752086892451e-01,
     5.787690154213567e-01, 6.153773349921458e-01, 6.503786770405923e-01, 6.834278586173648e-01,
     7.142702953716408e-01, 7.427476403869326e-01, 7.687946986070335e-01, 7.924289205182540e-01,
     8.137347922132799e-01, 8.328458692792989e-01, 8.499270198328847e-01, 8.651588016097912e-01,
     8.787250560031417e-01, 8.908040093261350e-01, 9.015625910920573e-01, 9.111533622086631e-01,
     9.197133565854327e-01, 9.273641961484557e-01, 9.342129682962188e-01, 9.403534976718694e-01,
     9.458677646981941e-01, 9.508273225009358e-01, 9.552946289851881e-01, 9.593242514728686e-01,
     9.629639386838167e-01, 9.662555465715050e-01, 9.692358425479507e-01, 9.719372021216177e-01,
     9.743882111568676e-01, 9.766141278943968e-01, 9.786373698849190e-01, 9.804778420731699e-01,
     9.821532811516551e-01, 9.836795130160503e-01, 9.850706740272015e-01, 9.863394492399232e-01,
     9.874972091274948e-01, 9.885541671623754e-01, 9.895195463975206e-01, 9.904015671946884e-01,
     9.912077851485863e-01, 9.919449707318418e-01, 9.926192297837247e-01, 9.932361028219120e-01,
     9.938006248722731e-01, 9.943173688128598e-01, 9.947904964619882e-01, 9.952237635494758e-01,
     9.956206036156809e-01, 9.959840685797559e-01, 9.963172215001466e-01, 9.966224531295311e-01,
     9.969021568157113e-01, 9.971585495409699e-01]
const _log_beta_min_holtsmark = -2.0
const _log_beta_max_holtsmark = log10(50.0)
const _log_beta_step_holtsmark = (_log_beta_max_holtsmark - _log_beta_min_holtsmark) /
                                 (length(Q_HOLTSMARK) - 1)

"""
    holtsmark_Q(β)

The probability that the plasma microfield strength ``F`` is less than ``\\beta F_0``, where ``F_0``
is the Holtsmark normal field strength (``F_0 = 1.25\\times10^{-9} n_e^{2/3}`` in cgs).

Linearly interpolated from [`Q_HOLTSMARK`](@ref) in ``\\log_{10}\\beta``, and clamped to 0 below
``\\beta = 0.01`` and to 1 above ``\\beta = 50``, exactly as ATLAS12/SYNTHE's `holtsmark_Q` does.
(The tabulated value at ``\\beta = 50`` is 0.9972; SYNTHE rounds the remaining 0.3% up to unity so
that unperturbed low-``n`` levels get exactly ``w = 1``.)

Reference: Holtsmark, J. 1919, Ann. Phys. 363, 577.
"""
function holtsmark_Q(β)
    if β <= 0.01
        return zero(β)
    elseif β >= 50.0
        return one(β)
    end

    log_β = log10(β)
    # index (0-based) of the grid cell containing log_β
    idx_f = (log_β - _log_beta_min_holtsmark) / _log_beta_step_holtsmark
    idx = clamp(floor(Int, ForwardDiff.value(idx_f)), 0, length(Q_HOLTSMARK) - 2)
    frac = idx_f - idx
    Q_HOLTSMARK[idx+1] + frac * (Q_HOLTSMARK[idx+2] - Q_HOLTSMARK[idx+1])
end

"""
    synthe_mhd_w(n_eff, ne)

Calculate the correction, w, to the occupation fraction of a hydrogen energy level using the
occupation probability formalism as implemented in ATLAS12/SYNTHE (`occupation_prob`).

Where [`hummer_mihalas_w`](@ref) evaluates the H&M 1988 eq. 4.71 exponential (a nearest-neighbour /
excluded-volume estimate summing a *neutral* perturber term and a charged perturber term), SYNTHE
instead reads the level's survival probability straight off the Holtsmark microfield distribution,

``w_n = Q(\\beta_\\mathrm{crit})``, with ``\\beta_\\mathrm{crit} = C / (n^5 n_e^{2/3})``,

where ``C = 8.798905203085208\\times10^{14}`` and ``Q`` is [`holtsmark_Q`](@ref).
``\\beta_\\mathrm{crit}`` is the critical field (at which the level is Stark-dissolved), in units of
the Holtsmark normal field ``F_0 = 1.25\\times10^{-9} n_e^{2/3}``, so ``w_n`` is just the fraction
of atoms sitting in a field weaker than critical.

Two consequences are worth knowing before you switch formalisms:

 1. **There is no neutral-perturber term and no temperature dependence.** SYNTHE counts only the
    ion microfield, while `hummer_mihalas_w` adds a neutral term ``\\propto n_\\mathrm{H\\,I}``.
    **Which formalism dissolves harder therefore depends on** ``n_\\mathrm{H\\,I}/n_e``, **and the
    sign of the difference reverses across the HRD:**

      + *Cool stars* (solar photosphere: ``n_\\mathrm{H\\,I} \\sim 10^{17}``, ``n_e \\sim 10^{13}``
        cm``^{-3}``, ratio ``\\sim 10^4``) — the neutral term dominates H&M, so Korg dissolves far
        more than SYNTHE (``w_{15} \\approx 0.04`` vs ``\\approx 0.6``).  `:synthe` gives *less*
        bound-free absorption red of the Balmer break, i.e. a sharper break and more flux there.
      + *Hot stars* (A-type, ``T_\\mathrm{eff} \\gtrsim 8000`` K) — hydrogen ionizes, the neutral
        term collapses, and the comparison becomes charged-term-vs-Holtsmark, where SYNTHE is much
        more aggressive.  At ``T_\\mathrm{eff} = 8500`` K, ``\\tau = 1``
        (``n_\\mathrm{H\\,I} = 2.3\\times10^{15}``, ``n_e = 4.8\\times10^{14}``):
        ``w_{12} = 0.61`` for H&M vs ``0.025`` here.  `:synthe` then gives ~4-5x *more* absorption
        red of the break, i.e. a more smeared break and ~30% less flux at 3800 Å.

    This is not just a difference of coefficients: H&M is ``\\exp(-C n^6)``, an exponentially sharp
    cutoff, whereas the Holtsmark CDF approaches 1 only as a power law
    (``1 - Q \\approx 0.997\\,\\beta^{-3/2}``), so SYNTHE always leaves a fatter dissolved tail at
    moderate ``n`` once ``n_e`` is high.
 2. The exponent of `n` is 5, not the 4 of the classical field-ionization threshold.  This is what
    SYNTHE does, and it is reproduced here deliberately: the point of this option is code-to-code
    agreement with SYNTHE, not a re-derivation of the microfield theory.

`ne` is the electron number density in cm``^{-3}`` and `n_eff` is the effective principal quantum
number (which need not be an integer; `H_I_bf` passes fractional values when computing the
dissolved fraction).  As in SYNTHE, `w = 1` is returned for `n_eff <= 1`.

References:

  - Hummer, D.G. & Mihalas, D. 1988, ApJ 331, 794
  - Nayfonov, A., Däppen, W., Hummer, D.G. & Mihalas, D. 1999, ApJ 526, 451
"""
function synthe_mhd_w(n_eff, ne)
    if (n_eff <= 1) || (ne <= 0)
        return one(promote_type(typeof(n_eff), typeof(ne)))
    end
    β_crit = HOLTSMARK_BETA_COEFF / (n_eff^5 * ne^(2 / 3))
    holtsmark_Q(β_crit)
end

"""
    mhd_occupation_w(T, n_eff, nH, nHe, ne; MHD_method=:hummer_mihalas,
                     use_hubeny_generalization=false)

Dispatch to whichever occupation-probability ("MHD") formalism `MHD_method` selects, returning the
correction `w` to the occupation fraction of the hydrogen level with effective principal quantum
number `n_eff`.

  - `:hummer_mihalas` (default): [`hummer_mihalas_w`](@ref), Korg's usual Hummer & Mihalas 1988
    eq. 4.71 treatment.
  - `:synthe`: [`synthe_mhd_w`](@ref), the Holtsmark-microfield treatment used by ATLAS12/SYNTHE.
    `T`, `nH`, `nHe`, and `use_hubeny_generalization` are unused in this case.
  - `:none`: no level dissolution, i.e. `w = 1` always.
"""
function mhd_occupation_w(T, n_eff, nH, nHe, ne; MHD_method=:hummer_mihalas,
                          use_hubeny_generalization=false)
    if MHD_method === :hummer_mihalas
        hummer_mihalas_w(T, n_eff, nH, nHe, ne; use_hubeny_generalization=use_hubeny_generalization)
    elseif MHD_method === :synthe
        synthe_mhd_w(n_eff, ne)
    elseif MHD_method === :none
        one(promote_type(typeof(T), typeof(n_eff), typeof(nH), typeof(nHe), typeof(ne)))
    else
        throw(ArgumentError("Unknown MHD_method: $MHD_method. " *
                            "Must be :hummer_mihalas, :synthe, or :none."))
    end
end

"""
    hummer_mihalas_U_H(T, nH, nHe, ne)

!!!note
This is experimental, and not used by Korg for spectral synthesis.

Calculate the partition function of neutral hydrogen using the occupation probability formalism
from Hummer and Mihalas 1988.  See [`hummer_mihalas_w`](@ref) for details.
"""
function hummer_mihalas_U_H(T, nH, nHe, ne; use_hubeny_generalization=false)
    # These are from NIST, but it would be nice to generate them on the fly.
    hydrogen_energy_levels = [
        0.0,
        10.19880615024,
        10.19881052514816,
        10.19885151459,
        12.0874936591,
        12.0874949611,
        12.0875070783,
        12.0875071004,
        12.0875115582,
        12.74853244632,
        12.74853299663,
        12.7485381084,
        12.74853811674,
        12.74853999753,
        12.748539998,
        12.7485409403,
        13.054498182,
        13.054498464,
        13.054501074,
        13.054501086,
        13.054502042,
        13.054502046336,
        13.054502526,
        13.054502529303,
        13.054502819633,
        13.22070146198,
        13.22070162532,
        13.22070313941,
        13.22070314214,
        13.220703699081,
        13.22070369934,
        13.220703978574,
        13.220703979103,
        13.220704146258,
        13.220704146589,
        13.220704258272,
        13.320916647,
        13.32091675,
        13.320917703,
        13.320917704,
        13.320918056,
        13.38596007869,
        13.38596014765,
        13.38596078636,
        13.38596078751,
        13.385961022639,
        13.4305536,
        13.430553648,
        13.430554096,
        13.430554098,
        13.430554262,
        13.462451058,
        13.462451094,
        13.46245141908,
        13.462451421,
        13.46245154007,
        13.486051554,
        13.486051581,
        13.486051825,
        13.486051827,
        13.486051916,
        13.504001658,
        13.504001678,
        13.50400186581,
        13.504001867,
        13.50400193582
    ]
    hydrogen_energy_level_degeneracies = [
        2,
        2,
        2,
        4,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        8,
        10,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        8,
        10,
        10,
        12,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6,
        2,
        2,
        4,
        4,
        6
    ]
    hydrogen_energy_level_n = [
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        9,
        10,
        10,
        10,
        10,
        10,
        11,
        11,
        11,
        11,
        11,
        12,
        12,
        12,
        12,
        12
    ]

    # for each level calculate the correction, w, and add the term to U
    # the expression for w comes from Hummer and Mihalas 1988 equation 4.71
    U = 0.0
    for (E, g,
    n) in zip(hydrogen_energy_levels, hydrogen_energy_level_degeneracies,
              hydrogen_energy_level_n)
        n_eff = sqrt(RydbergH_eV / (RydbergH_eV - E)) # times Z, which is 1 for hydrogen
        w = hummer_mihalas_w(T, n_eff, nH, nHe, ne;
                             use_hubeny_generalization=use_hubeny_generalization)
        U += w * g * exp(-E / (kboltz_eV * T))
    end
    U
end