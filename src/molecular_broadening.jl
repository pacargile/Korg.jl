# ============================================================================
#  Molecular van der Waals broadening
#
#  Korg historically applied no van der Waals broadening to molecular lines at
#  all: `approximate_gammas` returns 0 for any molecule, so a molecular line
#  whose linelist carries no width got a purely radiative Lorentz profile,
#  ~2900x narrower than the physical one at cool-photosphere pressures.
#
#  This file supplies the missing physics from per-species data: ExoMol's H2 and
#  He broadening coefficients, with the rotational (J) dependence of
#  Gharib-Nezhad et al. (2021).  It is a port of the treatment introduced in
#  ATLAS12/SYNTHE (`data/mol_broad.dat` + `load_mol_broad`, atlas12 commits
#  7312c2f and 697642b), and the data table is the same table.  The two codes
#  evaluate it differently; see "DIFFERENCES FROM ATLAS12/SYNTHE" below.
#
#  ---------------------------------------------------------------------------
#  THE CONVERSION
#
#  ExoMol tabulates, per perturber p, the Lorentz HALF width at half maximum in
#  wavenumber per atmosphere of that perturber at 296 K:
#
#      gamma_p(J, T) = max(g0_p - dgdJ_p * J, 0.1 g0_p) * (296/T)^n_p
#                                                    [cm^-1 per atm of p]
#
#  Korg (like Kurucz) wants the Lorentz FULL width at half maximum in ANGULAR
#  frequency, per unit perturber number density.  Three factors do that:
#
#      * 2         HWHM -> FWHM
#      * 2 pi c    cm^-1 -> rad/s
#      * k T / P_atm   "per atm of p" -> "per cm^-3 of p", since n_p = P/kT
#                      and P_atm = 1.01325e6 dyn/cm^2 (an ATM, not a bar)
#
#  so, summing the perturbers that matter in a cool photosphere,
#
#      Gamma_vdW = 4 pi c k T / P_atm
#                  * (n_H2 gamma_H2 + n_HeI gamma_He + n_HI gamma_H)
#
#  in rad/s, which is what `line_absorption!` adds to `gamma_rad`.
#
#  ---------------------------------------------------------------------------
#  DIFFERENCES FROM ATLAS12/SYNTHE  -- these are real, and Korg is not
#  reproducing SYNTHE's numbers here.  Recorded so that a Korg-vs-SYNTHE
#  molecular band comparison is read correctly.  SYNTHE should eventually be
#  patched to match; until it is, expect systematic differences in molecular
#  band depths, largest where H2O and the electronic bands of TiO/CaH form.
#
#  1. FACTOR OF 2 (SYNTHE is too narrow).  `load_mol_broad` multiplies by
#     2 pi c, not 4 pi c, i.e. it carries ExoMol's HWHM into a slot that the
#     rest of the code reads as a FWHM.  That the Kurucz gamma is an FWHM in
#     angular frequency is fixed by the same code's radiative constant,
#     gammar = 2.223e13 / lambda_nm^2, which is exactly the classical
#     8 pi^2 e^2 / (3 m c lambda^2); and by its Voigt damping parameter,
#     adamp = gamma / (4 pi * Delta_nu_D).  All molecular vdW widths in SYNTHE
#     are therefore a factor of 2 below the ExoMol data they were built from.
#     Korg uses the correct factor.
#
#  2. PERTURBER PARTITION.  SYNTHE's opacity kernel hardwires one effective
#     perturber density, txnxn = (n_HI + 0.42 n_HeI + 0.85 n_H2)(T/1e4)^0.3, so
#     it can only carry ONE number per species: it takes ExoMol's H2 column and
#     divides by 0.85, which makes the H2 term exact and forces He and H onto
#     the atomic Unsoeld weights.  That implies gamma_He/gamma_H2 = 0.494 for
#     EVERY species, against the table's own 0.54-0.60 for the hydrides and
#     oxides, 0.658 for the CO family, and 0.275 for H2O -- where the implied
#     value is 1.8x too high.  Korg has n_HI, n_HeI and n_H2
#     separately, so it uses both ExoMol columns with their own exponents and
#     the implied ratio never enters.
#
#  3. TEMPERATURE DEPENDENCE.  SYNTHE's kernel hardwires a single T^0.3 vdW
#     scaling, so `load_mol_broad` evaluates the true (296/T)^n law once at a
#     reference temperature (3000 K) and accepts a residual (T/3000)^(1-n-0.3)
#     error away from it -- about -4%/+6% over 2500-4000 K for n = 0.5, and much
#     worse for H2O (n = 0.16) or outside that range.  Korg applies (296/T)^n
#     per species per layer, so there is no reference temperature and no
#     residual.
#
#  4. ROTATIONAL AVERAGING.  SYNTHE evaluates gamma_L(J) per line for the 18
#     diatomics whose ASCII line lists carry J, and a single population-weighted
#     <gamma_L(J)> for TiO, H2O and CaOH, whose binary/super-line formats do
#     not.  Korg's `Line` carries no J at all, so it uses the population-
#     weighted average for everything -- but computes it at each layer's own
#     temperature rather than once at 3000 K.  This loses line-to-line variation
#     that SYNTHE has for the ASCII diatomics.  SYNTHE's own A/B test of the
#     converse question (atlas12 3ae33ff, recovering per-line J from E_lower for
#     TiO/H2O) found the variation largely averages out within a dense band, and
#     rejected it.  11 of the 22 species have dgdJ != 0; for the other 11 no
#     J-dependence is known and gamma is constant, so the question is moot.
#
#  5. H I BROADENING.  Nobody publishes H-atom broadening of molecules -- not
#     ExoMol, not Gharib-Nezhad et al.  SYNTHE's kernel implies
#     gamma_H/gamma_H2 = 1/0.85 = 1.176.  Korg scales from the H2 value with the
#     standard van der Waals velocity/polarizability argument, gamma ~
#     alpha^0.4 mu^-0.3, which is the same relation Korg already uses to build
#     its atomic perturber weights and which gives ~1.14 for a heavy molecule.
#     The two agree to a few percent, but neither is constrained by data, and
#     the term is not negligible: on a 2700 K M-dwarf structure H I carries
#     0.4% of the broadening high up but 17% where the H and K bands form and
#     30% at the bottom, as hydrogen dissociates.
#
#  6. SPECIES COVERAGE.  The table holds 22 species against SYNTHE's 20: AlO and
#     TiH are Korg additions, and neither is in SYNTHE's lines.list, so neither
#     affects a comparison run on the synthe2025 master linelist.  A molecular
#     line of any species outside the table (ZrO, YO, ...) keeps whatever `vdW`
#     its linelist gave it, which for most linelists is zero.
# ============================================================================

"""
    MolecularVdWParams

Per-species molecular van der Waals broadening parameters, as read from
`data/molecular_vdW_broadening.dat`.  See that file for units, sources and the
(considerable) uncertainties.

# Fields

  - `γ_H2`, `γ_He`: the Lorentz HWHM at ``J = 0`` in cm⁻¹ per atm of the
    perturber at 296 K
  - `n_H2`, `n_He`: temperature exponents, ``γ(T) = γ (296/T)^n``
  - `dγdJ_H2`, `dγdJ_He`: ``-dγ/dJ_\\mathrm{lower}``, in the same units as `γ`.
    Zero when no J-dependence is known.
  - `B`: the rotational constant in cm⁻¹, used only to population-weight
    ``γ(J)``
  - `src`: the species whose data was actually used (some species take a named
    chemical analogue)
"""
struct MolecularVdWParams{F}
    γ_H2::F
    n_H2::F
    γ_He::F
    n_He::F
    dγdJ_H2::F
    dγdJ_He::F
    B::F
    src::String
end

"""
    read_molecular_vdW_params([filename])

Parse `data/molecular_vdW_broadening.dat` into a `Dict` mapping
[`Korg.Species`](@ref) to [`Korg.MolecularVdWParams`](@ref).
"""
function read_molecular_vdW_params(fname=joinpath(_data_dir, "molecular_vdW_broadening.dat"))
    d = Dict{Species,MolecularVdWParams{Float64}}()
    for line in eachline(fname)
        line = strip(first(split(line, "#")))
        isempty(line) && continue
        toks = split(line)
        length(toks) == 9 ||
            throw(ArgumentError("Expected 9 columns in $fname, got $(length(toks)): $line"))
        d[Species(toks[1])] = MolecularVdWParams(parse.(Float64, toks[2:8])..., String(toks[9]))
    end
    d
end

"""
Molecular van der Waals broadening parameters, keyed by species.  Molecular lines of these species
have their `vdW` width computed by Korg from this table, overriding whatever the linelist supplied.
Molecules absent from it keep their linelist `vdW`.  See `data/molecular_vdW_broadening.dat`.
"""
const molecular_vdW_params = read_molecular_vdW_params()

"""
The fraction of its ``J = 0`` value at which ``γ_L(J)`` is floored, from
Gharib-Nezhad et al. (2021) Eq. 5.  It exists to stop the linear decline in J going negative.
"""
const molecular_vdW_J_floor = 0.1

"""
The highest J included when population-averaging ``γ_L(J)``.  Generous: the most slowly converging
species in the table (CaOH, ``B`` = 0.334 cm⁻¹) peaks near ``J`` = 100 at 4000 K.
"""
const molecular_vdW_J_max = 300

# polarizabilities in atomic units, from https://doi.org/10.1080/00268976.2018.1535143.
# (`line_absorption!` builds its atomic perturber weights from the same numbers.)
const polarizability_H_au = 4.50711
const polarizability_He_au = 1.38375
const polarizability_H2_au = 5.503 # ± 0.049 a.u. (in text, not Table 1)

# hc/k, the second radiation constant, in cm K.  Converts B J(J+1) [cm^-1] to a temperature.
const _c₂_cm_K = hplanck_cgs * c_cgs / kboltz_cgs
const _one_atm_cgs = 1.01325e6 # dyn/cm^2

"""
    rotationally_averaged_γ(γ₀, dγdJ, B, T)

The Boltzmann-weighted mean of ``γ_L(J) = \\max(γ_0 - (dγ/dJ) J, 0.1 γ_0)`` over the rotational
ladder of a rigid rotor with rotational constant `B` (cm⁻¹) at temperature `T`, i.e. weighted by
``(2J+1) \\exp[-B J (J+1) hc / kT]``.

This exists because [`Korg.Line`](@ref) carries no rotational quantum number, so a per-line
``γ_L(J)`` is not available.  A single ``J = 0`` value would be badly wrong for a heavy molecule:
TiO has ``B`` = 0.535 cm⁻¹, so at 3000 K its lines are populated near ``J`` = 44, far enough down
the decline to be sitting on the floor.

Returns `γ₀` unchanged when the species has no known J-dependence.
"""
function rotationally_averaged_γ(γ₀, dγdJ, B, T)
    if dγdJ <= 0 || B <= 0
        return γ₀ * one(T)
    end
    γ_floor = molecular_vdW_J_floor * γ₀
    weight_sum = zero(T)
    γ_sum = zero(T)
    for J in 0:molecular_vdW_J_max
        w = (2J + 1) * exp(-B * J * (J + 1) * _c₂_cm_K / T)
        weight_sum += w
        γ_sum += w * max(γ₀ - dγdJ * J, γ_floor)
    end
    γ_sum / weight_sum
end

"""
    molecular_vdW_Γ(species, params, temps, n_densities)

The van der Waals broadening ``Γ`` of every line of a molecular `species`, as a vector over
atmospheric layers.  This is the Lorentz FWHM in angular frequency (rad/s), the same quantity as
`Line.gamma_rad`, ready to be summed into the total damping.

Unlike the atomic case there is no line-dependence: `Γ` is a property of the species and the layer,
so `line_absorption!` computes it once per species rather than once per line.

`params` is the species' [`Korg.MolecularVdWParams`](@ref); `n_densities` maps species to number
density, as elsewhere in Korg.

H₂ and He use ExoMol's own coefficients.  H I is unconstrained by any published data, so it is
scaled from the H₂ value with the usual van der Waals argument ``γ ∝ α^{0.4} μ^{-0.3}`` (from
``C_6^{2/5} \\bar{v}^{3/5}``) using the molecule–perturber reduced masses, and inherits the H₂
temperature exponent.
"""
function molecular_vdW_Γ(species, params::MolecularVdWParams, temps, n_densities)
    m_mol = get_mass(species)
    m_H = atomic_masses[1]
    m_H2 = 2 * m_H
    # reduced masses of the molecule with each perturber
    μ_H = m_mol * m_H / (m_mol + m_H)
    μ_H2 = m_mol * m_H2 / (m_mol + m_H2)
    γ_H_over_γ_H2 = (polarizability_H_au / polarizability_H2_au)^0.4 * (μ_H / μ_H2)^-0.3

    # ⟨γ_L(J)⟩ at each layer's own temperature, then the (296/T)^n scaling
    γ_H2 = @. rotationally_averaged_γ(params.γ_H2, params.dγdJ_H2, params.B, temps) *
              (296 / temps)^params.n_H2
    γ_He = @. rotationally_averaged_γ(params.γ_He, params.dγdJ_He, params.B, temps) *
              (296 / temps)^params.n_He

    # 4πc  : HWHM in cm^-1 -> FWHM in rad/s
    # kT/P : per atm of perturber -> per perturber cm^-3
    @. 4π * c_cgs * kboltz_cgs * temps / _one_atm_cgs *
       (n_densities[species"H2"] * γ_H2 +
        n_densities[species"He_I"] * γ_He +
        n_densities[species"H_I"] * γ_H_over_γ_H2 * γ_H2)
end

"""
    molecular_vdW_Γs(linelist, temps, n_densities)

Precompute [`Korg.molecular_vdW_Γ`](@ref) for every species in `linelist` that has an entry in
[`Korg.molecular_vdW_params`](@ref), returning a `Dict` from species to a vector over layers.
Species absent from the returned `Dict` are broadened with their linelist `vdW` in the usual way.
"""
function molecular_vdW_Γs(linelist, temps, n_densities)
    # spelled out rather than written as a comprehension so that the Dict is concretely typed even
    # when the linelist contains no molecules with broadening data (the usual case for atomic work).
    F = promote_type(eltype(temps), eltype(n_densities[species"H2"]))
    Γs = Dict{Species,Vector{F}}()
    for spec in unique(l.species for l in linelist)
        if haskey(molecular_vdW_params, spec)
            Γs[spec] = molecular_vdW_Γ(spec, molecular_vdW_params[spec], temps, n_densities)
        end
    end
    Γs
end
