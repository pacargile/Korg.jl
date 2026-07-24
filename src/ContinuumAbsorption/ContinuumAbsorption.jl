module ContinuumAbsorption
export total_continuum_absorption, continuum_absorption_and_scattering

using ..Korg: ionization_energies, Species, @species_str, _data_dir # not sure that this is the best idea
using ..Korg: Interval, closed_interval, contained, contained_slice, λ_to_ν_bound, hummer_mihalas_w
include("../constants.jl") # I'm not thrilled to duplicate this, but I think it's probably alright

include("bounds_checking.jl") # define helper functions
include("hydrogenic_bf_ff.jl")
include("absorption_H.jl")
include("absorption_He.jl")
include("absorption_ff_positive_ion.jl")
include("absorption_metals_bf.jl")
include("scattering.jl")
include("absorption_mol_photodissociation.jl")
include("absorption_H2_CIA.jl")
include("absorption_He1_detailed.jl")
include("absorption_hotop_bf.jl")

"""
    total_continuum_absorption(νs, T, nₑ, number_densities, partition_funcs; error_oobounds)

The total continuum linear absoprtion coefficient, α, at many frequencies, ν.

# Arguments

  - `νs` are frequencies in Hz
  - `T` is temperature in K
  - `nₑ` is the electron number density in cm^-3
  - `number_densities` is a `Dict` mapping each `Species` to its number density
  - `partition_funcs` is a `Dict` mapping each `Species` to its partition function (e.g.
    `Korg.partition_funcs`)
  - `error_oobounds::Bool` specifies the behavior of most continnum absorption sources when passed
    frequencies or temperature values that are out of bounds for their implementation. When `false`
    (the default), those absorption sources are ignored at those values. Otherwise, an error is
    thrown.

!!! note

    For efficiency reasons, `νs` must be sorted. While this function technically supports any
    sorted `AbstractVector`, it is most effient when passed an  `AbstractRange`.
"""
function total_continuum_absorption(νs, T, nₑ, number_densities::Dict, partition_funcs::Dict;
                                    error_oobounds=false)
    α_abs, α_scat = continuum_absorption_and_scattering(νs, T, nₑ, number_densities,
                                                        partition_funcs;
                                                        error_oobounds=error_oobounds)
    α_abs .+= α_scat # scattering treated as (thermal) absorption, i.e. source function S = B
    α_abs
end

"""
    continuum_absorption_and_scattering(νs, T, nₑ, number_densities, partition_funcs; error_oobounds)

Like [`total_continuum_absorption`](@ref), but returns the tuple `(α_abs, α_scat)` with the
true (thermally-coupled) absorption coefficient and the coherent-scattering coefficient held
separately, rather than summed.  `α_scat` is electron (Thomson) scattering plus Rayleigh scattering
off H I, He I, and H₂.

`total_continuum_absorption` returns `α_abs .+ α_scat`, i.e. it treats scattering as thermal
absorption (source function `S = B`).  Keeping the two apart is required for coherent-scattering
radiative transfer, where scattering enters the source function as `a·J` (with albedo
`a = α_scat / (α_abs + α_scat)`) rather than emitting thermally.  See
[`total_continuum_absorption`](@ref) for the argument descriptions.
"""
function continuum_absorption_and_scattering(νs, T, nₑ, number_densities::Dict,
                                             partition_funcs::Dict; error_oobounds=false)
    α = zeros(promote_type(eltype(νs), typeof(T), typeof(nₑ), valtype(number_densities)),
              length(νs))

    kwargs = Dict(:out_α => α, :error_oobounds => error_oobounds)

    # used more than once
    nH_I = number_densities[species"H_I"]
    invU_H_I = 1 / partition_funcs[species"H I"](log(T))

    # Hydrogen continuum absorption
    # note: inclusion of He I ndens below is NOT a typo
    α .+= H_I_bf(νs, T, nH_I, number_densities[species"He I"], nₑ, invU_H_I)

    Hminus_bf(νs, T, number_densities[species"H-"], nₑ; kwargs...)
    Hminus_ff(νs, T, nH_I * invU_H_I, nₑ; kwargs...)
    H2plus_bf_and_ff(νs, T, nH_I, number_densities[species"H_II"]; kwargs...)

    # He⁻ free-free (John 1994)
    Heminus_ff(νs, T, number_densities[species"He_I"] / partition_funcs[species"He_I"](log(T)), nₑ;
               kwargs...)

    # ff absorption where participating species are positive ions
    # i.e. H I ff is included but not H⁻ ff or He⁻ ff
    # NOTE: this includes He II ff (Z=2), so He I free-free is covered here
    positive_ion_ff_absorption!(α, νs, T, number_densities, nₑ)

    # bf absorption by metals from TOPBase and NORAD
    metal_bf_absorption!(α, νs, T, number_densities)

    # NEW OPACITY SOURCES FROM SYNTHE

    # He I detailed bound-free (HE1OP port from SYNTHE)
    # This adds opacity from 10 resolved He I states + high-n levels +
    # inner-shell ionization + dissolved levels near the series limit.
    # He I free-free is NOT included here (it's in positive_ion_ff_absorption!).
    He1_detailed_bf!(α, νs, T, number_densities, partition_funcs)

    # molecular photodissociation (specifically OH and CH)
    mol_photodissociation_absorption!(α, νs, T, number_densities)

    # H2 collision-induced absorption (CIA)
    H2_CIA_absorption!(α, νs, T, number_densities)

    # HOTOP bound-free absorption from doubly+ ionized metals
    hotop_bf_absorption!(α, νs, T, number_densities, partition_funcs)

    # scattering (kept separate from α; coherent, so it enters RT as a·J, not thermally)
    α_scat = electron_scattering(nₑ) .+ # Thomson scattering (frequency-independent)
             rayleigh(νs, nH_I, number_densities[species"He_I"], number_densities[species"H2"])

    α, α_scat
end

end