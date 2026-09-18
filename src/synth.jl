"""
    synth(kwargs...)

This function creates a synthetic spectrum. It's easier to use than [`synthesize`](@ref), but it
gives you less control. `Korg.synth` also provides shortcuts for some ways
you might want to post-process the spectrum (applying a LSF, rotation, etc.).

# Returns

`synth` **returns a tuple of `(wavelengths, rectified_flux, cntm)`** (Wavelength in Å, rectified
flux as a unitless number between 0 and 1, and continuum in erg/s/cm^4/Å). Note that this is different
from the return type of [`synthesize`](@ref), which is [`SynthesisResult`](@ref).

# Keyword arguments

  - `Teff`: effective temperature in K (no default, must be provided)
  - `logg`: surface gravity in cgs units (no default, must be provided)
  - `M_H`: metallicity, [metals/H], (default: 0.0) (See [`format_A_X`](@ref) for precisely
    how this is interpreted.)
  - `alpha_H`: alpha enhancement, [α/H], (default: `M_H`) (See [`format_A_X`](@ref) for precisely
    how this is interpreted.)
  - _Any atomic symbol_ (e.g. `Fe` or `C`) can be used to specify a (solar relative, [_X_/H])
    abundance. These override `M_H` and `alpha_H`. Specifying an individual abundance means
    that the true metallicity and alpha will not correspond precisely to the values of `M_H`
    and `alpha_H`. See [`format_A_X`](@ref) for details.
  - `linelist`: a linelist (default: [`get_VALD_solar_linelist()`](@ref)). See also
    [`read_linelist`](@ref).
  - `wavelengths`: a tuple of the start and end wavelengths (default: (5000, 6000)), or a vector
    of `(λstart, λstop)` pairs. See
    [Wavelengths](https://ajwheeler.github.io/Korg.jl/stable/Wavelengths/) for all the ways the
    wavelengths can be specified, including how to override the default spacing,
    which may change between versions.
  - `rectify`: whether to rectify (continuum normalize) the spectrum (default: true)
  - `R`: resolution (default: `Inf`, no LSF applied). `R` can be a scalar, or a function from
    wavelength (in Å) to resolving power. See [`apply_LSF`](@ref) for details on how to do this
    manually.
  - `vsini`: projected rotational velocity in km/s (default: 0). This calls [`apply_rotation`](@ref)
    under the hood.
  - `vmic`: microturbulent velocity in km/s (default: 1.0).
  - `nlte` (default: `false`): `true` to apply NLTE departure coefficients to the eligible
    transitions in the linelist (Na I D, Mg I b, Ca I 4226, Ca II H/K and the IR triplet, and 11
    Fe I lines), interpolated from the published grids.  Requires the `.nlte` runtime grid files;
    see [`Korg.nlte_departures`](@ref).  May also be a `NamedTuple` of `nlte_departures` keyword
    arguments, or a [`Korg.NLTE`](@ref).
  - `nlte_grid_dir` (default: `nothing`): where the `.nlte` runtime grid files live, overriding
    `\$KORG_NLTE_DIR`.  Per-element `\$NLTE_GRID_*` variables still win over both.
  - `synthesize_kwargs`: additional keyword arguments passed to [`synthesize`](@ref). Note that if
    `vmic` is specified here, it will override the value passed to `synth`.
  - `format_A_X_kwargs`: additional keyword arguments passed to [`format_A_X`](@ref).
"""
function synth(;
               Teff=nothing,
               logg=nothing,
               M_H=0.0,
               alpha_H=M_H,
               linelist=get_VALD_solar_linelist(),
               wavelengths=(5000, 6000),
               rectify=true,
               R=Inf,
               vsini=0,
               vmic=1.0,
               nlte=false,
               nlte_grid_dir=nothing,
               synthesize_kwargs=Dict(),
               format_A_X_kwargs=Dict(),
               abundances...,)
    if :m_H in keys(abundances)
        throw(ArgumentError("m_H is no longer a supported keyword argument of synth (starting in Korg 1.0). Use M_H instead."))
    end

    # Check for invalid arguments.  Because we catch all in abundances, the default error message is
    # confusing.
    for key in keys(abundances)
        if !(String(key) in Korg.atomic_symbols)
            msg = "$key was passed as a keyword argument to synth, but it is not a valid keyword argument."
            if endswith(String(key), "_H")
                msg *= " To specify an elemental abundance, use the bare atomic symbol (e.g. Ca instead of Ca_H)."
            end
            throw(ArgumentError(msg))
        end
    end

    if isnothing(Teff) || isnothing(logg)
        throw(ArgumentError("Teff and logg must be passed to synth as keyword arguments"))
    end
    A_X = format_A_X(M_H, alpha_H, abundances; format_A_X_kwargs...)
    atm = interpolate_marcs(Teff, logg, A_X)

    wavelengths = Korg.Wavelengths(wavelengths)

    # `nlte=true` is sugar: synth knows Teff/logg (they are departure-coefficient grid axes that
    # synthesize cannot recover from the atmosphere), so it can fill them in.  Pass a NamedTuple to
    # override anything else nlte_departures takes, or a Korg.NLTE to supply coefficients directly.
    nlte_arg = if nlte === false
        nothing
    elseif nlte === true
        (; Teff, logg, vmic)
    elseif nlte isa NamedTuple
        merge((; Teff, logg, vmic), nlte)
    else
        nlte
    end

    # synthesize kwargs currently must be symbols, which is annoying
    spectrum = synthesize(atm, linelist, A_X, wavelengths; vmic, nlte=nlte_arg, nlte_grid_dir,
                          synthesize_kwargs...)
    flux = if rectify
        spectrum.flux ./ spectrum.cntm
    else
        spectrum.flux
    end
    flux = apply_LSF(flux, spectrum.wavelengths, R)
    if vsini > 0
        flux = apply_rotation(flux, wavelengths, vsini)
    end

    spectrum.wavelengths, flux, spectrum.cntm
end
