# NLTE departure coefficients for selected lines.
#
# WHAT THIS IS FOR
# ----------------
# Korg is an LTE synthesizer: line opacity comes from Boltzmann/Saha populations and the transfer
# step is handed S = B_ν.  That is wrong for resonance lines of minority species, most sharply the
# alkalis (Olander et al. 2021 measure −0.06 to −0.37 dex on the K I resonance lines in M dwarfs).
# This file is a port of the treatment in the modern ATLAS12/SYNTHE port
# (`src/mod_nlte.f90` there), reading the *same* runtime grid files so that the two codes can be
# compared line by line.
#
# THE PHYSICS, AND WHY TWO NUMBERS PER LINE PER LAYER ARE ENOUGH
# --------------------------------------------------------------
# With departure coefficients b_l = n_l/n_l(LTE), b_u = n_u/n_u(LTE), and x = hν₀/kT at line
# centre,
#
#     κ   = κ_LTE ⋅ b_l [1 − (b_u/b_l) e^−x] / [1 − e^−x]                            (1)
#     S_l = (2hν³/c²) / [(b_l/b_u) e^x − 1]                                          (2)
#
# Both stimulated-emission corrections are exact, not Wien-limit.  (1) and (2) multiply to
#
#     κ S_l = b_u ⋅ [κ_LTE (1 − e^−x)] ⋅ B_ν                                         (3)
#
# — the line *emissivity* is simply b_u times its LTE value.  That identity is what makes this
# cheap to retrofit.  Korg accumulates one summed absorption coefficient α per (layer, λ); the
# opacity-weighted source function the transfer wants is
#
#     S = Σ_i κ_i S_i / Σ_i κ_i = B_ν ⋅ [α + Σ_i κ_i (r_i − 1)] / α,     r_i = S_i/B_ν
#
# and the second sum runs only over lines that are NOT in LTE, because r_i − 1 vanishes for every
# other line — continuum included.  So the whole retrofit is ONE extra accumulator carrying the
# deviation of the emissivity from LTE, and the millions of LTE lines never touch it.
#
# Two conveniences relative to SYNTHE, both structural:
#
#   * Korg's `levels_factor` in `line_absorption.jl` is exactly exp(−βE_l) − exp(−βE_u), i.e. the
#     denominator of (1).  The NLTE opacity is obtained by writing it as
#     b_l exp(−βE_l) − b_u exp(−βE_u), which is (1) exactly, with no separate x to form.
#   * `RadiativeTransfer.radiative_transfer` already takes S as a full layers × λ matrix, so
#     nothing in the transfer needs to change.  SYNTHE had to thread a separate SLINE through JOSH.
#
# WHERE THE COEFFICIENTS COME FROM
# --------------------------------
# `read_nlte_grid` reads the per-element runtime files built by the SYNTHE port's
# `tools/nlte_extract_grid.py` → `nlte_build_index.py` → `nlte_build_runtime.py` chain, from the
# Amarsi et al. (2020) MARCS grids as repackaged by Gerber et al. (2023).  `nlte_departures`
# interpolates them over Teff, log g, [Fe/H], v_turb and A(X) onto this atmosphere's own τ₅₀₀₀
# layers.  Departure coefficients can also be supplied directly (see `NLTE`), which is how the
# opacity/source-function core is tested without an 800 MB file.

# Small formatting helpers for the diagnostics.  Korg does not depend on Printf.  These must not
# assume Float64: under ForwardDiff [Fe/H] and A(X) are `Dual`s, which are `Real` but not
# `AbstractFloat`, so `round(x; digits)` is not defined for them.
_fmt(x, digits) = x isa AbstractFloat ? string(round(x; digits=digits)) : string(x)
_sfmt(x, digits) = (x >= 0 ? "+" : "") * _fmt(x, digits)

"""
    NLTETransition(species, E_lower, E_upper, level_lower, level_upper, element, label)

One transition eligible for an NLTE treatment.

Eligible transitions are matched to lines on species plus BOTH level energies, never on
wavelength, so that every hyperfine and isotopic component of a multiplet is tagged with the one
pair of departure coefficients belonging to its pair of levels.

# Fields

  - `species`: the [`Species`](@ref), e.g. `species"Ca II"`.
  - `E_lower`, `E_upper`: level energies in cm⁻¹, on the linelist's scale (each ionization stage
    zeroed separately, as in `gfall`).
  - `level_lower`, `level_upper`: level indices in the MODEL ATOM the departure-coefficient grid
    was solved with.  These are a property of the grid, not of the linelist, and must be rechecked
    if a grid is reissued.
  - `element`: index into [`Korg.NLTE_ELEMENTS`](@ref).
  - `label`: human-readable name, used in diagnostics only.
"""
struct NLTETransition
    species::Species
    E_lower::Float64      # cm^-1
    E_upper::Float64      # cm^-1
    level_lower::Int
    level_upper::Int
    element::Int
    label::String
end

"""
    NLTE_ELEMENTS

The elements for which a departure-coefficient grid exists, in the order the grid array is indexed
by `NLTETransition.element`.  One published grid per element, so an element is also a runtime file
and an abundance axis.  Each entry gives the symbol, atomic number, default filename, and the
environment variable that overrides the path.
"""
const NLTE_ELEMENTS = [(sym="Na", Z=11, file="Na_MARCS_Jul-14-2023.nlte", env="NLTE_GRID_NA"),
    (sym="Mg", Z=12, file="Mg_MARCS_Nov-13-2024.nlte", env="NLTE_GRID_MG"),
    (sym="Ca", Z=20, file="Ca_MARCS_Jun-02-2021.nlte", env="NLTE_GRID_CA"),
    (sym="Fe", Z=26, file="Fe_MARCS_May-07-2021.nlte", env="NLTE_GRID_FE")]

# Solar reference for turning A(Fe) into [Fe/H].  The MARCS grids' own [Fe/H] axis is on a scale
# comparable to GS98, and this is the value the SYNTHE port uses, so the two codes land in the same
# grid cell.  Different solar scales differ by a few 0.01 dex, which shifts the metallicity lookup
# slightly; A(X) itself is absolute and unaffected.
const NLTE_A_FE_SUN = 7.50

"""
    NLTE_TRANSITIONS

The 22 transitions carried by the SYNTHE port, ported verbatim from its `mod_mklinelist`
transition table.  Na I D1/D2, Mg I b1/b2/b4, Ca I 4226, Ca II H and K, the Ca II infrared triplet,
and 11 Fe I lines.

Two traps recorded in the source of the Ca grid and preserved here: its model atom runs both
ionization stages in ONE level list, so `E_lower = 0` for both Ca I 4226 and Ca II H&K while the
level indices differ; and the weak triplet members transpose easily (8498 → 4p ²P₃/₂,
8662 → 4p ²P₁/₂) — check against 1e8/(E_upper − E_lower) rather than against memory before editing.

Fe II is deliberately absent: the grid's atom merges it into 58 SUPERLEVELS at term-averaged
energies, so no individual Fe II line is addressable by level energy at all.  Every Fe II line
tried fails to match, which is the correct outcome and must not be "fixed" by loosening
`energy_tolerance`.  Fe II is the dominant stage in these stars and near LTE, so little is lost.
"""
const NLTE_TRANSITIONS = [
    NLTETransition(species"Na I", 0.000, 16973.366, 1, 3, 1, "Na I D2 5891.58"),
    NLTETransition(species"Na I", 0.000, 16956.170, 1, 2, 1, "Na I D1 5897.56"),
    NLTETransition(species"Mg I", 21850.400, 41197.403, 2, 6, 2, "Mg I b4 5168.76"),
    NLTETransition(species"Mg I", 21870.460, 41197.403, 3, 6, 2, "Mg I b2 5174.13"),
    NLTETransition(species"Mg I", 21911.170, 41197.403, 4, 6, 2, "Mg I b1 5185.05"),
    NLTETransition(species"Ca I", 0.000, 23652.300, 1, 9, 3, "Ca I 4227.92"),
    NLTETransition(species"Ca II", 0.000, 25414.400, 68, 72, 3, "Ca II K 3934.77"),
    NLTETransition(species"Ca II", 0.000, 25191.510, 68, 71, 3, "Ca II H 3969.59"),
    NLTETransition(species"Ca II", 13650.190, 25414.400, 69, 72, 3, "Ca II 8500.36"),
    NLTETransition(species"Ca II", 13710.880, 25414.400, 70, 72, 3, "Ca II 8544.44"),
    NLTETransition(species"Ca II", 13650.190, 25191.510, 69, 71, 3, "Ca II 8664.52"),
    NLTETransition(species"Fe I", 704.007, 26140.170, 3, 56, 4, "Fe I 3931.4"),
    NLTETransition(species"Fe I", 888.132, 26339.690, 4, 58, 4, "Fe I 3929.0"),
    NLTETransition(species"Fe I", 415.933, 25899.980, 2, 54, 4, "Fe I 3924.0"),
    NLTETransition(species"Fe I", 978.074, 26479.380, 5, 61, 4, "Fe I 3921.4"),
    NLTETransition(species"Fe I", 415.933, 19757.030, 2, 23, 4, "Fe I 5170.3"),
    NLTETransition(species"Fe I", 0.000, 19350.890, 1, 18, 4, "Fe I 5167.7"),
    NLTETransition(species"Fe I", 32873.632, 52213.230, 87, 361, 4, "Fe I 5170.7"),
    NLTETransition(species"Fe I", 29371.812, 48702.530, 77, 252, 4, "Fe I 5173.1"),
    NLTETransition(species"Fe I", 24338.767, 36079.370, 49, 111, 4, "Fe I 8517.4"),
    NLTETransition(species"Fe I", 24118.819, 35767.560, 46, 109, 4, "Fe I 8584.6"),
    NLTETransition(species"Fe I", 23783.619, 35379.200, 45, 107, 4, "Fe I 8624.0")]

"""
    NLTE(transitions, b_lower, b_upper)

Departure coefficients for a set of transitions, evaluated on an atmosphere's layers.  This is
what [`synthesize`](@ref) consumes; it does not care where the coefficients came from.

# Fields

  - `transitions`: a vector of [`NLTETransition`](@ref)s.
  - `b_lower`, `b_upper`: matrices of size (transitions × layers) holding b_l and b_u.  A value of
    1 is exactly LTE — the opacity scale and the emissivity deviation both reduce to their LTE
    values identically, not approximately.

Construct one from the published grids with [`nlte_departures`](@ref), or directly from
coefficients you have computed yourself.
"""
struct NLTE{M<:AbstractMatrix}
    transitions::Vector{NLTETransition}
    b_lower::M
    b_upper::M

    function NLTE(transitions, b_lower::M, b_upper::M) where {M<:AbstractMatrix}
        if size(b_lower) != size(b_upper)
            throw(ArgumentError("b_lower and b_upper must have the same size"))
        end
        if size(b_lower, 1) != length(transitions)
            throw(ArgumentError("b_lower has $(size(b_lower,1)) rows but there are " *
                                "$(length(transitions)) transitions"))
        end
        new{M}(collect(transitions), b_lower, b_upper)
    end
end

Base.length(nlte::NLTE) = length(nlte.transitions)

"""
    NLTEGrid

A departure-coefficient runtime grid for one element, as written by the SYNTHE port's
`tools/nlte_build_runtime.py`.  One file per element: the axes, then the parameter → record index,
then the records themselves, so an index can never be paired with the wrong data.

Only the header and index are held in memory (a few MB); records are read from disk on demand, and
an interpolation touches at most 32 of them (~79 kB), so the file's size costs disk and not run
time.
"""
struct NLTEGrid
    path::String
    axes::NTuple{5,Vector{Float64}}   # Teff, log g, [Fe/H], v_turb, A(X) − [Fe/H]
    level_ids::Vector{Int}            # original model-atom level number of each slot
    rec::Array{Int32,5}               # 1-based record number, 0 = absent
    geo::Array{Int8,5}                # 0 absent, 1 plane-parallel, 2 spherical
    ndep::Int
    nlev::Int
    nrec::Int
    recw::Int                         # floats per record
    data0::Int                        # 0-based byte offset of the data block
end

function Base.show(io::IO, g::NLTEGrid)
    n = count(>(0), g.rec)
    print(io, "NLTEGrid(", basename(g.path), ": ", join(length.(g.axes), "×"), " axes, ",
          g.nlev, " levels, ", g.ndep, " depths, ", g.nrec, " records, ",
          round(Int, 100n / length(g.rec)), "% of cells filled)")
end

"""
    read_nlte_grid(path)

Read the header and parameter index of an `.nlte` runtime grid file, returning an
[`Korg.NLTEGrid`](@ref).  The record data itself is left on disk.

The file's header fixes its size exactly, and that is checked here.  A *missing* grid announces
itself, but a *truncated* one — an interrupted copy of an 800 MB file — passes an existence test
and then reads garbage or fails deep inside the interpolation, so it is caught at load, where the
message can still be useful.
"""
function read_nlte_grid(path)
    isfile(path) || throw(ArgumentError("NLTE grid file not found: $path"))
    open(path, "r") do io
        hdr = Vector{Int32}(undef, 8)
        read!(io, hdr)
        nT, nG, nF, nV, nD, nlev, ndep, nrec = Int.(hdr)
        if any(<=(0), (nT, nG, nF, nV, nD, nlev, ndep, nrec))
            throw(ArgumentError("$path: nonsensical header $(Int.(hdr)); not an .nlte file?"))
        end

        axes = map((nT, nG, nF, nV, nD)) do n
            a = Vector{Float32}(undef, n)
            read!(io, a)
            Float64.(a)
        end

        level_ids = Vector{Int32}(undef, nlev)
        read!(io, level_ids)

        ncell = nT * nG * nF * nV * nD
        recflat = Vector{Int32}(undef, ncell)
        read!(io, recflat)
        geoflat = Vector{Int8}(undef, ncell)
        read!(io, geoflat)

        # Julia and Fortran are both column-major, so the index arrays reshape directly.
        rec = reshape(recflat, nT, nG, nF, nV, nD)
        geo = reshape(geoflat, nT, nG, nF, nV, nD)

        data0 = position(io)
        recw = ndep + nlev * ndep
        want = data0 + nrec * recw * 4
        actual = filesize(path)
        if want != actual
            throw(ArgumentError("$path is not the size its own header implies: header implies " *
                                "$want bytes, file is $actual bytes.  Most likely an interrupted " *
                                "copy or a partial download; re-install it."))
        end

        NLTEGrid(path, axes, Int.(level_ids), rec, geo, ndep, nlev, nrec, recw, data0)
    end
end

"""
    nlte_grid_path(element_index; grid_dir=nothing)

Resolve the path to one element's runtime grid.  The element's own environment variable
(`\$NLTE_GRID_NA` and friends, the same ones the SYNTHE port honours) wins; otherwise the file is
looked for in `grid_dir`, which defaults to `\$KORG_NLTE_DIR`.
"""
function nlte_grid_path(e::Int; grid_dir=nothing)
    el = NLTE_ELEMENTS[e]
    override = get(ENV, el.env, "")
    isempty(override) || return override
    dir = something(grid_dir, get(ENV, "KORG_NLTE_DIR", ""))
    if isempty(dir)
        throw(ArgumentError("Don't know where to find the $(el.sym) NLTE grid ($(el.file)). " *
                            "Pass `grid_dir`, or set \$KORG_NLTE_DIR, or point \$$(el.env) " *
                            "at the file itself."))
    end
    joinpath(dir, el.file)
end

"""
    nlte_tags(linelist, transitions; energy_tolerance=5.0)

Return a vector of the same length as `linelist` giving, for each line, the index of the
[`NLTETransition`](@ref) it belongs to, or 0 if it is an ordinary LTE line.

Lines are matched on species plus BOTH level energies rather than on wavelength, so that all of a
multiplet's hyperfine and isotopic components — Ca II K alone has 11 in `gfall` — are tagged with
the one pair of departure coefficients belonging to their pair of levels.

`energy_tolerance` is in cm⁻¹ and applies to both levels.  The default of 5 cm⁻¹ is loose enough to
absorb level-energy revisions between linelist releases and tight enough that no other transition
of the same species can collide: the closest pair in [`Korg.NLTE_TRANSITIONS`](@ref) is Mg b4/b2,
whose lower levels are 20 cm⁻¹ apart.  Do not loosen it until something matches.
"""
function nlte_tags(linelist, transitions; energy_tolerance=5.0)
    tags = zeros(Int, length(linelist))
    isempty(transitions) && return tags

    # group by species so that the overwhelming majority of lines cost one dictionary lookup
    by_species = Dict{Species,Vector{Int}}()
    for (k, tr) in enumerate(transitions)
        push!(get!(by_species, tr.species, Int[]), k)
    end

    # eV → cm^-1
    per_cm = 1 / (hplanck_eV * c_cgs)

    for (i, line) in enumerate(linelist)
        candidates = get(by_species, line.species, nothing)
        isnothing(candidates) && continue
        E_lo = line.E_lower * per_cm
        E_up = (line.E_lower + c_cgs * hplanck_eV / line.wl) * per_cm
        for k in candidates
            tr = transitions[k]
            if abs(E_lo - tr.E_lower) < energy_tolerance &&
               abs(E_up - tr.E_upper) < energy_tolerance
                tags[i] = k
                break
            end
        end
    end
    tags
end

"""
    _bracket(axis, x)

Locate `x` on a sorted axis, returning the lower node index and the weight of the node above it.
Both are CLAMPED to the axis ends: outside the grid the edge value is held, never extrapolated.
"""
function _bracket(axis::Vector{Float64}, x)
    n = length(axis)
    n == 1 && return 1, zero(x)
    x <= axis[1] && return 1, zero(x)
    x >= axis[n] && return n - 1, one(x)
    lo = searchsortedlast(axis, x)
    lo = min(max(lo, 1), n - 1)
    lo, (x - axis[lo]) / (axis[lo+1] - axis[lo])
end

# How far off the end of an axis x fell, or 0 if it is inside.  A collapsed axis constrains
# nothing, so it never counts as clamped.
function _clamp_amount(axis::Vector{Float64}, x)
    length(axis) == 1 && return zero(x)
    x < axis[1] && return x - axis[1]
    x > axis[end] && return x - axis[end]
    zero(x)
end

"""
    _interpolate_element(grid, transitions, tau_5000, Teff, logg, M_H, vmic, A_X_element; kwargs...)

Multilinear interpolation of one element's departure coefficients over the five grid axes,
evaluated on this atmosphere's own layers.  Returns `(b_lower, b_upper, status)` where the two
matrices are (transitions × layers), or `nothing` if this element must revert to LTE.

TWO THINGS MAKE THIS MORE THAN A WEIGHTED SUM.

Every corner record carries its OWN τ₅₀₀₀ grid — they are different MARCS atmospheres — so b cannot
be combined index by index.  Each corner is first put onto THIS atmosphere's τ₅₀₀₀ and only then
combined.  Interpolation is in log b against log τ, with the corner's endpoint value held outside
its range: b turns over sharply at the top of the atmosphere, and a linear continuation there is
confident nonsense.

And corners can be MISSING: the grid is ~41% filled, because it is HR-diagram shaped.  Weight from
absent corners is dropped and the rest renormalised, which is a real approximation near the edges
of the populated region — so the surviving weight fraction is reported rather than silently
absorbed.
"""
function _interpolate_element(grid::NLTEGrid, transitions, tau_5000, Teff, logg, M_H, vmic, A_el;
                              held_to_LTE=false, verbose=true, sym="?")
    nlayers = length(tau_5000)
    ntrans = length(transitions)

    # Resolve each transition's model-atom levels to slots in this file's records.  The transition
    # table names ATOM levels; which of them a given runtime file carries is a property of the file,
    # so a file built with a narrower level set must fail loudly rather than read the wrong slot.
    slot_lo = Vector{Int}(undef, ntrans)
    slot_up = Vector{Int}(undef, ntrans)
    for (k, tr) in enumerate(transitions)
        lo = findfirst(==(tr.level_lower), grid.level_ids)
        up = findfirst(==(tr.level_upper), grid.level_ids)
        if isnothing(lo) || isnothing(up)
            throw(ArgumentError("$(tr.label) needs model-atom levels $(tr.level_lower) and " *
                                "$(tr.level_upper), but $(basename(grid.path)) does not carry " *
                                "both.  Rebuild it with a wider --levels set."))
        end
        slot_lo[k], slot_up[k] = lo, up
    end

    aT, aG, aF, aV, aD = grid.axes
    dX = A_el - M_H    # the grid's abundance axis is A(X) relative to the metallicity

    # Hotter than the grid?  That one axis is not clampable.  The grids stop at 8000 K, and a hotter
    # model would otherwise be handed the 8000 K departure coefficients silently — badly wrong,
    # since Teff is the axis along which these species' departures grow fastest and an A star is
    # nothing like an 8000 K one.  This case, and only this case, reverts to LTE.
    if Teff > aT[end]
        verbose &&
            @warn "NLTE: $sym is hotter than the grid's last Teff node ($(aT[end]) K); this " *
                  "element reverts to LTE rather than clamp (departures grow fastest along Teff)."
        return nothing
    end

    # Every other axis clamps, which is the right trade: log g 5.6 against an axis ending at 5.5,
    # or an abundance just off the end of the ladder, is a small extrapolation of a slowly varying
    # quantity, whereas refusing it would punch holes in a production grid for no physical reason.
    # Clamping is always REPORTED, per axis and with the amount, so it can be audited.
    if verbose
        notes = String[]
        for (ax, x, nm) in zip((aT, aG, aF, aV, aD), (Teff, logg, M_H, vmic, dX),
                               ("Teff", "logg", "[Fe/H]", "vmic", "dX"))
            d = _clamp_amount(ax, x)
            iszero(d) || push!(notes, nm * _sfmt(d, 3))
        end
        isempty(notes) ||
            @warn "NLTE: $sym is outside the grid, held at the edge for " * join(notes, " ")
    end

    iT, wT = _bracket(aT, Teff)
    iG, wG = _bracket(aG, logg)
    iF, wF = _bracket(aF, M_H)
    iV, wV = _bracket(aV, vmic)
    iD, wD = _bracket(aD, dX)

    wtype = promote_type(typeof(wT), typeof(wG), typeof(wF), typeof(wV), typeof(wD))
    acctype = promote_type(wtype, eltype(tau_5000), Float64)

    # --- PHASE 1: collect the populated corners and their weights ---
    crec = Int[]
    cwgt = acctype[]
    cgeo = Int[]
    dxoff = zero(acctype)
    for cT in 0:min(1, length(aT)-1), cG in 0:min(1, length(aG)-1),
        cF in 0:min(1, length(aF)-1), cV in 0:min(1, length(aV)-1),
        cD in 0:min(1, length(aD)-1)
        w = (cT == 1 ? wT : 1 - wT) * (cG == 1 ? wG : 1 - wG) * (cF == 1 ? wF : 1 - wF) *
            (cV == 1 ? wV : 1 - wV) * (cD == 1 ? wD : 1 - wD)
        w <= 0 && continue
        r = grid.rec[iT+cT, iG+cG, iF+cF, iV+cV, iD+cD]
        r <= 0 && continue                       # absent corner
        push!(crec, Int(r))
        push!(cwgt, w)
        push!(cgeo, Int(grid.geo[iT+cT, iG+cG, iF+cF, iV+cV, iD+cD]))
    end

    # --- PHASE 2: nothing populated?  step along the abundance axis ---
    # The published abundance ladders have holes: at 4500 K / 1.5 / [Fe/H] = −2 the Mg ladder covers
    # only [Mg/Fe] = −1.05..−0.45, so a scaled-solar model finds nothing.  The fallback is the
    # nearest POPULATED point on the abundance axis alone, because b depends far more weakly on
    # A(X) than on Teff or log g.
    if isempty(crec)
        # `wX > 0.5 ? 1 : 0` rather than `round(Int, wX)`: the weights can be ForwardDiff `Dual`s
        # (they depend on A(X)), for which `round(Int, _)` is not defined, while comparison is.
        jT = min(iT + (wT > 0.5), length(aT))
        jG = min(iG + (wG > 0.5), length(aG))
        jF = min(iF + (wF > 0.5), length(aF))
        jV = min(iV + (wV > 0.5), length(aV))
        found = false
        # NB: `for a, b` is a single loop nest in Julia, so `break` leaves both loops -- which is
        # what we want once a cell is found, but means the "+0 and -0 are the same cell" case must
        # be skipped with `continue` rather than broken out of.
        for off in 0:(length(aD)-1), sgn in (1, -1)
            (off == 0 && sgn == -1) && continue
            jD = iD + sgn * off
            (jD < 1 || jD > length(aD)) && continue
            if grid.rec[jT, jG, jF, jV, jD] > 0
                push!(crec, Int(grid.rec[jT, jG, jF, jV, jD]))
                push!(cwgt, one(acctype))
                push!(cgeo, Int(grid.geo[jT, jG, jF, jV, jD]))
                dxoff = aD[jD] - dX
                found = true
                break
            end
        end
        if !found
            verbose && @warn "NLTE: $sym has no populated grid cell anywhere on the abundance " *
                  "axis here; reverts to LTE."
            return nothing
        end
        verbose && @warn "NLTE: $sym — the grid has no cell at this abundance; using the " *
              "nearest populated one, offset $(_sfmt(dxoff, 2)) dex."
    end

    # --- PHASE 3: read the corners and put them on our depth scale ---
    nd = grid.ndep
    acc_lo = zeros(acctype, ntrans, nlayers)
    acc_up = zeros(acctype, ntrans, nlayers)
    log_tau_model = [log10(max(t, 1e-99)) for t in tau_5000]
    clamped = falses(nlayers)
    wsum = zero(acctype)
    nsph = 0

    buf = Vector{Float32}(undef, grid.recw)
    ltau = Vector{Float64}(undef, nd)
    lb = Vector{Float64}(undef, nd)

    open(grid.path, "r") do io
        for (r, w, g) in zip(crec, cwgt, cgeo)
            g == 2 && (nsph += 1)
            seek(io, grid.data0 + (r - 1) * grid.recw * 4)
            read!(io, buf)
            for j in 1:nd
                ltau[j] = log10(max(Float64(buf[j]), 1e-99))
            end

            for k in 1:ntrans, (slot, acc) in ((slot_lo[k], acc_lo), (slot_up[k], acc_up))
                for j in 1:nd
                    lb[j] = log10(max(Float64(buf[nd+(slot-1)*nd+j]), 1e-99))
                end
                for j in 1:nlayers
                    lt = log_tau_model[j]
                    # Mark layers outside the corner's own τ range.  This is reported, and is how
                    # the grids' truncation became visible at all; it must not be lost.
                    if lt < ltau[1] || lt > ltau[nd]
                        clamped[j] = true
                    end
                    t = if lt <= ltau[1]
                        lb[1]
                    elseif lt >= ltau[nd]
                        lb[nd]
                    else
                        i = min(max(searchsortedlast(ltau, lt), 1), nd - 1)
                        lb[i] + (lt - ltau[i]) / (ltau[i+1] - ltau[i]) * (lb[i+1] - lb[i])
                    end
                    acc[k, j] += w * t
                end
            end
            wsum += w
        end
    end

    b_lower = Matrix{acctype}(undef, ntrans, nlayers)
    b_upper = Matrix{acctype}(undef, ntrans, nlayers)
    for k in 1:ntrans, j in 1:nlayers
        if held_to_LTE && clamped[j]
            b_lower[k, j] = one(acctype)
            b_upper[k, j] = one(acctype)
        else
            b_lower[k, j] = 10^(acc_lo[k, j] / wsum)
            b_upper[k, j] = 10^(acc_up[k, j] / wsum)
        end
    end

    if verbose
        if wsum < 0.999
            @warn "NLTE: $sym — only $(_fmt(wsum, 3)) of the interpolation weight was " *
                  "populated; the rest was dropped and the remainder renormalised."
        end
        if 0 < nsph < length(crec)
            @warn "NLTE: $sym — corners mix spherical and plane-parallel MARCS geometry."
        end
        nheld = count(clamped)
        if nheld > 0
            @warn "NLTE: $sym — the grid does not cover this atmosphere in depth: $nheld of " *
                  "$nlayers layers hold an endpoint b (not extrapolated)."
        end
    end

    status = if dxoff != 0
        "nearest-dX(" * _sfmt(dxoff, 2) * ")"
    else
        "ok(w=" * _fmt(wsum, 3) * ")"
    end
    b_lower, b_upper, status
end

"""
    nlte_departures(atm, linelist, A_X; Teff, logg, kwargs...)

Interpolate departure coefficients from the published grids onto `atm`'s layers, returning an
[`Korg.NLTE`](@ref) ready to pass to [`synthesize`](@ref).

Only the transitions actually present in `linelist` are carried, and only the grids owning one of
them are opened — so a Na D synthesis needs no Ca file installed.  If the linelist contains no
eligible transition at all, this returns `nothing`, which `synthesize` treats as plain LTE.  That
is a legitimate outcome: synthesize 4000–4500 Å and Na D is simply not there.

# Arguments

  - `atm`: the model atmosphere.  Its `tau_ref` is used as τ₅₀₀₀, which is the coordinate the
    published grids are tabulated against, so its `reference_wavelength` must be 5000 Å.
  - `linelist`: the linelist the synthesis will use.
  - `A_X`: the 92-element abundance vector, as for [`synthesize`](@ref).

# Keyword arguments

  - `Teff`, `logg` (required): the atmosphere's parameters.  These are grid axes and cannot be
    recovered from the layers, so they must be given.
  - `vmic` (default: 1): microturbulence in km/s, a grid axis.  If you pass a vector, the value at
    the layer nearest τ₅₀₀₀ = 1 is used — where the lines form, rather than an average over a scale
    on which it may vary.
  - `transitions` (default: [`Korg.NLTE_TRANSITIONS`](@ref)): which transitions are eligible.
  - `grid_dir` (default: `\$KORG_NLTE_DIR`): where the `.nlte` runtime files live.  Per-element
    overrides (`\$NLTE_GRID_NA` etc.) win over this.
  - `energy_tolerance` (default: 5.0): level-energy match tolerance in cm⁻¹; see
    [`Korg.nlte_tags`](@ref).
  - `held_to_LTE` (default: `false`): a developer switch for bounding the truncation caveat.  The
    published grids' MARCS atmospheres stop short of Korg's surfaces, so the outermost layers hold
    the grid's endpoint b rather than a computed one — and the cores of strong lines form exactly
    there.  Setting this `true` forces b = 1 in every held layer instead, which is the opposite
    extreme; the truth lies between the two, so the pair brackets how much of a result rests on
    extrapolated data.  Not a physical option: it exists to be A/B'd against the default.
  - `verbose` (default: `true`): report the treatment each element received.  Every call prints one
    greppable `NLTE: STATUS` line, so a finished grid of syntheses can be audited rather than
    trusted.
"""
function nlte_departures(atm, linelist, A_X; Teff, logg, vmic=1.0,
                         transitions=NLTE_TRANSITIONS, grid_dir=nothing, energy_tolerance=5.0,
                         held_to_LTE=false, verbose=true)
    if !isapprox(atm.reference_wavelength, 5e-5; rtol=1e-6)
        throw(ArgumentError("NLTE departure coefficients are tabulated against τ₅₀₀₀, but this " *
                            "atmosphere's reference wavelength is " *
                            "$(atm.reference_wavelength * 1e8) Å."))
    end

    tags = nlte_tags(linelist, transitions; energy_tolerance=energy_tolerance)
    used = sort!(unique(filter(>(0), tags)))
    if isempty(used)
        verbose && @info "NLTE: no eligible transition lies in this linelist; continuing in LTE."
        return nothing
    end

    tau_5000 = [l.tau_ref for l in atm.layers]
    ξ = if vmic isa Number
        vmic
    else
        # the layer nearest τ₅₀₀₀ = 1, i.e. where the lines form
        vmic[argmin(abs.(log10.(max.(tau_5000, 1e-99))))]
    end
    M_H = A_X[26] - NLTE_A_FE_SUN

    kept = NLTETransition[]
    b_lo_blocks = []
    b_up_blocks = []
    status = String[]
    for e in unique(transitions[k].element for k in used)
        el = NLTE_ELEMENTS[e]
        ks = [k for k in used if transitions[k].element == e]
        trs = transitions[ks]
        grid = read_nlte_grid(nlte_grid_path(e; grid_dir=grid_dir))
        verbose && @info "NLTE: $(el.sym) grid — $grid"
        result = _interpolate_element(grid, trs, tau_5000, Teff, logg, M_H, ξ, A_X[el.Z];
                                      held_to_LTE=held_to_LTE, verbose=verbose, sym=el.sym)
        if isnothing(result)
            push!(status, "$(el.sym)=LTE")
            continue
        end
        b_lo, b_up, stat = result
        append!(kept, trs)
        push!(b_lo_blocks, b_lo)
        push!(b_up_blocks, b_up)
        push!(status, "$(el.sym)=$stat")
    end

    verbose && @info "NLTE: STATUS  Teff=$(_fmt(Teff, 1)) logg=$(_fmt(logg, 2)) " *
          "[Fe/H]=$(_sfmt(M_H, 2)) | " * join(status, " ")

    if isempty(kept)
        # Reverting to LTE is exact, not approximate: b = 1 makes the opacity scale and the
        # emissivity deviation identically 1 and 0.
        verbose &&
            @info "NLTE: every element reverted to LTE for this atmosphere; continuing in LTE."
        return nothing
    end
    verbose && @info "NLTE: on — $(count(>(0), tags)) tagged line components over " *
          "$(length(kept)) transitions"

    NLTE(kept, vcat(b_lo_blocks...), vcat(b_up_blocks...))
end

"""
    nlte_line_factors!(fkappa, fdev, b_lower, b_upper, β, E_lower, E_upper)

The two per-layer scalars an NLTE line needs, for one transition:

    fkappa = κ/κ_LTE = b_l [1 − (b_u/b_l)e^−x] / [1 − e^−x]
    fdev   = (b_u − fkappa) / fkappa

with x = hν₀/kT at line centre.  `fdev` is expressed per unit of the ALREADY-SCALED opacity so that
the caller can reuse the profile it built with the scaled amplitude: the emissivity deviation to
accumulate at each wavelength is just `profile * fdev`.

A population inversion (b_u/b_l > e^x) makes the opacity non-positive — the line masing rather than
absorbing.  Nothing downstream can represent that, so the line is forced back to LTE at that layer
(`fkappa = 1`, `fdev = 0`).  With sensible photospheric departure coefficients it never triggers;
if it does, the input grid is wrong.

Returns the number of layers that were forced back to LTE.
"""
function nlte_line_factors!(fkappa, fdev, b_lower, b_upper, β, E_lower, E_upper)
    ninverted = 0
    for j in eachindex(fkappa)
        bl, bu = b_lower[j], b_upper[j]
        lte = exp(-β[j] * E_lower) - exp(-β[j] * E_upper)
        nlte = bl * exp(-β[j] * E_lower) - bu * exp(-β[j] * E_upper)
        if nlte <= 0 || lte <= 0 || bl <= 0
            ninverted += 1
            fkappa[j] = one(eltype(fkappa))
            fdev[j] = zero(eltype(fdev))
        else
            fkappa[j] = nlte / lte
            fdev[j] = (bu - fkappa[j]) / fkappa[j]
        end
    end
    ninverted
end
