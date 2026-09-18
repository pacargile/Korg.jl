@testset "NLTE" begin
    @testset "transition table" begin
        # Every transition must name an element that exists, and the implied wavelength must be
        # sane.  The Ca II triplet members in particular transpose easily (8498 → 4p ²P₃/₂,
        # 8662 → 4p ²P₁/₂), so check the wavelength rather than trusting the level indices.
        for tr in Korg.NLTE_TRANSITIONS
            @test 1 <= tr.element <= length(Korg.NLTE_ELEMENTS)
            @test Korg.get_atoms(tr.species)[1] == Korg.NLTE_ELEMENTS[tr.element].Z
            @test tr.E_upper > tr.E_lower
            λ_vac = 1e8 / (tr.E_upper - tr.E_lower)   # Å
            # every label ends in the vacuum wavelength it should have
            λ_label = parse(Float64, split(tr.label)[end])
            @test λ_vac≈λ_label rtol=1e-4
        end
        # no two transitions of the same species may collide within the matching tolerance
        for (i, a) in enumerate(Korg.NLTE_TRANSITIONS), (j, b) in enumerate(Korg.NLTE_TRANSITIONS)
            if i < j && a.species == b.species
                @test abs(a.E_lower - b.E_lower) >= 5.0 || abs(a.E_upper - b.E_upper) >= 5.0
            end
        end
    end

    @testset "tagging" begin
        linelist = Korg.get_VALD_solar_linelist()
        tags = Korg.nlte_tags(linelist, Korg.NLTE_TRANSITIONS)

        # Na D1 and D2 are in the VALD solar linelist and must be found
        labels = Set(Korg.NLTE_TRANSITIONS[t].label for t in tags if t > 0)
        @test "Na I D2 5891.58" in labels
        @test "Na I D1 5897.56" in labels

        # every tagged line must actually be the species of its transition, and its wavelength must
        # match the transition's to within the linelist's own precision
        for (line, t) in zip(linelist, tags)
            t == 0 && continue
            tr = Korg.NLTE_TRANSITIONS[t]
            @test line.species == tr.species
            @test line.wl * 1e8≈1e8 / (tr.E_upper - tr.E_lower) rtol=1e-3
        end

        # an empty transition list tags nothing
        @test all(==(0), Korg.nlte_tags(linelist, Korg.NLTETransition[]))
    end

    @testset "line factors" begin
        β = [1 / (Korg.kboltz_eV * 5000.0)]
        E_lower, E_upper = 0.0, 2.1   # eV, roughly Na D
        fκ, fdev = zeros(1), zeros(1)

        # b = 1 is exactly LTE, not approximately
        Korg.nlte_line_factors!(fκ, fdev, [1.0], [1.0], β, E_lower, E_upper)
        @test fκ[1] == 1.0
        @test fdev[1] == 0.0

        # against the closed forms: κ/κ_LTE = b_l[1 − (b_u/b_l)e^−x]/[1 − e^−x], and the emissivity
        # is b_u times its LTE value, i.e. fκ(1 + fdev) == b_u
        for (bl, bu) in ((2.0, 1.2), (0.5, 0.8), (1.3, 1.3))
            Korg.nlte_line_factors!(fκ, fdev, [bl], [bu], β, E_lower, E_upper)
            x = β[1] * (E_upper - E_lower)
            @test fκ[1]≈bl * (1 - (bu / bl) * exp(-x)) / (1 - exp(-x)) rtol=1e-12
            @test fκ[1] * (1 + fdev[1])≈bu rtol=1e-12
            # S/B = b_u/fκ, which must equal (2hν³/c²)/[(b_l/b_u)e^x − 1] / B_ν
            S_over_B = (bu / fκ[1])
            @test S_over_B≈(exp(x) - 1) / ((bl / bu) * exp(x) - 1) rtol=1e-12
        end

        # a population inversion is forced back to LTE rather than producing negative opacity
        x = β[1] * (E_upper - E_lower)
        Korg.nlte_line_factors!(fκ, fdev, [1.0], [2exp(x)], β, E_lower, E_upper)
        @test fκ[1] == 1.0
        @test fdev[1] == 0.0
    end

    @testset "NLTE construction" begin
        trs = Korg.NLTE_TRANSITIONS[1:2]
        @test_throws ArgumentError Korg.NLTE(trs, ones(2, 5), ones(3, 5))
        @test_throws ArgumentError Korg.NLTE(trs, ones(3, 5), ones(3, 5))
        @test length(Korg.NLTE(trs, ones(2, 5), ones(2, 5))) == 2
    end

    @testset "null test: b = 1 reproduces LTE exactly" begin
        # This is the test that proves the retrofit is inert when it should be.  b = 1 makes the
        # opacity scale identically 1 and the emissivity deviation identically 0, by the same
        # algebra that makes reverting an element to LTE exact.
        atm = Korg.read_model_atmosphere("data/sun.mod")
        A_X = format_A_X()
        linelist = filter(Korg.get_VALD_solar_linelist()) do line
            5888e-8 < line.wl < 5900e-8
        end
        wls = (5890, 5898)

        lte = synthesize(atm, linelist, A_X, wls; hydrogen_lines=false)

        tags = Korg.nlte_tags(linelist, Korg.NLTE_TRANSITIONS)
        trs = Korg.NLTE_TRANSITIONS[sort!(unique(filter(>(0), tags)))]
        @test length(trs) == 2   # both Na D lines
        n = length(atm.layers)
        unity = Korg.NLTE(trs, ones(length(trs), n), ones(length(trs), n))
        null = synthesize(atm, linelist, A_X, wls; hydrogen_lines=false, nlte=unity)

        @test null.flux == lte.flux
        @test null.alpha == lte.alpha
        @test all(iszero, null.nlte_emissivity_deviation)
        @test isnothing(lte.nlte_emissivity_deviation)
    end

    @testset "departures deepen the Na D cores" begin
        # b_lower > b_upper > 1 is the photon-loss signature of an alkali resonance line: the lower
        # level is overpopulated (more opacity) and S < B (darker core).  Both must show up.
        atm = Korg.read_model_atmosphere("data/sun.mod")
        A_X = format_A_X()
        linelist = filter(Korg.get_VALD_solar_linelist()) do line
            5888e-8 < line.wl < 5900e-8
        end
        wls = (5890, 5898)

        tags = Korg.nlte_tags(linelist, Korg.NLTE_TRANSITIONS)
        trs = Korg.NLTE_TRANSITIONS[sort!(unique(filter(>(0), tags)))]
        n = length(atm.layers)
        # a synthetic, monotonic departure profile with the right shape, so this test does not
        # need the 800 MB published grid
        depth_factor = range(1.8, 1.0; length=n)
        b_lower = repeat(collect(depth_factor)', length(trs))
        b_upper = repeat(collect(1 .+ (depth_factor .- 1) ./ 3)', length(trs))

        lte = synthesize(atm, linelist, A_X, wls; hydrogen_lines=false)
        nlte = synthesize(atm, linelist, A_X, wls; hydrogen_lines=false,
                          nlte=Korg.NLTE(trs, b_lower, b_upper))

        λ = lte.wavelengths
        for λ0 in (5891.583, 5897.558)
            i = argmin(abs.(λ .- λ0))
            @test nlte.flux[i] < lte.flux[i]
        end
        # the deviation is negative where S < B (b_upper < the opacity scale), and it follows the
        # tagged lines' profiles -- far larger at a core than at the edge of the window.  (There is
        # no point in this window truly free of Na D: 5890 Å is still inside D2's damping wings.)
        @test any(<(0), nlte.nlte_emissivity_deviation)
        surf = nlte.nlte_emissivity_deviation[1, :]
        core = argmin(abs.(λ .- 5891.583))
        @test abs(surf[core]) > 10 * abs(surf[1])
    end

    @testset "autodiff through the NLTE source function" begin
        # The NLTE path must stay generic over number types like the rest of the pipeline.  Uses
        # hand-supplied coefficients so it does not need the published grid.
        atm = Korg.read_model_atmosphere("data/sun.mod")
        linelist = filter(Korg.get_VALD_solar_linelist()) do line
            5888e-8 < line.wl < 5900e-8
        end
        tags = Korg.nlte_tags(linelist, Korg.NLTE_TRANSITIONS)
        trs = Korg.NLTE_TRANSITIONS[sort!(unique(filter(>(0), tags)))]
        n = length(atm.layers)
        nlte = Korg.NLTE(trs, fill(1.5, length(trs), n), fill(1.2, length(trs), n))

        flux_at(ANa) =
            let A_X = format_A_X(0.0, Dict("Na" => ANa))
                sum(synthesize(atm, linelist, A_X, (5891, 5892); hydrogen_lines=false,
                               nlte=nlte).flux)
            end
        d = ForwardDiff.derivative(flux_at, 0.0)
        @test isfinite(d)
        @test d != 0
        # finite differences agree
        fd = (flux_at(0.01) - flux_at(-0.01)) / 0.02
        @test d≈fd rtol=1e-3
    end

    @testset "synthetic grid" begin
        # A tiny .nlte file, written here in the documented format, so that the reader, the 5-axis
        # interpolation, the τ remapping and the fallback policy are all covered without the ~800 MB
        # published grids.
        nT, nG, nF, nV, nD, nlev, ndep = 2, 2, 2, 1, 3, 3, 4
        aT = Float32[5000, 6000]
        aG = Float32[4.0, 5.0]
        aF = Float32[-1.0, 0.0]
        aV = Float32[1.0]
        aD = Float32[5.0, 6.0, 7.0]
        level_ids = Int32[1, 2, 7]
        τ_grid = Float32[1e-4, 1e-2, 1e0, 1e2]

        # `rec` is Fortran/column-major over (T, G, F, V, D); 0 marks an absent cell.  Populate
        # every cell of the dX = 5.0 and dX = 7.0 planes, and leave the whole middle (dX = 6.0)
        # plane empty, so the abundance fallback has something to find.
        rec = zeros(Int32, nT, nG, nF, nV, nD)
        n = 0
        for iD in (1, 3), iF in 1:nF, iG in 1:nG, iT in 1:nT
            n += 1
            rec[iT, iG, iF, 1, iD] = n
        end
        nrec = n
        geo = Int8.((rec .> 0))   # all plane-parallel

        # each record's b is a constant per level, keyed off the record number so a mis-indexed
        # read is unmistakable
        recdata = map(1:nrec) do r
            vcat(τ_grid, fill(Float32(1 + r / 100), ndep), fill(Float32(2 + r / 100), ndep),
                 fill(Float32(3 + r / 100), ndep))
        end

        path = joinpath(mktempdir(), "Test_synthetic.nlte")
        open(path, "w") do io
            write(io, Int32[nT, nG, nF, nV, nD, nlev, ndep, nrec])
            write(io, aT)
            write(io, aG)
            write(io, aF)
            write(io, aV)
            write(io, aD)
            write(io, level_ids)
            write(io, vec(rec))
            write(io, vec(geo))
            for d in recdata
                write(io, d)
            end
        end

        grid = Korg.read_nlte_grid(path)
        @test length.(grid.axes) == (nT, nG, nF, nV, nD)
        @test grid.level_ids == [1, 2, 7]
        @test grid.nrec == nrec
        @test grid.recw == ndep + nlev * ndep

        # a truncated file is caught at load, not deep inside the interpolation
        truncated = joinpath(mktempdir(), "Test_truncated.nlte")
        write(truncated, read(path)[1:(end-16)])
        @test_throws ArgumentError Korg.read_nlte_grid(truncated)

        tr(lo, up) = Korg.NLTETransition(Korg.Species("Na I"), 0.0, 1e4, lo, up, 1, "test 10000.0")
        τ_model = [1e-3, 1e-1, 1e1]

        # exactly on a grid node: b must be that record's constant value, on every layer
        b_lo, b_up, status = Korg._interpolate_element(grid, [tr(1, 2)], τ_model, 5000.0, 4.0,
                                                       -1.0, 1.0, 4.0; verbose=false)
        @test all(b_lo .≈ 1 + rec[1, 1, 1, 1, 1] / 100)   # level 1 (slot 1)
        @test all(b_up .≈ 2 + rec[1, 1, 1, 1, 1] / 100)   # level 2 (slot 2)
        @test startswith(status, "ok(")

        # level 7 lives in slot 3, not slot 7: the level list is a property of the FILE
        _, b_up7, _ = Korg._interpolate_element(grid, [tr(1, 7)], τ_model, 5000.0, 4.0, -1.0, 1.0,
                                                4.0; verbose=false)
        @test all(b_up7 .≈ 3 + rec[1, 1, 1, 1, 1] / 100)

        # a level the file does not carry must fail loudly rather than read the wrong slot
        @test_throws ArgumentError Korg._interpolate_element(grid, [tr(1, 4)], τ_model, 5000.0,
                                                             4.0, -1.0, 1.0, 4.0; verbose=false)

        # hotter than the last Teff node reverts to LTE rather than clamping
        @test isnothing(Korg._interpolate_element(grid, [tr(1, 2)], τ_model, 9000.0, 4.0, -1.0,
                                                  1.0, 4.0; verbose=false))
        # ... while every other axis clamps and still returns coefficients
        @test !isnothing(Korg._interpolate_element(grid, [tr(1, 2)], τ_model, 5000.0, 9.0, -1.0,
                                                   1.0, 4.0; verbose=false))

        # THE ABUNDANCE FALLBACK.  A(X) − [Fe/H] = 6.0 lands squarely on the empty middle plane, so
        # every interpolation corner is absent and the nearest populated abundance node must be
        # used.  (`for a, b` in Julia is one loop nest, so a `break` meant for the inner loop would
        # abandon the whole search and silently revert to LTE here.)
        b_lo2, _, status2 = Korg._interpolate_element(grid, [tr(1, 2)], τ_model, 5000.0, 4.0, -1.0,
                                                      1.0, 5.0; verbose=false)
        @test startswith(status2, "nearest-dX")
        @test all(isfinite, b_lo2)
        @test all(>(0), b_lo2)

        # τ outside the grid's own range is held at the endpoint, never extrapolated
        b_lo3, _, _ = Korg._interpolate_element(grid, [tr(1, 2)], [1e-9, 1e9], 5000.0, 4.0, -1.0,
                                                1.0, 4.0; verbose=false)
        @test b_lo3[1, 1] ≈ b_lo3[1, 2]   # both ends hold the same constant record
    end

    @testset "grid file" begin
        # Skipped unless the published runtime grid is installed; these files are ~800 MB and are
        # not shipped with Korg.  Set $KORG_NLTE_DIR (or $NLTE_GRID_NA) to run them.
        path = try
            Korg.nlte_grid_path(1)
        catch
            nothing
        end
        if isnothing(path) || !isfile(path)
            @info "NLTE grid tests skipped: no Na grid installed (set \$KORG_NLTE_DIR)"
        else
            grid = Korg.read_nlte_grid(path)
            @test length(grid.axes) == 5
            @test grid.recw == grid.ndep + grid.nlev * grid.ndep
            @test all(>(0), length.(grid.axes))
            @test count(>(0), grid.rec) <= grid.nrec

            atm = Korg.read_model_atmosphere("data/sun.mod")
            A_X = format_A_X()
            linelist = filter(Korg.get_VALD_solar_linelist()) do line
                5888e-8 < line.wl < 5900e-8
            end
            nlte = Korg.nlte_departures(atm, linelist, A_X; Teff=5777.0, logg=4.44, verbose=false)
            @test length(nlte) == 2
            # both D lines share the 3s ²S lower level, so their b_lower must be identical
            @test nlte.b_lower[1, :] == nlte.b_lower[2, :]
            # ... but the two 3p fine-structure levels are not the same level
            @test nlte.b_upper[1, :] != nlte.b_upper[2, :]
            # departures go to LTE at depth and overpopulate the lower level at the surface
            @test nlte.b_lower[1, end]≈1.0 atol=1e-2
            @test nlte.b_lower[1, 1] > 1.5
            @test all(>(0), nlte.b_lower)
            @test all(>(0), nlte.b_upper)

            # no eligible transition in the window ⇒ nothing, i.e. plain LTE
            far = filter(l -> 4500e-8 < l.wl < 4520e-8, Korg.get_VALD_solar_linelist())
            @test isnothing(Korg.nlte_departures(atm, far, A_X; Teff=5777.0, logg=4.44,
                                                 verbose=false))
        end
    end
end
