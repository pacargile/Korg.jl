@testset "precomputed molecular opacities" begin
    # use 10 Å of the APOGEE linelist for testing
    linelist = Korg.get_APOGEE_DR17_linelist()
    filter!(linelist) do line
        15020 < line.wl * 1e8 < 15030
    end

    # split into water and non-water lines
    water_mask = [l.species == Korg.species"H2O" for l in linelist]
    linelist_less_water = linelist[.!water_mask]
    water_lines = linelist[water_mask]

    wls = [15020:0.01:15025, 15030:0.01:15035]

    # MolecularCrossSection is indexed by (vmic, T, λ), which assumes a density-independent line
    # profile.  That held upstream, where molecular lines got no pressure broadening.  This fork
    # broadens them (see molecular_broadening.jl and the vdW term in line_absorption!), so the
    # table can no longer represent the profile and the constructor refuses -- rather than silently
    # tabulating zeros, which is what it did before the guard was added.  Full diagnosis in
    # ai/KORG_CHANGES_SUMMARY.md §13.
    #
    # The branch is on the capability, not hard-coded, so the original round-trip tests come back
    # by themselves if the table gains a perturber-density axis (or molecular vdW is turned off).
    if Korg._density_dependent_broadening(water_lines)
        @test_throws ArgumentError Korg.MolecularCrossSection(water_lines, wls)
    else
        table = Korg.MolecularCrossSection(water_lines, wls)

        # this test fails if I truncate the atmosphere to 40:43, which _may_ indicate that it's worth
        # better understanding the precision of interpolation here
        atm = interpolate_marcs(4000.0, 4.5)

        filename = tempname()
        Korg.save_molecular_cross_section(filename, table)
        deserialized_table = Korg.read_molecular_cross_section(filename)
        @test table.itp.itp.coefs == deserialized_table.itp.itp.coefs
        @test all(table.wls .≈ deserialized_table.wls)
        @test table.species == deserialized_table.species

        depth_dependent_vmics = fill(0.5, length(atm.layers))
        depth_dependent_vmics[20:end] .= 1.0
        @testset for vmic in [0.5, depth_dependent_vmics]
            sol_without = synthesize(atm, linelist, format_A_X(), wls; vmic=vmic)
            sol_with = synthesize(atm, linelist_less_water, format_A_X(), wls;
                                  molecular_cross_sections=[deserialized_table], vmic=vmic)

            @test assert_allclose(sol_without.flux, sol_with.flux; rtol=1e-3)
        end
    end
end
