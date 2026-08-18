# tests of things in src/line_absorption.jl and src/hydrogen_line_absorption.jl

@testset "line profiles" begin
    @testset "generic line profile" begin
        Δ = 0.01
        wls = (4750:Δ:5250) * 1e-8
        Δ *= 1e-8
        amplitude = 7.0
        for σ in [1e-7, 1e-8, 1e-9], γ in [3e-8, 3e-9, 3e-10]
            ϕ = Korg.line_profile.(5e-5, σ, γ, amplitude, wls)
            # the profile isn't perfectly monotonic because the approximation has "seams" at v=5
            # this allows for slight nonmonotonicity
            @test all(diff(ϕ[1:Int(ceil(end / 2))]) .> -1e-3 * maximum(ϕ))
            @test all(diff(ϕ[Int(ceil(end / 2)):end]) .< 1e-3 * maximum(ϕ))
            @test 0.98 < sum(ϕ .* Δ) / amplitude < 1
        end
    end

    @testset "hydrogen stark profiles" begin
        # This test data was generated with Korg.hydrogen_line_absorption shortly
        # after writing the function. This data is consistent with the results
        # produced by the Fortran code distributed with Stehle & Hutcheon 1999
        fname = "data/lyman_absorption.h5"
        αs_ref = h5read(fname, "profile")

        fid = h5open(fname)
        T = HDF5.read_attribute(fid["profile"], "T")
        ne = HDF5.read_attribute(fid["profile"], "ne")
        nH_I = HDF5.read_attribute(fid["profile"], "nH_I")
        wls = Korg.Wavelengths((HDF5.read_attribute(fid["profile"], "start_wl"):
                                HDF5.read_attribute(fid["profile"], "wl_step"):
                                HDF5.read_attribute(fid["profile"], "stop_wl")))
        close(fid)

        αs = zeros(length(wls))
        Korg.hydrogen_line_absorption!(αs, wls, 9000.0, ne, nH_I, 0.0,
                                       Korg.default_partition_funcs[Korg.species"H_I"](log(9000.0)),
                                       0.0, 15e-7; use_MHD=false)
        @test assert_allclose_grid(αs_ref, αs, [("λ", wls * 1e8, "Å")]; atol=5e-9)

        #make sure that H line absorption doesn't return NaNs on inputs where it used to
        wls = Korg.Wavelengths(3800:0.01:4200)
        αs = zeros(length(wls))
        Korg.hydrogen_line_absorption!(αs, wls, 9000.0, 1.1e16, 1, 0.0,
                                       Korg.default_partition_funcs[Korg.species"H_I"](log(9000.0)),
                                       0.0, 15e-7)
        @test all(.!isnan.(αs))
    end

    @testset "Brackett line profile centered correctly" begin
        # The Brackett-series Stark profile should peak near the line center.
        # Bug: off-by-one in convolution extraction shifts the profile by one bin.
        n_upper = 7  # Brackett γ (transition 4 → 7)
        n_lower = 4
        E = Korg.RydbergH_eV * (1 / n_lower^2 - 1 / n_upper^2)
        λ₀ = Korg.hplanck_eV * Korg.c_cgs / E  # cm

        T = 8000.0
        nₑ = 1e14
        ξ = 1e5  # 1 km/s in cm/s

        itp, window = Korg.bracket_line_interpolator(n_upper, λ₀, T, nₑ, ξ)

        # Sample finely around line center to find the peak
        test_wls = range(λ₀ - window / 10, λ₀ + window / 10; length=10001)
        vals = [itp(w) for w in test_wls]
        peak_wl = test_wls[argmax(vals)]

        # The peak should be very close to the line center.
        # The internal grid spacing is roughly 2*window/201 ≈ window/100.
        # With the off-by-one bug, the peak is shifted by one full grid step.
        @test abs(peak_wl - λ₀) < window / 500
    end
end

@testset "molecular van der Waals broadening" begin
    # per-species molecular vdW widths from ExoMol + Gharib-Nezhad et al. (2021).
    # See src/molecular_broadening.jl.

    @testset "parameter table" begin
        params = Korg.molecular_vdW_params
        @test length(params) == 22
        for spec in ["TiO", "H2O", "CaH", "CO", "CaOH", "AlO", "TiH"]
            @test haskey(params, Korg.Species(spec))
        end
        # AlO and TiH are Korg additions beyond SYNTHE's 20. TiH takes GN+21's metal-hydride
        # group values, matching its siblings; AlO takes its own ExoMol file and has no J-slope.
        @test params[Korg.Species("TiH")].γ_H2 == params[Korg.Species("CrH")].γ_H2
        @test params[Korg.Species("TiH")].dγdJ_H2 == params[Korg.Species("FeH")].dγdJ_H2
        @test params[Korg.Species("AlO")].dγdJ_H2 == 0
        # every species Korg has broadening data for must also have the chemistry to make it
        for spec in keys(params)
            @test haskey(Korg.default_partition_funcs, spec)
            @test haskey(Korg.default_log_equilibrium_constants, spec)
        end
        # every entry must be physical: positive widths, non-negative slopes and B
        for (spec, p) in params
            @test Korg.ismolecule(spec)
            @test p.γ_H2 > 0 && p.γ_He > 0
            @test p.n_H2 > 0 && p.n_He > 0
            @test p.dγdJ_H2 >= 0 && p.dγdJ_He >= 0
            @test p.B > 0
            # He broadening is always weaker than H2 broadening
            @test p.γ_He < p.γ_H2
        end

        tio = params[Korg.Species("TiO")]
        @test tio.γ_H2 == 0.1        # Gharib-Nezhad+21 Table 1
        @test tio.dγdJ_H2 == 0.002
        @test tio.B == 0.535
    end

    @testset "rotational averaging" begin
        # with no J-dependence, ⟨γ(J)⟩ is just γ₀ at any temperature
        @test Korg.rotationally_averaged_γ(0.1, 0.0, 0.535, 3000.0) == 0.1
        @test Korg.rotationally_averaged_γ(0.1, 0.0, 0.535, 300.0) == 0.1
        # and with no rotational constant we have nothing to average over
        @test Korg.rotationally_averaged_γ(0.1, 0.002, 0.0, 3000.0) == 0.1

        # ⟨γ(J)⟩ is bounded by γ₀ and the Gharib-Nezhad floor, and decreases with temperature as
        # higher J become populated
        γs = [Korg.rotationally_averaged_γ(0.1, 0.002, 0.535, T) for T in [300, 1000, 3000, 5000]]
        @test all(0.1 * Korg.molecular_vdW_J_floor .<= γs .<= 0.1)
        @test issorted(γs; rev=true)

        # a light molecule (large B) stays near J=0 where the decline hasn't bitten;
        # a heavy one (small B) is far down it.  This is the whole reason for the J-dependence.
        γ_light = Korg.rotationally_averaged_γ(0.1, 0.002, 14.5, 3000.0)  # H2O-like B
        γ_heavy = Korg.rotationally_averaged_γ(0.1, 0.002, 0.535, 3000.0) # TiO-like B
        @test γ_heavy < γ_light < 0.1

        # in the T → ∞ limit essentially every populated J is past the floor.  It doesn't converge
        # exactly onto it because the sum is truncated at molecular_vdW_J_max, leaving the
        # (2J+1)-weighted 2% of the ladder below J = 45 unfloored.
        @test Korg.rotationally_averaged_γ(0.1, 0.002, 0.535, 1e8) ≈ 0.1 * Korg.molecular_vdW_J_floor rtol=0.1
    end

    @testset "Γ conversion" begin
        TiO = Korg.Species("TiO")
        params = Korg.molecular_vdW_params[TiO]

        # Cross-check against ATLAS12/SYNTHE (load_mol_broad, atlas12 697642b), which converts the
        # same ExoMol/GN+21 data with the same arithmetic *except* that it omits the HWHM → FWHM
        # factor of 2.  Its γ_w for TiO at its 3000 K reference is 9.73e-10, multiplying an
        # effective perturber density txnxn = (n_HI + 0.42 n_HeI + 0.85 n_H2)(T/1e4)^0.3.  In a
        # pure-H₂ gas its ÷0.85 and ×0.85 cancel exactly, so the ratio isolates that factor.
        T = [3000.0]
        n_H2 = 1e17
        nd = Dict(Korg.species"H2" => [n_H2], Korg.species"He_I" => [0.0],
                  Korg.species"H_I" => [0.0])
        Γ = Korg.molecular_vdW_Γ(TiO, params, T, nd)
        Γ_synthe = 9.73e-10 * 0.85 * n_H2 * (3000 / 1e4)^0.3
        @test Γ[1] / Γ_synthe ≈ 2.0 rtol=1e-3

        # Γ is linear in each perturber density...
        nd2 = Dict(Korg.species"H2" => [2n_H2], Korg.species"He_I" => [0.0],
                   Korg.species"H_I" => [0.0])
        @test Korg.molecular_vdW_Γ(TiO, params, T, nd2)[1] ≈ 2Γ[1]
        # ...and each perturber adds to it, He less than H2 per particle (γ_He < γ_H2)
        nd_He = Dict(Korg.species"H2" => [0.0], Korg.species"He_I" => [n_H2],
                     Korg.species"H_I" => [0.0])
        nd_H = Dict(Korg.species"H2" => [0.0], Korg.species"He_I" => [0.0],
                    Korg.species"H_I" => [n_H2])
        Γ_He = Korg.molecular_vdW_Γ(TiO, params, T, nd_He)[1]
        Γ_H = Korg.molecular_vdW_Γ(TiO, params, T, nd_H)[1]
        @test 0 < Γ_He < Γ[1]
        # H I broadening is scaled from H₂ by α^0.4 μ^-0.3, ≈ 1.14 for a molecule this heavy
        @test Γ_H / Γ[1] ≈ 1.14 rtol=0.02

        nd_all = Dict(Korg.species"H2" => [n_H2], Korg.species"He_I" => [n_H2],
                      Korg.species"H_I" => [n_H2])
        @test Korg.molecular_vdW_Γ(TiO, params, T, nd_all)[1] ≈ Γ[1] + Γ_He + Γ_H

        # the temperature dependence is (296/T)^n × the kT of the ideal gas law, per perturber, and
        # the rotational average, so Γ at fixed density is not monotonic in a trivial way -- but it
        # must always be positive and finite over the range molecules exist in
        for T in [1000.0, 2000.0, 3000.0, 5000.0]
            Γ_T = Korg.molecular_vdW_Γ(TiO, params, [T], nd)[1]
            @test isfinite(Γ_T) && Γ_T > 0
        end
    end

    @testset "line_absorption! integration" begin
        # Korg's table overrides the linelist's vdW for molecules it covers, and leaves molecules
        # it doesn't cover alone.
        wls = Korg.Wavelengths(6157.0:0.01:6162.0)
        temps = [3000.0, 3500.0]
        nₑ = [1e10, 2e10]
        n_densities = Dict(Korg.species"H_I" => [1e17, 1e17],
                           Korg.species"He_I" => [1e16, 1e16],
                           Korg.species"H2" => [1e16, 1e16],
                           Korg.species"TiO" => [1e10, 1e10],
                           Korg.species"ZrO" => [1e10, 1e10])
        pfs = Korg.default_partition_funcs
        α_cntm = λ -> 1e-10

        function opacity(spec, vdW)
            line = Korg.Line(6159.5e-8, -1.0, Korg.Species(spec), 1.0, 1e7, 0.0, vdW)
            α = zeros(length(temps), length(wls))
            Korg.line_absorption!(α, [line], wls, temps, nₑ, n_densities, pfs, 1e5, α_cntm)
            α
        end

        # TiO is in the table: the linelist's γ_vdW is ignored, so a 100x change does nothing
        @test opacity("TiO", -7.0) == opacity("TiO", -9.0)
        # ZrO is not: it still uses whatever the linelist gave it
        @test opacity("ZrO", -7.0) != opacity("ZrO", -9.0)

        # the un-tabulated species is the control: broadening redistributes opacity from the core
        # into the wings without changing the wavelength-integrated absorption
        α_broadened = opacity("ZrO", -7.0)
        α_radiative_only = opacity("ZrO", 0.0)
        @test maximum(α_broadened) < maximum(α_radiative_only)
        @test sum(α_broadened[1, :]) ≈ sum(α_radiative_only[1, :]) rtol=1e-3

        # TiO gets real Lorentz wings from the table.  At 3000 K its Doppler σ here is 0.019 Å, so
        # 0.2 Å out is 10σ, where a Doppler-only profile is down by exp(-50) ≈ 8e-24.  The observed
        # ratio is ~6e-5, i.e. nineteen orders of magnitude of Lorentz wing.
        α_TiO = opacity("TiO", -7.0)
        i_center = argmax(α_TiO[1, :])
        i_wing = searchsortedfirst(wls, (6159.5 + 0.2) * 1e-8)
        σ_doppler = Korg.doppler_width(6159.5e-8, temps[1], Korg.get_mass(Korg.Species("TiO")), 1e5)
        @test 0.2e-8 / σ_doppler > 10 # the offset really is deep in the Gaussian's tail
        @test α_TiO[1, i_wing] / α_TiO[1, i_center] > 1e-5
    end
end
