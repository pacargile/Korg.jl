@testset "autodiff" begin
    using ForwardDiff, FiniteDiff

    linelist = [Korg.get_VALD_solar_linelist(); Korg.get_APOGEE_DR17_linelist()]
    #cover the optical and the IR to catch different H lines
    wls = [6564:0.01:6565, 15_045:0.01:15_046]
    for atm in Korg.read_model_atmosphere.([
                                               "data/sun.mod",
                                               "data/s6000_g+1.0_m0.5_t05_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod"
                                           ])
        # the second model atmosphere happens to be in a weird (probably unphysical) part of
        # parameter space where the electron number densities calculated doesn't match the marcs
        # numbers.
        function flux(p)
            synthesize(atm, linelist, format_A_X(p[1], Dict("Ni" => p[2])), wls;
                       vmic=p[3], electron_number_density_warn_threshold=Inf).flux
        end
        #make sure this works.
        J = ForwardDiff.jacobian(flux, [0.0, 0.0, 1.5])
        @test .!any(isnan.(J))
    end

    @testset "autodiff through coherent scattering" begin
        # Regression test for the buffer-typing bug fixed alongside this test. With
        # coherent_scattering=true the scattering solver is handed the CONTINUUM opacity, which
        # carries no line-parameter dependence and is therefore Float64 under ForwardDiff, while
        # α_ref (and so the anchored-τ integrand factor, Λ*, and the ALI source function) is Dual.
        # Buffers sized off eltype(α) alone then threw MethodError(Float64, ::Dual) — first in
        # lambda_star_diagonal's τ/integrand_buffer, then in solve_scattering_source_function's
        # Ng-acceleration history, which is typed off the (Float64) Planck function B.
        #
        # Nothing else in the suite exercises coherent_scattering at all, let alone under AD,
        # which is why the bug survived. Keep this test.
        atm = Korg.read_model_atmosphere("data/sun.mod")
        ll = filter(l -> 5165e-8 <= l.wl <= 5185e-8, Korg.get_VALD_solar_linelist())
        A_X = format_A_X(0.0)
        wls = (5170.0, 5180.0, 0.02)

        loss(t) = let l2 = [Korg.Line(l; log_gf=l.log_gf + t) for l in ll]
            s = synthesize(atm, l2, A_X, wls; coherent_scattering=true,
                           electron_number_density_warn_threshold=Inf)
            sum(abs2, 1 .- s.flux ./ s.cntm)
        end

        g = ForwardDiff.derivative(loss, 0.0)
        @test isfinite(g)
        @test !iszero(g)   # a dropped-partials regression would give exactly 0, not a throw

        # the derivative must be RIGHT, not merely finite: a type fix that silently discards
        # partials still returns a number. The scattering solver converges to tol=1e-4 on
        # max|ΔS/S|, so compare against a central difference at a loose-but-meaningful tolerance.
        h = 1e-5
        fd = (loss(h) - loss(-h)) / 2h
        @test g ≈ fd rtol=1e-3

        # coherent_scattering must actually change the answer, or the test above proves nothing
        lte = synthesize(atm, ll, A_X, wls; coherent_scattering=false,
                         electron_number_density_warn_threshold=Inf)
        coh = synthesize(atm, ll, A_X, wls; coherent_scattering=true,
                         electron_number_density_warn_threshold=Inf)
        @test lte.flux != coh.flux
    end

    @testset "autodiff just one abundance" begin
        atm = Korg.read_model_atmosphere("data/sun.mod")
        linelist = [Korg.Line(6000e-8, 0.0, Korg.species"C I", 0.0)]
        # If this line is super weak (or removed), the test will fail due to numerics in the
        # abundances and continuum opacities. It used to be a fake line at 5000, but the fact that
        # that's the reference wavelength caused instability in the finie differnces calculation.
        function f(A_C)
            A_X = format_A_X(Dict("C" => A_C))
            synthesize(atm, linelist, A_X, (6000, 6000)).flux[1]
        end
        @test FiniteDiff.finite_difference_derivative(f, 0.0)≈ForwardDiff.derivative(f, 0.0) rtol=1e-4
    end

    @testset "line params" begin
        atm = Korg.read_model_atmosphere("data/sun.mod")
        function f(loggf)
            linelist = [Korg.Line(6000e-8, loggf, Korg.species"Na I", 0.0)]
            # This used to be a fake line at 5000, but the fact that that's the reference wavelength
            # caused instability in the finie differnces calculation.
            synthesize(atm, linelist, format_A_X(), (6000, 6000)).flux[1]
        end
        @test FiniteDiff.finite_difference_derivative(f, 0.0)≈ForwardDiff.derivative(f, 0.0) rtol=1e-4
    end
end
