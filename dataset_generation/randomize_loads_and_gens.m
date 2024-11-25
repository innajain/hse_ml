function power_variation = randomize_loads_and_gens(mpc)
    % Copy the original MATPOWER case structure
    power_variation = mpc;

    frac_variation_load = 0.2;
    frac_variation_gen = 0.2;
    frac_variation_genV = 0.2;

    % Randomize load demands (Pd, Qd)
    % Pd - Active power demand (column 3 of bus matrix)
    power_variation.bus(:, 3) = mpc.bus(:, 3) .* (
        1 - frac_variation_load + 2 * frac_variation_load
         * rand(size(mpc.bus(:, 3))));
    % Qd - Reactive power demand (column 4 of bus matrix)
    power_variation.bus(:, 4) = mpc.bus(:, 4) .* (
        1 - frac_variation_load + 2 * frac_variation_load
         * rand(size(mpc.bus(:, 4))));

    % Randomize generator outputs (Pg, Qg)
    % Pg - Active power generation (column 2 of gen matrix)
    power_variation.gen(:, 2) = mpc.gen(:, 2) .* (
        1 - frac_variation_gen + 2 * frac_variation_gen
         * rand(size(mpc.gen(:, 2))));
    % Qg - Reactive power generation (column 3 of gen matrix)
    power_variation.gen(:, 3) = mpc.gen(:, 3) .* (
        1 - frac_variation_gen + 2 * frac_variation_gen
         * rand(size(mpc.gen(:, 3))));

    % Randomize generator voltage setpoints (Vg)
    % Vg - Voltage magnitude setpoints (column 6 of gen matrix)
    power_variation.gen(:, 6) = mpc.gen(:, 6) .* (
        1 - frac_variation_genV + 2 * frac_variation_genV
         * rand(size(mpc.gen(:, 6))));

    % Return the modified MATPOWER case structure
end
