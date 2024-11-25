function measurements = calculate_measurements(pf_result)
    % Initialize Pg and Qg
    Pg = zeros(size(pf_result.bus, 1), 1);
    Qg = zeros(size(pf_result.bus, 1), 1);
    
    % Map generation values
    gen_buses = pf_result.gen(:, 1);
    Pg(gen_buses) = pf_result.gen(:, 2);
    Qg(gen_buses) = pf_result.gen(:, 3);
    
    % Calculate measurements
    baseMVA = pf_result.baseMVA;
    measurements.p_injections = (Pg' - pf_result.bus(:, 3)') ./ baseMVA;
    measurements.q_injections = (Qg' - pf_result.bus(:, 4)') ./ baseMVA;
    
    measurements.from_bus_p_injection = pf_result.branch(:, 14)' ./ baseMVA;
    measurements.from_bus_q_injection = pf_result.branch(:, 15)' ./ baseMVA;
    measurements.to_bus_p_injection = pf_result.branch(:, 16)' ./ baseMVA;
    measurements.to_bus_q_injection = pf_result.branch(:, 17)' ./ baseMVA;
    
    measurements.voltage_mag = pf_result.bus(:, 8)';
    measurements.voltage_angles_deg = pf_result.bus(:, 9)';
    
    % Calculate current measurements
    [measurements.from_current_magnitudes, measurements.from_current_angles_deg, ...
     measurements.to_current_magnitudes, measurements.to_current_angles_deg] = ...
        calculate_currents(pf_result, measurements);
end
