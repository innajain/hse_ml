function successful_count = vary_run_save(mpc, mpopt, config, measurements, true_voltages)
    successful_count = 0;
    
    for i = 1:config.num_power_variations
        variation = randomize_loads_and_gens(mpc);
        pf_result = runpf(variation, mpopt);
        
        if ~pf_result.success
            disp(['Power flow FAILED for load variation ' num2str(i)]);
            continue;
        end
        
        successful_count = successful_count + 1;
        
        % Store true voltage values for the current iteration
        true_voltages.magnitude = pf_result.bus(:, 8)';
        true_voltages.angle = pf_result.bus(:, 9)';
        
        % Calculate power flow measurements
        measurements = calculate_measurements(pf_result);
        
        % Generate noisy measurements for the current iteration
        noisy_measurements = add_noise_variations(measurements, config);
        
        % Save the results incrementally
        save_incremental_measurements(noisy_measurements, true_voltages);
        
        
        disp(['Power flow completed for load variation ' num2str(i)]);
    end
end
