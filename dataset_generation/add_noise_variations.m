function noisy_measurements = add_noise_variations(measurements, config)
    % Get the field names of the input measurements
    fields = fieldnames(measurements);
    % Initialize the noisy_measurements structure
    noisy_measurements = struct();
    
    % Iterate over each measurement field
    for k = 1:length(fields)
        field = fields{k};
        data = measurements.(field);
        
        % Generate multiple noisy variations for this field
        noisy_data = repmat(data, config.num_noise_variations, 1) + ...
                     normrnd(config.noise_mu, config.noise_sigma, config.num_noise_variations, size(data, 2));
        
        % Store the noisy variations in the output structure
        noisy_measurements.(field) = noisy_data;
    end
end
