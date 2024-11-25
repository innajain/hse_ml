function save_incremental_measurements(measurements, true_voltages)
    % Create directories if they don't exist
    if ~exist('synthetic_measurements', 'dir')
        mkdir('synthetic_measurements');
    end
    if ~exist('pf_states', 'dir')
        mkdir('pf_states');
    end
    if ~exist('io_for_ml_model', 'dir')
        mkdir('io_for_ml_model');
    end

    % Save true voltage values incrementally
    voltage_mag_file = 'pf_states/true_voltage_magnitudes.csv';
    voltage_angle_file = 'pf_states/true_voltage_angles.csv';
    dlmwrite(voltage_mag_file, true_voltages.magnitude, '-append');
    dlmwrite(voltage_angle_file, true_voltages.angle, '-append');

    % Save individual noisy measurement files incrementally
    fields = fieldnames(measurements);
    all_measurements = [];
    
    for i = 1:length(fields)
        filename = ['synthetic_measurements/' fields{i} '.csv'];
        dlmwrite(filename, measurements.(fields{i}), '-append');
        % Consolidate measurements for ML model input
        all_measurements = [all_measurements, measurements.(fields{i})];
    end

    % Save field names (headers) if not already saved
    headers_file = 'io_for_ml_model/field_names.csv';
    if ~exist(headers_file, 'file')
        fid = fopen(headers_file, 'w');
        fprintf(fid, '%s\n', strjoin(fields, ','));
        fclose(fid);
    end

    % Combine current true voltage values for ML model output
    true_volt_combined = [true_voltages.magnitude, true_voltages.angle];

    % Append inputs and outputs for ML model
    inputs_file = 'io_for_ml_model/inputs.csv';
    outputs_file = 'io_for_ml_model/outputs.csv';

    % Generate masked measurements for inputs
    masked_measurements = generate_masked_measurements(all_measurements);
    dlmwrite(inputs_file, masked_measurements, '-append');
    dlmwrite(outputs_file, true_volt_combined, '-append');
end
