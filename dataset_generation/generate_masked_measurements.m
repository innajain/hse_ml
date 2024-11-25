function masked_measurements = generate_masked_measurements(all_measurements)
    % Initialize a masked matrix for selected measurements
    masked_measurements = all_measurements;
    num_rows = size(all_measurements, 1);
    num_columns = size(all_measurements, 2);
    
    % Process each row in all_measurements
    for i = 1:num_rows
        % Determine a random number of columns to mask for this row
        n = randi([1, num_columns]);
        
        % Randomly select n columns to mask with NaN
        masked_columns = randperm(num_columns, n);
        
        % Mask the selected columns in the current row
        masked_measurements(i, masked_columns) = NaN;
    end
end
