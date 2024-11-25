% main.m
% Main script to run power flow analysis and generate synthetic measurements

% Load the MATPOWER case and set options
mpc = loadcase('case14');
mpopt = mpoption('out.all', 0, 'verbose', 0);

% Configuration parameters
config.num_power_variations = 1000;
config.num_noise_variations = 10;
config.noise_mu = 0;     % Mean of noise
config.noise_sigma = 0.01; % Standard deviation of noise

% Initialize measurement arrays
measurements = initialize_measurements();

% Initialize arrays for true voltage values
true_voltages.magnitude = [];
true_voltages.angle = [];

% Run power flow analysis
successful_count = vary_run_save(mpc, mpopt, config, measurements, true_voltages);

disp(['Power flow analysis completed successfully for ' num2str(successful_count) ' load variations.']);