function measurements = initialize_measurements()
    measurements.p_injections = [];
    measurements.q_injections = [];
    measurements.from_bus_p_injection = [];
    measurements.from_bus_q_injection = [];
    measurements.to_bus_p_injection = [];
    measurements.to_bus_q_injection = [];
    measurements.voltage_mag = [];
    measurements.voltage_angles_deg = [];
    measurements.from_current_magnitudes = [];
    measurements.from_current_angles_deg = [];
    measurements.to_current_magnitudes = [];
    measurements.to_current_angles_deg = [];
end