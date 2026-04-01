clearvars; close all; clc;
%% =========================================================================
%  PUCCH Format 0 - Multi-User Dataset Generation
%  =========================================================================
%  This script generates datasets for multi-user PUCCH Format 0 scenarios.
%
%  In 5G NR, multiple UEs can be multiplexed on the same PUCCH Format 0
%  resources using different Initial Cyclic Shifts (m0).
%
%  For 1 HARQ + 1 SR, each user occupies 4 cyclic shifts:
%    user_shifts = mod(m0 + [0, 3, 6, 9], 12)
%
%  To avoid overlap, m0 values must NOT differ by multiples of 3.
%  Maximum 3 users for 1 HARQ + 1 SR (12 shifts / 4 per user = 3).
%
%  Correct m0 allocation:
%    User 1: m0=0 -> shifts {0, 3, 6, 9}
%    User 2: m0=1 -> shifts {1, 4, 7, 10}
%    User 3: m0=2 -> shifts {2, 5, 8, 11}
%
%  Physical model:
%    - Each user transmits through an INDEPENDENT fading channel
%    - Base station receives the SUM of all signals + AWGN
%    - We decode the TARGET USER (m0=0) in presence of interference
%
%  Scenarios:
%    1. 2 users: m0 = {0, 1}     (moderate interference)
%    2. 3 users: m0 = {0, 1, 2}  (maximum loading)
%% =========================================================================

%% =========================================================================
%  1. CONFIGURATION PARAMETERS
%% =========================================================================

fprintf('========================================\n');
fprintf(' PUCCH Format 0 - Multi-User Generator\n');
fprintf('========================================\n\n');

% --- Data Generation Parameters ---
numSamplesPerSNR = 200000;
trainSNR = 10;
testSNRs = [0 5 10 15 20];
allSNRs = unique([trainSNR testSNRs]);

% --- Antenna Configuration ---
NTxAnts = 1;
NRxAnts = 1;

% --- Slot Configuration ---
allowedSlots = [13, 14];

% --- Carrier Settings ---
carrier = nrCarrierConfig;
carrier.NCellID = 2;
carrier.SubcarrierSpacing = 15;
carrier.NSizeGrid = 25;

% --- Waveform Info ---
waveformInfo = nrOFDMInfo(carrier);
nFFT = waveformInfo.Nfft;
symbolsPerSlot = carrier.SymbolsPerSlot;

%% =========================================================================
%  2. DEFINE MULTI-USER SCENARIOS
%% =========================================================================

scenarios = struct();

% Scenario 1: 2 Users
% User 1 (m0=0, TARGET): shifts {0, 3, 6, 9}
% User 2 (m0=1, INTERFERER): shifts {1, 4, 7, 10}
% No overlap -> orthogonal
scenarios(1).name = '2users';
scenarios(1).description = '2 Users (m0=0, m0=1)';
scenarios(1).num_users = 2;
scenarios(1).m0_values = [0, 1];
scenarios(1).target_user_index = 1;
scenarios(1).target_m0 = 0;

% Scenario 2: 3 Users (Maximum loading for 1 HARQ + 1 SR)
% User 1 (m0=0, TARGET): shifts {0, 3, 6, 9}
% User 2 (m0=1, INTERFERER 1): shifts {1, 4, 7, 10}
% User 3 (m0=2, INTERFERER 2): shifts {2, 5, 8, 11}
% No overlap -> orthogonal, ALL 12 shifts used
scenarios(2).name = '3users';
scenarios(2).description = '3 Users (m0=0, m0=1, m0=2)';
scenarios(2).num_users = 3;
scenarios(2).m0_values = [0, 1, 2];
scenarios(2).target_user_index = 1;
scenarios(2).target_m0 = 0;

%% =========================================================================
%  3. DEFINE UCI COMBINATIONS
%% =========================================================================

uciCombinations = {
    {0, 0};   % Class 0: NACK, -ve SR
    {0, 1};   % Class 1: NACK, +ve SR
    {1, 0};   % Class 2: ACK, -ve SR
    {1, 1};   % Class 3: ACK, +ve SR
};
numClasses = length(uciCombinations);
samplesPerClass = numSamplesPerSNR / numClasses;

assert(mod(numSamplesPerSNR, numClasses) == 0, ...
    'numSamplesPerSNR must be divisible by numClasses');

%% =========================================================================
%  4. VERIFY CYCLIC SHIFT ORTHOGONALITY
%% =========================================================================

fprintf('--- Verifying Cyclic Shift Orthogonality ---\n\n');

for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    fprintf('Scenario: %s\n', scenario.description);
    
    allShiftsUsed = [];
    for userIdx = 1:scenario.num_users
        m0 = scenario.m0_values(userIdx);
        userShifts = mod(m0 + [0, 3, 6, 9], 12);
        fprintf('  User %d (m0=%d): cyclic shifts = {%s}\n', ...
            userIdx, m0, num2str(userShifts));
        allShiftsUsed = [allShiftsUsed, userShifts];
    end
    
    uniqueShifts = unique(allShiftsUsed);
    if length(uniqueShifts) == length(allShiftsUsed)
        fprintf('  Orthogonality check: PASS (no collisions)\n\n');
    else
        [counts, values] = hist(allShiftsUsed, 0:11);
        collisions = values(counts > 1);
        fprintf('  Orthogonality check: FAIL! Collisions at shifts: %s\n\n', ...
            num2str(collisions));
        error('Cyclic shift collision detected! Choose different m0 values.');
    end
end

%% =========================================================================
%  5. PREPARE CSV HEADER
%% =========================================================================

headerNames = cell(1, 25);
for h = 1:12
    headerNames{h} = sprintf('real_%d', h);
    headerNames{h+12} = sprintf('imag_%d', h);
end
headerNames{25} = 'label';
headerLine = strjoin(headerNames, ',');

%% =========================================================================
%  6. VERIFY TARGET USER DECODING (without interference or noise)
%% =========================================================================

fprintf('--- Verifying Target User Decoding ---\n\n');

carrier.NSlot = allowedSlots(1);
ouciVerify = [1, 1];

for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    fprintf('Scenario: %s\n', scenario.description);
    
    pucchVerify = nrPUCCH0Config;
    pucchVerify.PRBSet = 0;
    pucchVerify.SymbolAllocation = [13 1];
    pucchVerify.InitialCyclicShift = scenario.target_m0;
    pucchVerify.FrequencyHopping = 'neither';
    
    verifyPassed = true;
    
    for c = 1:numClasses
        ackVal = uciCombinations{c}{1};
        srVal = uciCombinations{c}{2};
        
        txSig = mynrPUCCH0(carrier, pucchVerify, {ackVal, srVal});
        [decBits, ~, detMet] = mynrPUCCHDecode(carrier, pucchVerify, ouciVerify, txSig);
        
        decodedACK = decBits{1};
        decodedSR = decBits{2};
        
        if ~isempty(decodedACK) && ~isempty(decodedSR) && ...
                decodedACK == ackVal && decodedSR == srVal
            status = 'PASS';
        else
            status = 'FAIL';
            verifyPassed = false;
        end
        
        fprintf('  Class %d: TX(ACK=%d,SR=%d) -> RX(ACK=%d,SR=%d) [%s] detMet=%.4f\n', ...
            c-1, ackVal, srVal, decodedACK, decodedSR, status, detMet);
    end
    
    if verifyPassed
        fprintf('  >> Verification PASSED\n\n');
    else
        error('Verification FAILED for scenario: %s', scenario.description);
    end
end

%% =========================================================================
%  7. PRINT CONFIGURATION SUMMARY
%% =========================================================================

fprintf('--- Configuration Summary ---\n');
fprintf('SNR values: ');
fprintf('%d ', allSNRs);
fprintf('dB\n');
fprintf('Training SNR: %d dB\n', trainSNR);
fprintf('Samples per SNR: %d (%d per class)\n', numSamplesPerSNR, samplesPerClass);
fprintf('Classes: %d\n', numClasses);
fprintf('Antennas: %d TX per user, %d RX at BS\n', NTxAnts, NRxAnts);
fprintf('Channel: TDL-C, 300ns delay spread, 100Hz Doppler\n');
fprintf('Independent channel per user: Yes\n');
fprintf('Freq Hopping: neither\n');
fprintf('\nScenarios to generate:\n');
for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    fprintf('  %d. %s (m0=[%s], target=m0=%d)\n', ...
        scenIdx, scenario.description, num2str(scenario.m0_values), scenario.target_m0);
end
fprintf('========================================\n\n');

%% =========================================================================
%  8. CREATE CHANNEL OBJECTS (once, reused across SNR values)
%% =========================================================================

fprintf('Creating channel objects...\n');

maxUsers = max([scenarios.num_users]);

channelPool = cell(1, maxUsers);
for userIdx = 1:maxUsers
    channelPool{userIdx} = nrTDLChannel;
    channelPool{userIdx}.DelayProfile = 'TDL-C';
    channelPool{userIdx}.DelaySpread = 300e-9;
    channelPool{userIdx}.MaximumDopplerShift = 100;
    channelPool{userIdx}.MIMOCorrelation = 'Low';
    channelPool{userIdx}.TransmissionDirection = 'Uplink';
    channelPool{userIdx}.NumTransmitAntennas = NTxAnts;
    channelPool{userIdx}.NumReceiveAntennas = NRxAnts;
    channelPool{userIdx}.NormalizeChannelOutputs = false;
    channelPool{userIdx}.SampleRate = waveformInfo.SampleRate;
    channelPool{userIdx}.InitialTime = (userIdx - 1) * 0.5;
    
    fprintf('  Channel %d created (InitialTime=%.1fs)\n', ...
        userIdx, (userIdx-1)*0.5);
end

chInfo = info(channelPool{1});

fprintf('Channel objects created.\n\n');

%% =========================================================================
%  9. PRE-COMPUTE WAVEFORM LENGTH (once, same for all samples)
%% =========================================================================

dummyGrid = nrResourceGrid(carrier, NTxAnts);
dummyWaveform = nrOFDMModulate(carrier, dummyGrid);
expectedWaveformLen = length(dummyWaveform) + chInfo.MaximumChannelDelay;

fprintf('Expected waveform length: %d samples\n\n', expectedWaveformLen);

%% =========================================================================
%  10. MAIN DATA GENERATION LOOP
%% =========================================================================

totalStartTime = tic;

for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    scenarioStartTime = tic;
    
    fprintf('\n');
    fprintf('################################################################\n');
    fprintf('# SCENARIO %d: %s\n', scenIdx, scenario.description);
    fprintf('# Users: %d, m0=[%s], Target: m0=%d\n', ...
        scenario.num_users, num2str(scenario.m0_values), scenario.target_m0);
    fprintf('################################################################\n\n');
    
    % Create PUCCH config for EACH user in this scenario
    pucchConfigs = cell(1, scenario.num_users);
    for userIdx = 1:scenario.num_users
        pucchConfigs{userIdx} = nrPUCCH0Config;
        pucchConfigs{userIdx}.PRBSet = 0;
        pucchConfigs{userIdx}.SymbolAllocation = [13 1];
        pucchConfigs{userIdx}.InitialCyclicShift = scenario.m0_values(userIdx);
        pucchConfigs{userIdx}.FrequencyHopping = 'neither';
    end
    
    % Target user config (for receiver-side extraction)
    pucchTarget = pucchConfigs{scenario.target_user_index};
    
    % Pre-compute target user PUCCH indices (same for all samples)
    carrier.NSlot = allowedSlots(1);
    [targetIndicesTemplate, ~] = nrPUCCHIndices(carrier, pucchTarget);
    
    for snrIdx = 1:length(allSNRs)
        SNRdB = allSNRs(snrIdx);
        snrStartTime = tic;
        
        if SNRdB == trainSNR
            fprintf('>>> SNR = %d dB [TRAINING SET] <<<\n', SNRdB);
        else
            fprintf('--- SNR = %d dB [TEST SET] ---\n', SNRdB);
        end
        
        % Pre-allocate storage
        rxSamples = zeros(numSamplesPerSNR, 24);
        labels = zeros(numSamplesPerSNR, 1);
        
        % Reset random number generator for reproducibility
        rng('default');
        
        % Reset ALL channels for this SNR point
        for userIdx = 1:scenario.num_users
            reset(channelPool{userIdx});
        end
        
        % --- Generate samples ---
        for i = 1:numSamplesPerSNR
            
            % Round-robin class selection for TARGET user (balanced)
            classIdx = mod(i-1, numClasses) + 1;
            targetAck = uciCombinations{classIdx}{1};
            targetSr = uciCombinations{classIdx}{2};
            
            % Select slot from allowed slots
            carrier.NSlot = allowedSlots(randi(length(allowedSlots)));
            
            % =============================================================
            % TRANSMITTER + CHANNEL: Process each user independently
            % =============================================================
            
            % Initialize total received signal at base station
            totalRxWaveform = zeros(expectedWaveformLen, NRxAnts);
            
            for userIdx = 1:scenario.num_users
                
                % Determine UCI for this user
                if userIdx == scenario.target_user_index
                    % Target user: use the selected class
                    userUci = {targetAck, targetSr};
                else
                    % Interfering user: random UCI (independent each sample)
                    intfClassIdx = randi(numClasses);
                    userUci = {uciCombinations{intfClassIdx}{1}, ...
                               uciCombinations{intfClassIdx}{2}};
                end
                
                % Generate PUCCH symbols for this user
                userSymbols = mynrPUCCH0(carrier, pucchConfigs{userIdx}, userUci);
                
                % Create resource grid for this user (separate grid)
                userGrid = nrResourceGrid(carrier, NTxAnts);
                [userIndices, ~] = nrPUCCHIndices(carrier, pucchConfigs{userIdx});
                userGrid(userIndices) = userSymbols;
                
                % OFDM modulation
                userTxWaveform = nrOFDMModulate(carrier, userGrid);
                
                % Add delay padding for channel
                userTxPadded = [userTxWaveform; ...
                    zeros(chInfo.MaximumChannelDelay, size(userTxWaveform, 2))];
                
                % Pass through INDEPENDENT fading channel
                [userRxWaveform, ~, ~] = channelPool{userIdx}(userTxPadded);
                
                % Add to total received signal at base station
                rxLen = min(expectedWaveformLen, length(userRxWaveform));
                totalRxWaveform(1:rxLen, :) = totalRxWaveform(1:rxLen, :) + ...
                                               userRxWaveform(1:rxLen, :);
            end
            
            % =============================================================
            % ADD AWGN NOISE (on the combined received signal)
            % =============================================================
            
            SNR = 10^(SNRdB / 20);
            N0 = 1 / (sqrt(2.0 * NRxAnts * nFFT) * SNR);
            noise = N0 * complex(randn(size(totalRxWaveform)), ...
                                  randn(size(totalRxWaveform)));
            totalRxWaveform = totalRxWaveform + noise;
            
            % =============================================================
            % RECEIVER: Decode target user
            % =============================================================
            
            % OFDM demodulation
            rxGrid = nrOFDMDemodulate(carrier, totalRxWaveform);
            [K, L, R] = size(rxGrid);
            if L < symbolsPerSlot
                rxGrid = cat(2, rxGrid, zeros(K, symbolsPerSlot - L, R));
            end
            
            % Extract PUCCH samples for TARGET user
            [targetIndices, ~] = nrPUCCHIndices(carrier, pucchTarget);
            [pucchRx, ~] = nrExtractResources(targetIndices, rxGrid);
            
            % Store sample (split complex to real)
            realPart = real(pucchRx(:))';
            imagPart = imag(pucchRx(:))';
            rxSamples(i, :) = [realPart, imagPart];
            labels(i) = classIdx - 1;  % 0-indexed
            
            % Progress display
            if mod(i, 50000) == 0
                elapsed = toc(snrStartTime);
                remaining = elapsed / i * (numSamplesPerSNR - i);
                fprintf('  %d/%d samples (%.0fs elapsed, ~%.0fs remaining)\n', ...
                    i, numSamplesPerSNR, elapsed, remaining);
            end
        end
        
        % =============================================================
        % POST-PROCESSING
        % =============================================================
        
        % Shuffle data
        shuffleIdx = randperm(numSamplesPerSNR);
        rxSamples = rxSamples(shuffleIdx, :);
        labels = labels(shuffleIdx);
        
        % Verify class balance
        fprintf('  Class distribution: ');
        for c = 0:numClasses-1
            count = sum(labels == c);
            fprintf('C%d=%d ', c, count);
        end
        fprintf('\n');
        
        % Validate data quality
        numZeroRows = sum(all(rxSamples == 0, 2));
        numNanRows = sum(any(isnan(rxSamples), 2));
        fprintf('  Validation: zero_rows=%d, NaN_rows=%d\n', numZeroRows, numNanRows);
        
        if numNanRows > 0
            warning('NaN values found at SNR=%d dB, Scenario=%s!', ...
                SNRdB, scenario.name);
        end
        
        % =============================================================
        % SAVE DATASET
        % =============================================================
        
        % Save .mat file
        filename_mat = sprintf('pucch_f0_%s_SNR_%ddB.mat', scenario.name, SNRdB);
        save(filename_mat, 'rxSamples', 'labels', 'SNRdB');
        
        % Save .csv file (Python compatible)
        filename_csv = sprintf('pucch_f0_%s_SNR_%ddB.csv', scenario.name, SNRdB);
        dataWithLabels = [rxSamples, labels];
        
        fid = fopen(filename_csv, 'w');
        fprintf(fid, '%s\n', headerLine);
        fclose(fid);
        
        try
            writematrix(dataWithLabels, filename_csv, 'WriteMode', 'append');
        catch
            dlmwrite(filename_csv, dataWithLabels, '-append', ...
                     'precision', '%.10f', 'delimiter', ',');
        end
        
        elapsed = toc(snrStartTime);
        fprintf('  Saved: %s (%.1f min)\n\n', filename_csv, elapsed/60);
    end
    
    scenarioTime = toc(scenarioStartTime);
    fprintf('################################################################\n');
    fprintf('# SCENARIO %d COMPLETE: %s\n', scenIdx, scenario.description);
    fprintf('# Time: %.1f minutes\n', scenarioTime/60);
    fprintf('################################################################\n\n');
end

%% =========================================================================
%  11. SAVE EXPERIMENT CONFIGURATION
%% =========================================================================

multiuser_config = struct();
multiuser_config.numSamplesPerSNR = numSamplesPerSNR;
multiuser_config.trainSNR = trainSNR;
multiuser_config.testSNRs = testSNRs;
multiuser_config.allSNRs = allSNRs;
multiuser_config.NTxAnts = NTxAnts;
multiuser_config.NRxAnts = NRxAnts;
multiuser_config.allowedSlots = allowedSlots;
multiuser_config.NCellID = carrier.NCellID;
multiuser_config.SubcarrierSpacing = carrier.SubcarrierSpacing;
multiuser_config.NSizeGrid = carrier.NSizeGrid;
multiuser_config.numClasses = numClasses;
multiuser_config.samplesPerClass = samplesPerClass;
multiuser_config.numScenarios = length(scenarios);
multiuser_config.channelModel = 'TDL-C';
multiuser_config.delaySpread = 300e-9;
multiuser_config.maxDoppler = 100;
multiuser_config.independentChannelsPerUser = true;
multiuser_config.dateGenerated = datestr(now);

for scenIdx = 1:length(scenarios)
    fieldName = sprintf('scenario_%d', scenIdx);
    multiuser_config.(fieldName) = scenarios(scenIdx);
end

save('multiuser_experiment_config.mat', 'multiuser_config');

%% =========================================================================
%  12. FINAL SUMMARY
%% =========================================================================

totalTime = toc(totalStartTime);

fprintf('\n========================================\n');
fprintf('  MULTI-USER GENERATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total time: %.1f minutes (%.1f hours)\n', totalTime/60, totalTime/3600);

fprintf('\nGenerated files:\n');
for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    fprintf('\n  %s:\n', scenario.description);
    for snrIdx = 1:length(allSNRs)
        snr = allSNRs(snrIdx);
        tag = '';
        if snr == trainSNR; tag = '  <-- TRAINING'; end
        fprintf('    pucch_f0_%s_SNR_%ddB.csv%s\n', scenario.name, snr, tag);
    end
end

fprintf('\n  Config: multiuser_experiment_config.mat\n');

fprintf('\n--- Physical Setup ---\n');
for scenIdx = 1:length(scenarios)
    scenario = scenarios(scenIdx);
    fprintf('\n  %s:\n', scenario.description);
    fprintf('    Number of users: %d\n', scenario.num_users);
    fprintf('    m0 values: [%s]\n', num2str(scenario.m0_values));
    fprintf('    Target user: m0=%d (User %d)\n', ...
        scenario.target_m0, scenario.target_user_index);
    fprintf('    Independent channels per user: Yes\n');
    fprintf('    Channel model: TDL-C (300ns, 100Hz)\n');
    
    fprintf('    Cyclic shift allocation:\n');
    for userIdx = 1:scenario.num_users
        m0 = scenario.m0_values(userIdx);
        shifts = mod(m0 + [0, 3, 6, 9], 12);
        role = 'Interferer';
        if userIdx == scenario.target_user_index
            role = 'TARGET';
        end
        fprintf('      User %d (m0=%d, %s): shifts={%s}\n', ...
            userIdx, m0, role, num2str(shifts));
    end
end

fprintf('\n--- Next Steps ---\n');
fprintf('1. Copy CSV files to Python project directory\n');
fprintf('2. Run main_multiuser.py for training and evaluation\n');
fprintf('3. Compare with single-user baseline results\n');
fprintf('========================================\n');