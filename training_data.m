clearvars; close all; clc;
%% =========================================================================
%  PUCCH Format 0 - Dataset Generation for ML Training & Testing
%  Paper: "ML Decoder for 5G NR PUCCH Format 0"
%  Case: 1 HARQ + 1 SR = 4 classes (mcs = 0, 3, 6, 9)
%  SNR Range: 0 to 20 dB, step = 5 dB
%% =========================================================================

%% 1. Configuration Parameters
% -------------------------------------------------------------------------
numSamplesPerSNR = 200000;
trainSNR = 10;
testSNRs = [0 5 10 15 20];
allSNRs = unique([trainSNR testSNRs]);

NTxAnts = 1;
NRxAnts = 1;

% Slot constraint as per paper: "slots 13 and 14"
% Note: With SCS=15kHz, SlotsPerFrame=10, so internally:
%   NSlot=13 -> mod(13,10)=3, NSlot=14 -> mod(14,10)=4
% This gives specific ncs values that are consistent for training/testing
allowedSlots = [13, 14];

% --- Carrier Settings ---
carrier = nrCarrierConfig;
carrier.NCellID = 2;
carrier.SubcarrierSpacing = 15;
carrier.NSizeGrid = 25;

% --- PUCCH Format 0 Settings ---
pucch = nrPUCCH0Config;
pucch.PRBSet = 0;
pucch.SymbolAllocation = [13 1];     % Symbol 13, 1 symbol
pucch.InitialCyclicShift = 0;        % m0 = 0
pucch.FrequencyHopping = 'neither';  % Disabled as per paper

% --- Channel Settings ---
channel = nrTDLChannel;
channel.DelayProfile = 'TDL-C';
channel.DelaySpread = 300e-9;
channel.MaximumDopplerShift = 100;
channel.MIMOCorrelation = 'Low';
channel.TransmissionDirection = 'Uplink';
channel.NumTransmitAntennas = NTxAnts;
channel.NumReceiveAntennas = NRxAnts;
channel.NormalizeChannelOutputs = false;

% --- Waveform Info ---
waveformInfo = nrOFDMInfo(carrier);
channel.SampleRate = waveformInfo.SampleRate;
nFFT = waveformInfo.Nfft;
symbolsPerSlot = carrier.SymbolsPerSlot;
chInfo = info(channel);

%% 2. Define UCI Combinations (4 Classes)
% -------------------------------------------------------------------------
uciCombinations = {
    {0, 0};   % Class 0: NACK, -ve SR
    {0, 1};   % Class 1: NACK, +ve SR
    {1, 0};   % Class 2: ACK,  -ve SR
    {1, 1};   % Class 3: ACK,  +ve SR
};

numClasses = length(uciCombinations);
samplesPerClass = numSamplesPerSNR / numClasses;

% Verify divisibility
assert(mod(numSamplesPerSNR, numClasses) == 0, ...
    'numSamplesPerSNR (%d) must be divisible by numClasses (%d)', ...
    numSamplesPerSNR, numClasses);

%% 3. Verify UCI Encoding/Decoding (Round-Trip Test)
% -------------------------------------------------------------------------
fprintf('========================================\n');
fprintf(' PUCCH Format 0 - ML Dataset Generator\n');
fprintf('========================================\n');
fprintf('\nRound-Trip Verification (TX -> Decode without noise):\n');

carrier.NSlot = allowedSlots(1);
ouciVerify = [1, 1];  % Expect 1 ACK bit + 1 SR bit
verificationPassed = true;

for c = 1:numClasses
    ackVal = uciCombinations{c}{1};
    srVal = uciCombinations{c}{2};
    
    % Generate PUCCH signal (no channel, no noise)
    txSig = mynrPUCCH0(carrier, pucch, {ackVal, srVal});
    
    % Decode directly (perfect conditions)
    [decBits, ~, detMet] = mynrPUCCHDecode(carrier, pucch, ouciVerify, txSig);
    
    decodedACK = decBits{1};
    decodedSR = decBits{2};
    
    % Check if decoded matches transmitted
    if ~isempty(decodedACK) && ~isempty(decodedSR)
        ackMatch = (decodedACK == ackVal);
        srMatch = (decodedSR == srVal);
        
        if ackMatch && srMatch
            status = 'PASS';
        else
            status = 'FAIL';
            verificationPassed = false;
        end
    else
        status = 'FAIL (empty output)';
        verificationPassed = false;
    end
    
    fprintf('  Class %d: TX(ACK=%d,SR=%d) -> RX(ACK=%d,SR=%d) [%s] detMet=%.4f\n', ...
        c-1, ackVal, srVal, decodedACK, decodedSR, status, detMet);
end

if verificationPassed
    fprintf('  >> All classes verified successfully!\n');
else
    error('Verification FAILED! Check mynrPUCCH0 and mynrPUCCHDecode functions.');
end

%% 4. Print Configuration Summary
% -------------------------------------------------------------------------
fprintf('\n--- Configuration ---\n');
fprintf('SNR values: ');
fprintf('%d ', allSNRs);
fprintf('dB\n');
fprintf('Training SNR: %d dB\n', trainSNR);
fprintf('Samples per SNR: %d (%d per class)\n', numSamplesPerSNR, samplesPerClass);
fprintf('Classes: %d\n', numClasses);
fprintf('Antennas: %d TX, %d RX\n', NTxAnts, NRxAnts);
fprintf('Allowed slots: %d, %d (map to %d, %d in frame)\n', ...
    allowedSlots(1), allowedSlots(2), ...
    mod(allowedSlots(1), carrier.SlotsPerFrame), ...
    mod(allowedSlots(2), carrier.SlotsPerFrame));
fprintf('Channel: %s, Doppler: %d Hz\n', channel.DelayProfile, channel.MaximumDopplerShift);
fprintf('Freq Hopping: %s\n', pucch.FrequencyHopping);
fprintf('========================================\n\n');

%% 5. Prepare CSV Header
% -------------------------------------------------------------------------
headerNames = cell(1, 25);
for h = 1:12
    headerNames{h} = sprintf('real_%d', h);
    headerNames{h+12} = sprintf('imag_%d', h);
end
headerNames{25} = 'label';
headerLine = strjoin(headerNames, ',');

%% 6. Generate Dataset for Each SNR
% -------------------------------------------------------------------------
totalStartTime = tic;

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
    
    % Reset for reproducibility
    rng('default');
    reset(channel);
    
    % --- ROUND-ROBIN CLASS GENERATION ---
    for i = 1:numSamplesPerSNR
        
        % Round-robin class selection (balanced + interleaved)
        classIdx = mod(i-1, numClasses) + 1;  % 1,2,3,4,1,2,3,4,...
        
        ackBit = uciCombinations{classIdx}{1};
        srBit = uciCombinations{classIdx}{2};
        uci = {ackBit, srBit};
        
        % Select slot from allowed slots
        carrier.NSlot = allowedSlots(randi(length(allowedSlots)));
        
        % --- Transmitter ---
        pucchSymbols = mynrPUCCH0(carrier, pucch, uci);
        pucchGrid = nrResourceGrid(carrier, NTxAnts);
        [pucchIndices, ~] = nrPUCCHIndices(carrier, pucch);
        pucchGrid(pucchIndices) = pucchSymbols;
        txWaveform = nrOFDMModulate(carrier, pucchGrid);
        txWaveformChDelay = [txWaveform; ...
            zeros(chInfo.MaximumChannelDelay, size(txWaveform, 2))];
        
        % --- Channel ---
        [rxWaveform, ~, ~] = channel(txWaveformChDelay);
        
        % --- Add AWGN Noise ---
        SNR = 10^(SNRdB / 20);
        N0 = 1 / (sqrt(2.0 * NRxAnts * nFFT) * SNR);
        noise = N0 * complex(randn(size(rxWaveform)), ...
                              randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;
        
        % --- Receiver ---
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        [K, L, R] = size(rxGrid);
        if L < symbolsPerSlot
            rxGrid = cat(2, rxGrid, zeros(K, symbolsPerSlot - L, R));
        end
        
        [pucchIndices, ~] = nrPUCCHIndices(carrier, pucch);
        [pucchRx, ~] = nrExtractResources(pucchIndices, rxGrid);
        
        % --- Store sample ---
        realPart = real(pucchRx(:))';   % 1x12
        imagPart = imag(pucchRx(:))';   % 1x12
        rxSamples(i, :) = [realPart, imagPart];  % 1x24
        labels(i) = classIdx - 1;       % 0-indexed: 0,1,2,3
        
        % Progress display
        if mod(i, 50000) == 0
            elapsed = toc(snrStartTime);
            remaining = elapsed / i * (numSamplesPerSNR - i);
            fprintf('  %d/%d samples (%.0fs elapsed, ~%.0fs remaining)\n', ...
                i, numSamplesPerSNR, elapsed, remaining);
        end
    end
    
    % --- Shuffle the data ---
    shuffleIdx = randperm(numSamplesPerSNR);
    rxSamples = rxSamples(shuffleIdx, :);
    labels = labels(shuffleIdx);
    
    %% 7. Verify Class Balance
    fprintf('  Class distribution: ');
    for c = 0:numClasses-1
        count = sum(labels == c);
        fprintf('C%d=%d ', c, count);
    end
    fprintf('\n');
    
    %% 8. Data Validation
    numZeroRows = sum(all(rxSamples == 0, 2));
    numNanRows = sum(any(isnan(rxSamples), 2));
    fprintf('  Validation: zero_rows=%d, NaN_rows=%d\n', numZeroRows, numNanRows);
    
    if numNanRows > 0
        warning('Found NaN values at SNR=%d dB!', SNRdB);
    end
    
    %% 9. Save Dataset
    % Save .mat file
    filename_mat = sprintf('pucch_f0_dataset_SNR_%ddB.mat', SNRdB);
    save(filename_mat, 'rxSamples', 'labels', 'SNRdB');
    
    % Save .csv file for Python
    filename_csv = sprintf('pucch_f0_dataset_SNR_%ddB.csv', SNRdB);
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

%% 10. Generate DTX Dataset
% -------------------------------------------------------------------------
fprintf('=== Generating DTX Data (No Transmission) ===\n');

for snrIdx = 1:length(allSNRs)
    SNRdB = allSNRs(snrIdx);
    dtxStartTime = tic;
    fprintf('  DTX at SNR = %d dB ... ', SNRdB);
    
    numDTXSamples = 50000;
    rxSamplesDTX = zeros(numDTXSamples, 24);
    labelsDTX = ones(numDTXSamples, 1) * 4;
    
    rng(42);
    reset(channel);
    
    for i = 1:numDTXSamples
        carrier.NSlot = allowedSlots(randi(length(allowedSlots)));
        
        pucchGrid = nrResourceGrid(carrier, NTxAnts);
        txWaveform = nrOFDMModulate(carrier, pucchGrid);
        txWaveformChDelay = [txWaveform; ...
            zeros(chInfo.MaximumChannelDelay, size(txWaveform, 2))];
        
        [rxWaveform, ~, ~] = channel(txWaveformChDelay);
        
        SNR = 10^(SNRdB / 20);
        N0 = 1 / (sqrt(2.0 * NRxAnts * nFFT) * SNR);
        noise = N0 * complex(randn(size(rxWaveform)), ...
                              randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;
        
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        [K, L, R] = size(rxGrid);
        if L < symbolsPerSlot
            rxGrid = cat(2, rxGrid, zeros(K, symbolsPerSlot - L, R));
        end
        
        [pucchIndices, ~] = nrPUCCHIndices(carrier, pucch);
        [pucchRx, ~] = nrExtractResources(pucchIndices, rxGrid);
        
        realPart = real(pucchRx(:))';
        imagPart = imag(pucchRx(:))';
        rxSamplesDTX(i, :) = [realPart, imagPart];
    end
    
    filename_dtx = sprintf('pucch_f0_DTX_dataset_SNR_%ddB.csv', SNRdB);
    dataWithLabelsDTX = [rxSamplesDTX, labelsDTX];
    
    fid = fopen(filename_dtx, 'w');
    fprintf(fid, '%s\n', headerLine);
    fclose(fid);
    
    try
        writematrix(dataWithLabelsDTX, filename_dtx, 'WriteMode', 'append');
    catch
        dlmwrite(filename_dtx, dataWithLabelsDTX, '-append', ...
                 'precision', '%.10f', 'delimiter', ',');
    end
    
    elapsed = toc(dtxStartTime);
    fprintf('Done (%.1f sec)\n', elapsed);
end

%% 11. Save Experiment Configuration
% -------------------------------------------------------------------------
config = struct();
config.numSamplesPerSNR = numSamplesPerSNR;
config.trainSNR = trainSNR;
config.testSNRs = testSNRs;
config.allSNRs = allSNRs;
config.NTxAnts = NTxAnts;
config.NRxAnts = NRxAnts;
config.allowedSlots = allowedSlots;
config.NCellID = carrier.NCellID;
config.SubcarrierSpacing = carrier.SubcarrierSpacing;
config.NSizeGrid = carrier.NSizeGrid;
config.PRBSet = pucch.PRBSet;
config.SymbolAllocation = pucch.SymbolAllocation;
config.InitialCyclicShift = pucch.InitialCyclicShift;
config.FrequencyHopping = pucch.FrequencyHopping;
config.DelayProfile = channel.DelayProfile;
config.DelaySpread = channel.DelaySpread;
config.MaximumDopplerShift = channel.MaximumDopplerShift;
config.numClasses = numClasses;
config.samplesPerClass = samplesPerClass;
config.generationMethod = 'round-robin interleaved then shuffled';
config.dateGenerated = datestr(now);

save('experiment_config.mat', 'config');

%% 12. Final Summary
% -------------------------------------------------------------------------
totalTime = toc(totalStartTime);
fprintf('\n========================================\n');
fprintf('  DATASET GENERATION COMPLETE!\n');
fprintf('========================================\n');
fprintf('Total time: %.1f minutes\n', totalTime/60);

fprintf('\n  UCI Data (4 classes, %d samples each):\n', numSamplesPerSNR);
for snrIdx = 1:length(allSNRs)
    snr = allSNRs(snrIdx);
    tag = '';
    if snr == trainSNR; tag = '  <-- TRAINING'; end
    fprintf('    pucch_f0_dataset_SNR_%ddB.csv%s\n', snr, tag);
end

fprintf('\n  DTX Data (class 4, %d samples each):\n', 50000);
for snrIdx = 1:length(allSNRs)
    fprintf('    pucch_f0_DTX_dataset_SNR_%ddB.csv\n', allSNRs(snrIdx));
end

fprintf('\n  Config: experiment_config.mat\n');
fprintf('\n--- Next Step ---\n');
fprintf('Use pucch_f0_dataset_SNR_10dB.csv for training (75/25 split)\n');
fprintf('Use other SNR files for testing\n');
fprintf('========================================\n');