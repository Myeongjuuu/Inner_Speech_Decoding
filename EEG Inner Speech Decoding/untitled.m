% EEGLAB 실행
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% base path 설정 
base_path = 'D:/Users/windows/Desktop/EEG Inner Speech Decoding/Dataset';

% subject 번호와 session 번호 설정
subject = 'sub-01';
session = 'ses-01';

% EEG 파일 경로
% .bdf 파일 경로: base_path/sub-01/ses-01/eeg/sub-01_ses-01_task-innerspeech_eeg.bdf
file_path = fullfile(base_path, subject, session, 'eeg');
file_name = [subject '_' session '_task-innerspeech_eeg.bdf'];

% EEG data 불러오기
EEG = pop_biosig(fullfile(file_path, file_name));
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);

disp(['EEG 데이터 불러오기 완료: ' fullfile(file_path, file_name)]);