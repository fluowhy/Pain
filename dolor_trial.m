% dolor
clear all
fs = 2048;
l = 2048;
channels = [7 8 9];
dataT = zeros(40, 2048);
trial_info = zeros(40, 2);
m = input('num');
name = strcat('SA00', int2str(m), '_');
named = strcat(name, 'D');
n = 9;

for k = 1:2
    r = load(strcat(named, int2str(k)));
    for j=1:40
        data = r.data.trialinfo(3*j - 2, 1:2);     
        trial_info(j, :) = data;           
    end
save(strcat('C:\Users\mauricio.romero.j\Downloads\SA00', int2str(m), '\', named, int2str(k),'.txt'), 'trial_info', '-ascii')
end   


dataT = zeros(20, 2048);
trial_info = zeros(20, 2);
namea = strcat(name, 'A');

for k = 1:3
    r = load(strcat(namea, int2str(k)));
    p = 20;
    if k==1 & m==3
        p = 19;
    end
    for j=1:p
        data = r.data.trialinfo(3*j - 2, 1:2);      
        trial_info(j, :) = data;
    end 
save(strcat('C:\Users\mauricio.romero.j\Downloads\SA00', int2str(m), '\', namea, int2str(k), '.txt'), 'trial_info', '-ascii')
end
