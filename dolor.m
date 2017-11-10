% dolor
clear all
fs = 2048;
l = 2048;
channels = [7 8 9];
dataT = zeros(40, 2048);
m = input('num');
name = strcat('SA00', int2str(m), '_');
named = strcat(name, 'D');
n = 9;
for i=[7, 8, 9]
    for k = 1:2
        r = load(strcat(named, int2str(k)));
        for j=1:40
            data = r.data.trial{1, 3*j - 2}(i,:);
            data = data(1:2048);
            dataT(j,:)=data;            
        end
    save(strcat('C:\Users\mauricio.romero.j\Downloads\SA00', int2str(m), '\', named, int2str(k), '_', int2str(i), '.txt'), 'dataT', '-ascii')
    end   
end

dataT = zeros(20, 2048);
namea = strcat(name, 'A');
for i=[7, 8, 9]
    for k = 1:3
        r = load(strcat(namea, int2str(k)));
        p = 20;
        if k==1 & m==3
            p = 19;
        end
        p
        for j=1:p
            data = r.data.trial{1, 3*j - 2}(i,:);  
            data = data(1:2048);
            dataT(j,:) = data;
        end 
    save(strcat('C:\Users\mauricio.romero.j\Downloads\SA00', int2str(m), '\', namea, int2str(k), '_', int2str(i), '.txt'), 'dataT', '-ascii')
    end
end

%plot(data(1,:));
%hold on;
%plot(data(2,:));
%hold on;
%plot(data(3,:));
%f = fs*(0:(l/2))/l;
%ft1 = abs(fft(data(1,:))/l);