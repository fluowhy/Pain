% dolor
clear all
fs = 2048;
l = 2048;
channels = [7 8 9];
dataT = zeros(60, 2048);
name = 'SA009_D';
n = 9;
for i=[7, 8, 9]
    for k = 1:2
        r = load(strcat(name, int2str(k)));
        for j=1:40
            data = r.data.trial{1, 3*j - 2}(i,:);
            data = data(1:2048);
            dataT(j,:)=data;            
        end
        save(strcat(name, int2str(k), '_', int2str(i), '.txt'), 'dataT', '-ascii')
    end   
end

name = 'SA009_A';
for i=[7, 8, 9]
    for k = 1:3
        r = load(strcat(name, int2str(k)));
        for j=1:20
            data = r.data.trial{1, 3*j - 2}(i,:);  
            data = data(1:2048);
            dataT(j,:) = data;
        end 
    save(strcat(name, int2str(k), '_', int2str(i), '.txt'), 'dataT', '-ascii')
    end
end
    







%plot(data(1,:));
%hold on;
%plot(data(2,:));
%hold on;
%plot(data(3,:));
f = fs*(0:(l/2))/l;
ft1 = abs(fft(data(1,:))/l);
