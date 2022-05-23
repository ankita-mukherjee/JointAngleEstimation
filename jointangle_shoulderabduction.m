clear
clc

data = csvread ('C:\Users\cmhughes\Documents\GitHub\deeplearning\dl009_hipflexviconfiles\dl009_hipflex_001.csv',5,2);

%assign columns to data
ASIx = data(:,1);
ASIy = data(:,2);
ASIz = data(:,3);
SHOx = data(:,4);
SHOy = data(:,5);
SHOz = data(:,6);
ELBx = data(:,7);
ELBy = data(:,8);
ELBz = data(:,9);
WRIx = data(:,10);
WRIy = data(:,11);
WRIz = data(:,12);

%Lengths
shohip_length = (sqrt(((SHOx - ASIx).^2) + ((SHOz - ASIz).^2)));
shoelb_length = (sqrt(((SHOx - ELBx).^2) + ((SHOz - ELBz).^2)));
showri_length = (sqrt(((SHOx - WRIx).^2) + ((SHOz - WRIz).^2)));
elbwri_length = (sqrt(((ELBx - WRIx).^2) + ((ELBz - WRIz).^2)));
elbhip_length = (sqrt(((ELBx - ASIx).^2) + ((ELBz - ASIz).^2)));

%Lengths y and z dimensions
% shohip_length = (sqrt(((SHOy - ASIy).^2) + ((SHOz - ASIz).^2)));
% shoelb_length = (sqrt(((SHOy - ELBy).^2) + ((SHOz - ELBz).^2)));
% showri_length = (sqrt(((SHOy - WRIy).^2) + ((SHOz - WRIz).^2)));
% elbwri_length = (sqrt(((ELBy - WRIy).^2) + ((ELBz - WRIz).^2)));
% elbhip_length = (sqrt(((ELBy - ASIy).^2) + ((ELBz - ASIz).^2))); 

% Shoulder
cSHOa = (shohip_length.^2 + shoelb_length.^2 - elbhip_length.^2) ./ (2 .* shohip_length .* shoelb_length);
shoulderangle = acos(cSHOa);
thetashoulder = shoulderangle*180/pi;
thetashoulder1 = real(thetashoulder);

% Elbow
cELBa = (shoelb_length.^2 + elbwri_length.^2 - showri_length.^2) ./ (2 .* shoelb_length .* elbwri_length);
elbowangle = acos(cELBa);
thetaelbow = elbowangle*180/pi;

%%%%%%%%% %Y and Z
figure (999)
subplot (1,2,1)
line ([SHOx,ASIx],[SHOz,ASIz], 'Color','b');
hold on; line([ELBx,WRIx], [ELBz, WRIz], 'Color', 'b');
hold on;line([SHOx,ELBx], [SHOz, ELBz], 'Color', 'b'); 

subplot (1,2,2)
line ([SHOy,ASIy],[SHOz,ASIz], 'Color','b');
hold on; line([ELBy,WRIy], [ELBz, WRIz], 'Color', 'b');
hold on;line([SHOy,ELBy], [SHOz, ELBz], 'Color', 'b'); 

writematrix(thetashoulder, 'dl009_hipflexoutput_001hip.csv');
%writematrix(thetaelbow, 'dl009_elbflexoutput_020elbow.csv');