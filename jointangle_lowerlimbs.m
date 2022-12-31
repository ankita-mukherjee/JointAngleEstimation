%lower body joint angle calculation

clear
clc

data = csvread ('C:\Users\cmhughes\Documents\GitHub\deeplearning\dl009_kneeflexviconfiles\dl009_kneeflex_020.csv',5,2);

%assign columns to data
ASIx = data(:,1);
ASIy = data(:,2);
ASIz = data(:,3);
KNEx = data(:,4);
KNEy = data(:,5);
KNEz = data(:,6);
ANKx = data(:,7);
ANKy = data(:,8);
ANKz = data(:,9);

%Lengths
hipknee_length = (sqrt(((ASIx - KNEx).^2) + ((ASIz - KNEz).^2)));
hipank_length = (sqrt(((ASIx - ANKx).^2) + ((ASIz - ANKz).^2)));
kneank_length = (sqrt(((KNEx - ANKx).^2) + ((KNEz - ANKz).^2)));

%Lengths y and z dimensions
% hipknee_length = (sqrt(((ASIy - KNEy).^2) + ((ASIz - KNEz).^2)));
% hipank_length = (sqrt(((ASIy - ANKy).^2) + ((ASIz - ANKz).^2)));
% kneank_length = (sqrt(((KNEy - ANKy).^2) + ((KNEz - ANKz).^2)));

% Hip
cASIa = (hipknee_length.^2 + hipank_length.^2 - kneank_length.^2) ./ (2 .* hipknee_length .* hipank_length);
hipangle = acos(cASIa);
thetahip = hipangle*180/pi;
thetahip1 = real(thetahip);

% Knee
cKNEa = (hipknee_length.^2 + kneank_length.^2 - hipank_length.^2) ./ (2 .* hipknee_length .* kneank_length);
kneeangle = acos(cKNEa);
thetaknee = kneeangle*180/pi;

%%%%%%%%% %Y and Z
figure (999)
subplot (1,2,1)
line ([ASIx, KNEx],[ASIz, KNEz], 'Color','b');
hold on; line([ASIx, ANKx], [ASIz, ANKz], 'Color', 'b');
hold on;line([KNEx, ANKx], [KNEz, ANKz], 'Color', 'b'); 

subplot (1,2,2)
line ([ASIy, KNEy],[ASIz, KNEz], 'Color','b');
hold on; line([ASIy, ANKy], [ASIz, ANKz], 'Color', 'b');
hold on;line([KNEy, ANKy], [KNEz, ANKz], 'Color', 'b'); 

%writematrix(thetahip, 'dl009_hipextoutput_024hip.csv');
writematrix(thetaknee, 'dl009_kneeflexoutput_020knee.csv');